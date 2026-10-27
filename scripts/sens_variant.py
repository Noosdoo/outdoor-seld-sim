# -*- coding: utf-8 -*-
"""sens_variant.py — 〔仮〕の値の感度を「学習し直さず・推論だけ」で測るための変種の束（2026-10-21 Fable・本人 13:57「それでお願いします」・23:39「お願いします」）。

考え方: 評価側の小さな束（既定 200 本）を 2 つ描く。「基準」= 今の設計のまま、「変種」= 定数を 1 つ変える（_sens_overrides.py）。場面と種（seed）は同じ。
固定したモデル（確定評価で使う ckpt）で両方を推論し、同じ採点器（統一採点器・知覚層）で測って差を見る。封印の eval v3 には触れない・学習の置き場にも書かない。
いつ: 確定評価が 1 回出たあと（本人 10/21 13:57）。どの値を測るかは本人が決める（この道具は仕組みだけ）。

使い方:
  1. 束の plan を作る（手元・学習 plan の val の行から層別に n 本を写し、新しい番号 eval_syn_clip9NNNN と新しい seed を振る）:
     python scripts/sens_variant.py plan --tag base --src-plan out/dataset_outdoor_siren_v2final/plan/assignment_v2final.csv --n 200 --seed 777
     python scripts/sens_variant.py plan --tag bic20 --copy-from base            # 変種は基準と同じ plan（同じ場面・同じ seed）
  2. 上書きの JSON を書く（基準は --set 無しで空の記録を置く・変種は定数を 1 つ）:
     python scripts/sens_variant.py overrides --tag base
     python scripts/sens_variant.py overrides --tag bic20 --set step11_v22_render.BIC_SLOPE_HIGH=20 --note "自転車の走行音の傾きを 20 に"
  3. 配備してサーバで描く → 推論 → 採点（server_sde/sens_variant.sbatch・SENS_TAG と V2FINAL_CKPT を渡す）:
     scp -r out/dataset_outdoor_siren_v2final_eval_sens_<tag> is-server:~/research/outdoor_seld_e2e/out/
     ssh is-server 'cd ~/research/outdoor_seld_e2e && SENS_TAG=<tag> V2FINAL_CKPT=<ckpt> sbatch server_sde/sens_variant.sbatch'
  4. 取り寄せて差を見る: python scripts/sens_variant.py fetch --tag base; python scripts/sens_variant.py fetch --tag bic20
                         python scripts/sens_variant.py compare --base base --var bic20   （同じ ckpt・同じ plan かを照合してから、知覚層の json と統一採点器の丸め前の行を並べる）
置き場: out/dataset_outdoor_siren_v2final_eval_sens_<tag>/（plan/assignment_v2final.csv・SENS_PLAN.json・SENS_OVERRIDES.json・サーバで foa／metadata／metadata_dist・unified/・SENS_META.json）
  python scripts/sens_variant.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DS_PREFIX = "dataset_outdoor_siren_v2final_eval_sens_"
CLIP_BASE = 90000                                     # 番号 eval_syn_clip90001〜（封印の eval v3 は 00001〜06000・衝突しない）
SERVER = "is-server"
SERVER_E2E = "~/research/outdoor_seld_e2e"


def ds_dir(tag: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
        raise SystemExit(f"tag は英数字と _ だけ: {tag}")
    return ROOT / "out" / f"{DS_PREFIX}{tag}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list[dict], cols: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def stratified_sample(rows: list[dict], n: int, seed: int, key: str = "scenario") -> list[dict]:
    """層（scenario）の割合に比例して n 本を選ぶ（各層 1 本以上・余りは大きい層から）。同じ seed なら同じ選び方。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key, "")].append(r)
    total = len(rows)
    quota = {k: max(1, int(round(n * len(v) / total))) for k, v in groups.items()}
    while sum(quota.values()) > n:
        k = max(quota, key=lambda q: quota[q]); quota[k] -= 1
    while sum(quota.values()) < n:
        k = max(groups, key=lambda q: len(groups[q]) / max(quota[q], 1)); quota[k] += 1
    out = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["clip_id"])
        rng.shuffle(g)
        out.extend(g[:quota[k]])
    return out


def cmd_plan(a) -> int:
    d = ds_dir(a.tag); plan = d / "plan" / "assignment_v2final.csv"
    if plan.exists():
        print(f"NG 既に plan がある（{plan}）。作り直すなら別の tag"); return 1
    if a.copy_from:
        src = ds_dir(a.copy_from) / "plan" / "assignment_v2final.csv"
        if not src.is_file():
            print(f"NG 写す元の plan が無い: {src}"); return 1
        rows = read_rows(src); cols = list(rows[0].keys())
        write_rows(plan, rows, cols)
        meta = {"tag": a.tag, "copy_from": a.copy_from, "src_plan_sha256": sha256(src), "n": len(rows)}
    else:
        if not a.src_plan or not Path(a.src_plan).is_file():
            print("NG --src-plan（学習の plan csv・val の行を使う）が要る"); return 1
        src_rows = [r for r in read_rows(Path(a.src_plan)) if r.get("split") == "val"]
        if len(src_rows) < a.n:
            print(f"NG val の行が足りない（{len(src_rows)} < {a.n}）"); return 1
        picked = stratified_sample(src_rows, a.n, a.seed)
        cols = list(src_rows[0].keys())
        rows = []
        for i, r in enumerate(picked, start=1):
            q = dict(r)
            q["clip_id"] = f"eval_syn_clip{CLIP_BASE + i:05d}"
            q["split"] = "eval"
            q["seed"] = str(a.seed * 100000 + i)           # 新しい種（元の val の音とは別の音になる・基準と変種は同じ種）
            if "twin_of" in q:
                q["twin_of"] = ""
            rows.append(q)
        write_rows(plan, rows, cols)
        meta = {"tag": a.tag, "src_plan": str(a.src_plan), "src_plan_sha256": sha256(Path(a.src_plan)), "n": len(rows), "seed": a.seed,
                "strata": dict(Counter(r.get("scenario", "") for r in rows)), "note": "val の行から層別に写し、番号 eval_syn_clip9NNNN・新しい seed。封印の eval v3 とは別"}
    meta["plan_sha256"] = sha256(plan)
    (d / "SENS_PLAN.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"OK plan {len(rows)} 本 → {plan}（sha {meta['plan_sha256'][:12]}）")
    return 0


def _parse_value(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _from_import_warnings(key: str) -> list[str]:
    """`from <module> import ATTR` で写し取っている所があれば、setattr の上書きが効かない（警告。定数は呼ぶときに module の名前で読むこと）。"""
    modname, attr = key.rsplit(".", 1)
    last = modname.rsplit(".", 1)[-1]
    pat = re.compile(rf"^\s*from\s+\S*{re.escape(last)}\s+import\s+(?:\([^)]*\)|[^\n]*)", re.M)
    hits = []
    for p in sorted(list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src" / "outdoor_seld").glob("*.py"))):
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in pat.finditer(t):
            if re.search(rf"\b{re.escape(attr)}\b", m.group(0)):
                hits.append(f"{p.relative_to(ROOT).as_posix()}: {m.group(0).strip()[:80]}")
    return hits


def cmd_overrides(a) -> int:
    d = ds_dir(a.tag)
    if not (d / "plan" / "assignment_v2final.csv").is_file():
        print(f"NG 先に plan を作る（{d}）"); return 1
    ov = {}
    for item in a.set:
        if "=" not in item:
            print(f"NG --set は module.ATTR=値: {item}"); return 1
        k, v = item.split("=", 1); ov[k.strip()] = _parse_value(v.strip())
    sys.path.insert(0, str(ROOT / "scripts"))
    import _sens_overrides as S  # noqa: E402
    try:
        rec = S.apply(ov)                                   # 手元で当ててみて（種類と名前の検査）、元に戻す
        for k, r in rec.items():
            modname, attr = k.rsplit(".", 1)
            setattr(sys.modules[modname], attr, r["old"])
    except S.OverrideError as e:
        print(f"NG 上書きが当てられない: {e}"); return 1
    warns = [w for k in ov for w in _from_import_warnings(k)]
    J = {"tag": a.tag, "note": a.note, "overrides": ov, "checked_old_values": {k: r["old"] for k, r in rec.items()}, "from_import_warnings": warns}
    (d / "SENS_OVERRIDES.json").write_text(json.dumps(J, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"OK 上書き {len(ov)} 件 → {d / 'SENS_OVERRIDES.json'}" + (": " + "・".join(f"{k}: {r['old']!r} → {r['new']!r}" for k, r in rec.items()) if rec else "（基準 = 上書き無し）"))
    for w in warns:
        print(f"注意 from-import で写している所がある（そこでは上書きが効かない）: {w}")
    return 0


def _raw_rows(md: Path) -> dict[str, list[str]]:
    """統一採点器の md の『## 丸め前』の節の行（| <名前> | a/b = x% | … |）を名前 → セルの list で返す（無ければ空）。"""
    out: dict[str, list[str]] = {}
    if not md.is_file():
        return out
    inside = False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.startswith("## 丸め前"); continue
        if inside and line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| 予測"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0]:
                out.setdefault(cells[0], cells[1:])
    return out


def _meta(tag: str) -> dict | None:
    p = ds_dir(tag) / "SENS_META.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cmd_compare(a) -> int:
    base, var = ds_dir(a.base), ds_dir(a.var)
    lines = [f"# 感度: {a.var} − {a.base}（推論だけ・同じ場面と種・固定したモデル）", ""]
    ov = var / "SENS_OVERRIDES.json"
    if ov.is_file():
        J = json.loads(ov.read_text(encoding="utf-8"))
        lines += ["- 変えた定数: " + ("・".join(f"{k} = {v!r}（元 {J.get('checked_old_values', {}).get(k)!r}）" for k, v in J.get("overrides", {}).items()) or "（無し）"), f"- 注: {J.get('note', '')}", ""]
    mb, mv = _meta(a.base), _meta(a.var)
    rc = 0
    if mb and mv:
        same_ck = mb.get("ckpt_sha256") == mv.get("ckpt_sha256"); same_plan = mb.get("plan_sha256") == mv.get("plan_sha256"); same_dt = mb.get("dist_table_sha256") == mv.get("dist_table_sha256")
        lines += [f"- 照合: ckpt {'同じ' if same_ck else '違う ← NG'}（{str(mb.get('ckpt_sha256'))[:12]}）・plan {'同じ' if same_plan else '違う ← NG'}・補正表 {'同じ' if same_dt else '違う ← NG'}・N {mb.get('n')}／{mv.get('n')}", ""]
        if not (same_ck and same_plan and same_dt):
            rc = 1
    else:
        lines += ["- 照合: SENS_META.json がまだ無い（fetch してから）", ""]
    pj_b, pj_v = base / "unified/perception_sens.json", var / "unified/perception_sens.json"
    if pj_b.is_file() and pj_v.is_file():
        B = json.loads(pj_b.read_text(encoding="utf-8")).get("v2final", {}); V = json.loads(pj_v.read_text(encoding="utf-8")).get("v2final", {})
        lines += ["## 知覚層（perception_sens.json・v2final・差 = 変種 − 基準）", ""]
        for k in B:
            if k not in V:
                continue
            b, v = B[k], V[k]
            if isinstance(b, dict) and isinstance(v, dict):
                parts = []
                for sk in b:
                    if isinstance(b.get(sk), (int, float)) and isinstance(v.get(sk), (int, float)) and not isinstance(b[sk], bool):
                        parts.append(f"{sk} {b[sk]:.4g}→{v[sk]:.4g}（{v[sk] - b[sk]:+.4g}）")
                if parts:
                    lines.append(f"- {k}: " + "・".join(parts))
            elif isinstance(b, (int, float)) and isinstance(v, (int, float)):
                lines.append(f"- {k}: {b:.4g}→{v:.4g}（{v - b:+.4g}）")
        lines.append("")
    rb, rv = _raw_rows(base / "unified/unified_sens.md"), _raw_rows(var / "unified/unified_sens.md")
    if rb and rv:
        lines += ["## 知らせ（統一採点器の丸め前の行・基準 → 変種）", ""]
        for name in rb:
            if name in rv:
                lines.append(f"- {name}: " + " | ".join(f"{x} → {y}" if x != y else x for x, y in zip(rb[name], rv[name])))
        lines.append("")
    if not (pj_b.is_file() and pj_v.is_file()) and not (rb and rv):
        lines.append("（採点の出力がまだ無い: server_sde/sens_variant.sbatch を両方の tag で回し、fetch してから）")
    out = var / f"SENS_COMPARE_vs_{a.base}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"-> {out}")
    return rc


def cmd_fetch(a) -> int:
    d = ds_dir(a.tag); d.mkdir(parents=True, exist_ok=True)
    name = d.name
    cmds = [["scp", "-q", "-r", f"{SERVER}:{SERVER_E2E}/out/{name}/unified", str(d)],
            ["scp", "-q", f"{SERVER}:{SERVER_E2E}/out/{name}/SENS_META.json", f"{SERVER}:{SERVER_E2E}/out/{name}/GEN_VERSIONS.txt", f"{SERVER}:{SERVER_E2E}/out/{name}/REGEN_CHECK.md", str(d)]]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            print("NG " + " ".join(c) + "\n" + r.stderr.strip()); return 1
    m = _meta(a.tag)
    print(f"OK {name}: unified/・SENS_META.json・GEN_VERSIONS.txt・REGEN_CHECK.md を取り寄せた" + (f"（N {m.get('n')}・ckpt {m.get('ckpt_sha256', '')[:12]}・job {m.get('job')}）" if m else ""))
    return 0


def selftest() -> int:
    import shutil
    import tempfile
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="sens_"))
    cols = ["clip_id", "split", "scenario", "mix", "seed", "mic_z", "twin_of"]
    rows = []
    for i in range(1, 601):
        sc = "v11core" if i % 3 else ("v19mix" if i % 2 else "v12bike")
        rows.append({"clip_id": f"val_syn_clip{i:05d}", "split": "val", "scenario": sc, "mix": "a", "seed": str(1000 + i), "mic_z": "1.6", "twin_of": ""})
    for i in range(601, 700):
        rows.append({"clip_id": f"train_syn_clip{i:05d}", "split": "train", "scenario": "v11core", "mix": "a", "seed": str(i), "mic_z": "1.6", "twin_of": ""})
    src = tmp / "plan.csv"; write_rows(src, rows, cols)
    s1 = stratified_sample([r for r in rows if r["split"] == "val"], 60, 7); s2 = stratified_sample([r for r in rows if r["split"] == "val"], 60, 7)
    c = Counter(r["scenario"] for r in s1)
    ok.append(("層別の抽出: 60 本・割合に比例（v11core 40・v19mix 10・v12bike 10）・同じ seed なら同じ", len(s1) == 60 and c == {"v11core": 40, "v19mix": 10, "v12bike": 10} and [r["clip_id"] for r in s1] == [r["clip_id"] for r in s2]))
    tag = f"selftest_{tmp.name[-6:]}"; tag2 = tag + "_v"
    d, d2 = ds_dir(tag), ds_dir(tag2)
    try:
        rc = main(["plan", "--tag", tag, "--src-plan", str(src), "--n", "30", "--seed", "5"])
        P = read_rows(d / "plan" / "assignment_v2final.csv")
        ok.append(("plan: 30 本・番号 eval_syn_clip90001〜・split=eval・seed は新しい・SENS_PLAN.json",
                   rc == 0 and len(P) == 30 and P[0]["clip_id"] == "eval_syn_clip90001" and all(r["split"] == "eval" for r in P) and P[0]["seed"] == str(5 * 100000 + 1)
                   and (d / "SENS_PLAN.json").is_file()))
        ok.append(("同じ tag の plan は上書きしない（1）", main(["plan", "--tag", tag, "--src-plan", str(src), "--n", "30"]) == 1))
        rc3 = main(["plan", "--tag", tag2, "--copy-from", tag])
        ok.append(("--copy-from で基準と同じ plan（同じ場面・同じ seed）", rc3 == 0 and sha256(d2 / "plan/assignment_v2final.csv") == sha256(d / "plan/assignment_v2final.csv")))
        rc0 = main(["overrides", "--tag", tag])
        J0 = json.loads((d / "SENS_OVERRIDES.json").read_text(encoding="utf-8"))
        ok.append(("overrides: 基準は --set 無しで空の記録", rc0 == 0 and J0["overrides"] == {}))
        rc4 = main(["overrides", "--tag", tag2, "--set", "outdoor_seld.avas_v26.AVAS_L2M_DBA=[50, 70]", "--note", "試験"])
        J = json.loads((d2 / "SENS_OVERRIDES.json").read_text(encoding="utf-8"))
        ok.append(("overrides: 当てられる定数なら JSON に書く（元の値も記録・from-import の警告は 0）", rc4 == 0 and J["overrides"]["outdoor_seld.avas_v26.AVAS_L2M_DBA"] == [50, 70]
                   and J["checked_old_values"]["outdoor_seld.avas_v26.AVAS_L2M_DBA"] == [46.0, 70.0] and J["from_import_warnings"] == []))
        ok.append(("overrides: 無い定数は 1", main(["overrides", "--tag", tag2, "--set", "outdoor_seld.avas_v26.NO_SUCH=1"]) == 1))
        # compare: 採点の出力を模した json と md・SENS_META（同じ ckpt／違う ckpt）
        for dd, ra, az in ((d, 90.0, 3.0), (d2, 85.0, 3.5)):
            (dd / "unified").mkdir(exist_ok=True)
            (dd / "unified/perception_sens.json").write_text(json.dumps({"v2final": {"距離クラス": {"n_gt": 10, "recall_any": ra, "az_med": az}, "方向:前": {"recall20": ra - 1}}}), encoding="utf-8")
            (dd / "unified/unified_sens.md").write_text("# x\n\n## 表\n\n| 予測 | a |\n| --- | --- |\n| v2final | 1/2 = 50% |\n\n## 丸め前（判定用）\n\n| 予測 | 強到達 n_strong/n_crit | 至近 |\n| --- | --- | --- |\n"
                                                        f"| v2final | {int(ra)}/100 = {ra:.4f}% | 7/9 |\n\n## 次\n", encoding="utf-8")
        for dd in (d, d2):
            (dd / "SENS_META.json").write_text(json.dumps({"ckpt_sha256": "aaa", "plan_sha256": sha256(dd / "plan/assignment_v2final.csv"), "dist_table_sha256": None, "n": 30}), encoding="utf-8")
        rc6 = main(["compare", "--base", tag, "--var", tag2])
        md = (d2 / f"SENS_COMPARE_vs_{tag}.md").read_text(encoding="utf-8")
        ok.append(("compare: 同じ ckpt・plan → 0・知覚層の差（recall_any 90→85 −5・az_med +0.5）と丸め前の行（90/100 → 85/100）", rc6 == 0 and "ckpt 同じ" in md and "recall_any 90→85（-5）" in md and "az_med 3→3.5（+0.5）" in md
                   and "90/100 = 90.0000% → 85/100 = 85.0000%" in md and "7/9" in md))
        (d2 / "SENS_META.json").write_text(json.dumps({"ckpt_sha256": "bbb", "plan_sha256": sha256(d2 / "plan/assignment_v2final.csv"), "dist_table_sha256": None, "n": 30}), encoding="utf-8")
        rc7 = main(["compare", "--base", tag, "--var", tag2])
        ok.append(("compare: ckpt が違えば 1（NG と書く）", rc7 == 1 and "違う ← NG" in (d2 / f"SENS_COMPARE_vs_{tag}.md").read_text(encoding="utf-8")))
    finally:
        shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d2, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p1 = sub.add_parser("plan"); p1.add_argument("--tag", required=True); p1.add_argument("--src-plan", default=None); p1.add_argument("--n", type=int, default=200); p1.add_argument("--seed", type=int, default=777); p1.add_argument("--copy-from", default=None)
    p2 = sub.add_parser("overrides"); p2.add_argument("--tag", required=True); p2.add_argument("--set", action="append", default=[]); p2.add_argument("--note", default="")
    p3 = sub.add_parser("compare"); p3.add_argument("--base", required=True); p3.add_argument("--var", required=True)
    p4 = sub.add_parser("fetch"); p4.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "overrides":
        return cmd_overrides(a)
    if a.cmd == "compare":
        return cmd_compare(a)
    if a.cmd == "fetch":
        return cmd_fetch(a)
    ap.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
