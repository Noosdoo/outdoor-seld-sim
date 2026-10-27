# -*- coding: utf-8 -*-
"""v30 の開発の比べ（Fable 仕様（本体）§4b・本人 10/26 14:15「デモの評価が治ったか・既存手法を上回り平等にできているか」・2026-10-27 Opus・本人 14:58「7 お願いします」）。

学習し直しのあと、顔ぶれ（主・s1・s2・アブレーション 4 本・相手 B・相手 D）の予測を、同じ採点器・同じ条件で並べる。本文には載せない（開発の記録）。
**評価の本（本番の学内の実録・eval v3・eval v4）は入力に取らない**（置き場の名前で止める）。読むのは合成の val と開発用の実録（学内のデモの組 19 本・家の 13 本）の予測だけ。

  1. 合成の val（12,420 本）: 統一採点器 `_score_unified.py`（--events = v30 の事象台帳・知らせの表・距離の誤差の帯・警告音の届いた行）と知覚層 `_perception_eval.py`
     （クラス別の検出・向き・距離）。どちらも `--confusion` で取り違えの表と本ごとの種類の正答率（参考・本人 10/27 (a)(b)）。相手 D は距離を出さないので知覚層だけ（§4b）。
  2. 学内のデモの組（`demo_field_unused_score.py`・取り違えの表と本ごとの種類つき）と家の 13 本（`demo_bike_check_score.py`）は別の表（足し合わせない・本人 10/27 00:01）。
  3. まとめ: <out>/index.md（顔ぶれごとの主な数・ファイルへの道・命令と予測の sha）と index.json。表は使わず 1 行ずつ（md では表を使わない・本人 9/26）。
入力: --config <json>（下の形）。予測の無い顔ぶれ・置き場は飛ばして「無し」と書く。--dry は命令を印字するだけ。
  {"date": "2026-11-xx",
   "val": {"plan": "<manifest_val_syn_v2final.csv>", "meta": "<v30 の val の正解の置き場>", "events": "<_event_ledger_v30.py の台帳 csv・無ければ省く>"},
   "demo": {"infer": "out/_demo_field_unused_149/infer"},
   "home": {"dir": "out/_demo_bike_check_149"},
   "members": [{"label": "main", "kind": "main", "val": "<val の予測>", "demo": "<デモの組の予測>", "home": "<家の 13 本の予測>"}, …]}
  kind = main・seed・ablation・B・D（D は統一採点器に通さない）。
使い方: python scripts/v30_dev_compare.py --config <json> [--out out/v30_dev_compare/<date>] [--dry]
試験: scripts/test_v30_dev_compare.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KINDS = ("main", "seed", "ablation", "B", "D")
FORBIDDEN = re.compile(r"eval_field_|realsmoke|eval_v3|eval_v4|_eval_v4|dataset_outdoor_siren_v2final_eval|field_eval|raw[\\/]")   # 評価の本の置き場・名前


class Stop(RuntimeError):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_paths(cfg: dict) -> None:
    """評価の本（本番の学内の実録・eval v3・v4）を入力に取らない（本人 10/27 15:01）。"""
    paths = [cfg.get("val", {}).get(k, "") for k in ("plan", "meta", "events")] + [cfg.get("demo", {}).get("infer", ""), cfg.get("home", {}).get("dir", "")]
    for m in cfg.get("members", []):
        paths += [m.get(k, "") or "" for k in ("val", "demo", "home")]
    bad = [p for p in paths if p and FORBIDDEN.search(str(p).replace("\\", "/"))]
    if bad:
        raise Stop("評価の本の置き場・名前は入力に取らない（本番の学内の実録・eval v3・v4）: " + ", ".join(bad))
    for m in cfg.get("members", []):
        if m.get("kind") not in KINDS:
            raise Stop(f"kind は {KINDS} のどれか: {m.get('label')} {m.get('kind')}")


def plan_commands(cfg: dict, out: Path) -> list:
    """[(顔ぶれ, 何, 命令, 出力の主なファイル)]。予測の無いものは入れない。"""
    py = sys.executable
    v, d, h = cfg.get("val", {}), cfg.get("demo", {}), cfg.get("home", {})
    cmds = []
    for m in cfg.get("members", []):
        lb = m["label"]
        if m.get("val"):
            if m.get("kind") != "D":
                c = [py, str(E2E / "scripts/_score_unified.py"), str(out / f"val_unified_{lb}.md"), "--plan", v["plan"], "--split", "val", "--meta", v["meta"],
                     "--confusion", str(out / f"val_unified_confusion_{lb}.json")]
                if v.get("events"):
                    c += ["--events", v["events"]]
                cmds.append((lb, "val 統一採点器", c + [f"{lb}={m['val']}"], out / f"val_unified_{lb}.md"))
            c = [py, str(E2E / "scripts/_perception_eval.py"), str(out / f"val_perception_{lb}.md"), "--plan", v["plan"], "--split", "val", "--meta", v["meta"],
                 "--json", str(out / f"val_perception_{lb}.json"), "--confusion", str(out / f"val_perception_confusion_{lb}.json"), f"{lb}={m['val']}"]
            cmds.append((lb, "val 知覚層", c, out / f"val_perception_{lb}.json"))
        if m.get("demo") and d.get("infer"):
            cmds.append((lb, "学内のデモの組", [py, str(E2E / "scripts/demo_field_unused_score.py"), "--infer", d["infer"], "--pred", m["demo"],
                                             "--out", str(out / f"demo_{lb}.json"), "--label", lb], out / f"demo_{lb}.json"))
        if m.get("home") and h.get("dir"):
            cmds.append((lb, "家の 13 本", [py, str(E2E / "scripts/demo_bike_check_score.py"), "--pred", m["home"], "--dir", h["dir"], "--label", lb],
                         out / f"home_{lb}.txt"))
    return cmds


def summarize_member(out: Path, lb: str) -> list:
    """顔ぶれ 1 つの主な数（表を使わない行）。"""
    L = []
    pj = out / f"val_perception_{lb}.json"
    if pj.is_file():
        r = json.loads(pj.read_text(encoding="utf-8")).get(lb, {})
        parts = []
        for k, s in r.items():
            if k.startswith("cls:") and s.get("n_gt"):
                parts.append(f"{k[4:]} 検出(20°) {s.get('recall20', 0):.1f}%・F1(20°) {s.get('f1_20', 0):.1f}" + (f"・向き {s['az_med']:.0f}°" if "az_med" in s else "")
                             + (f"・距離の相対誤差 {s['rel_med']:.0f}%" if "rel_med" in s else ""))
        if parts:
            L.append("- val 知覚層: " + "／".join(parts))
    cj = out / f"val_perception_confusion_{lb}.json"
    if cj.is_file():
        ta = json.loads(cj.read_text(encoding="utf-8")).get(lb, {}).get("take_accuracy", {})
        if ta.get("n"):
            L.append(f"- val 本ごとの種類（参考）: {ta['correct']}/{ta['n']}（{100 * ta['acc']:.1f}%）")
    dj = out / f"demo_{lb}.json"
    if dj.is_file():
        r = json.loads(dj.read_text(encoding="utf-8"))
        h = r.get("head", {}); s = r.get("summary", {}); ta = r.get("take_accuracy", {})
        L.append(f"- 学内のデモの組: 本 {h.get('n_takes')}・警告音 届いた {h.get('n_success')}/{h.get('n_scored_events')}（部分観察 {h.get('n_partial')}）・"
                 + "・".join(f"{k} {v.get('frames')}" for k, v in s.items()) + (f"・本ごとの種類 {ta.get('correct')}/{ta.get('n')}" if ta.get("n") else ""))
    ht = out / f"home_{lb}.txt"
    if ht.is_file():
        lines = [x for x in ht.read_text(encoding="utf-8").splitlines() if x.strip()][:3]
        L.append("- 家の 13 本: " + " ｜ ".join(lines))
    for f, name in ((out / f"val_unified_{lb}.md", "val 統一採点器（知らせの表）"), (out / f"val_unified_confusion_{lb}.md", "取り違えの表（統一）"),
                    (out / f"val_perception_confusion_{lb}.md", "取り違えの表（知覚層）")):
        if f.is_file():
            L.append(f"- {name}: `{f.name}`")
    return L or ["- （予測が無い・この顔ぶれは無し）"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    try:
        check_paths(cfg)
    except Stop as e:
        print("止める:", e); return 2
    out = a.out or (E2E / "out" / "v30_dev_compare" / cfg.get("date", "undated"))
    cmds = plan_commands(cfg, out)
    if a.dry:
        for lb, what, c, _ in cmds:
            print(f"[dry] {lb}・{what}: {' '.join(str(x) for x in c)}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    log, rc_all = [], 0
    for lb, what, c, main_out in cmds:
        r = subprocess.run([str(x) for x in c], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(E2E))
        if what == "家の 13 本":
            main_out.write_text(r.stdout, encoding="utf-8")
        log.append({"member": lb, "what": what, "rc": r.returncode, "cmd": [str(x) for x in c]})
        rc_all |= r.returncode != 0
        print(f"{lb}・{what}: rc {r.returncode}" + ("" if r.returncode == 0 else "\n" + (r.stdout + r.stderr)[-800:]))
    members = cfg.get("members", [])
    idx = [f"# v30 の開発の比べ（{cfg.get('date', '')}・§4b・開発の記録・本文には載せない）", "",
           "- 評価の本（本番の学内の実録・eval v3・v4）は入れていない。学内のデモの組と家の 13 本は別の表（足し合わせない）。取り違えの表と本ごとの種類は参考（主の指標は変えない）。",
           f"- 顔ぶれ: " + "・".join(f"{m['label']}（{m['kind']}）" for m in members), ""]
    for m in members:
        idx += [f"## {m['label']}（{m['kind']}）", ""] + summarize_member(out, m["label"]) + [""]
    idx += ["## 予測の sha と命令", ""]
    for m in members:
        for k in ("val", "demo", "home"):
            p = m.get(k)
            if p and Path(p).is_file():
                idx.append(f"- {m['label']}・{k}: `{p}`（sha {sha(Path(p))[:16]}）")
    idx += [f"- {e['member']}・{e['what']}（rc {e['rc']}）: `{' '.join(e['cmd'])}`" for e in log]
    (out / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    (out / "index.json").write_text(json.dumps({"config": cfg, "log": log}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("->", out / "index.md")
    return 1 if rc_all else 0


if __name__ == "__main__":
    sys.exit(main())
