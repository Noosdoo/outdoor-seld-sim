# -*- coding: utf-8 -*-
"""評価セットの契約の 2 段（本体 §3-4・㉑・R30-10・2026-10-26）: 封印の記録（seal.json）を書く・開封のときに実物と比べる。

1 段目 = 封印時（eval v4 を作った直後・生成のジョブの中）:
  `python server_sde/_eval_seal.py write --ds out/<eval v4 の置き場> --out <置き場>/eval_v4_seal.json`
  → 手元に取って `md/design/eval_v4_seal.json` としてコミットする（封印の記録をリポジトリに残す＝あとで書き換えたら git に出る）。
  中身: 評価セットの契約（CONTRACT_eval_v4.json）の sha256・本数・ファイル数・封印の時刻／plan/ の全ファイルの sha256／GEN_VERSIONS.txt の sha256
  （描画の連なりと読み込んだ全モジュールの sha = 描画のコードの識別子）／**描画とラベルの規約**（全部の scene.json から集めたゲートの規則・線・背景・
  マイクの向きの有無・背景の形の有無と、全部の masks の見出し）とその内容の sha256。規約が本ごとに混ざっていれば封印しない（版の混在で止める・§9-4 の 6）。
  規約を変えるときは新しい置き場で作り直して新しく封印する（同じ置き場の上書きはしない）。
2 段目 = 開封時（確定評価の 1 回・`_conf_exec_contract.py --eval-contract <CONTRACT_eval_v4.json> --seal-json md/design/eval_v4_seal.json`）:
  `check_seal` が seal.json の値と実物（契約のファイル・plan・GEN_VERSIONS）の sha を比べ、1 つでも違えば止める。
  契約の名前が eval_v4 なのに seal.json を渡さなければ止める（_conf_exec_contract 側）。eval v3 の道（seal なし）は旧のまま。
試験: scripts/test_eval_seal_v4.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEAL_VERSION = "eval_seal/1"
CONTRACT_NAME = "CONTRACT_eval_v4.json"


class SealError(Exception):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def conventions(ds: Path) -> dict:
    """全部の scene.json と masks から描画とラベルの規約を集める。本ごとに違えば SealError（版の混在）。"""
    scenes = sorted((ds / "work").glob("*/scene.json"))
    masks = sorted((ds / "masks").glob("*.csv"))
    if not scenes or not masks:
        raise SealError(f"scene.json（{len(scenes)}）か masks（{len(masks)}）が無い: {ds}")
    seen: dict[str, set] = {"gate": set(), "mic_heading": set(), "noise_shape": set(), "masks_header": set()}
    for p in scenes:
        s = json.loads(p.read_text(encoding="utf-8"))
        seen["gate"].add((s.get("gate_rule"), s.get("gate_line_db"), s.get("gate_bg")))
        seen["mic_heading"].add("mic_heading_deg" in (s.get("mic") or {}))
        seen["noise_shape"].add("shape" in (s.get("noise") or {}))
    for p in masks:
        with open(p, encoding="utf-8") as f:
            seen["masks_header"].add(f.readline().strip())
    mixed = {k: sorted(map(str, v)) for k, v in seen.items() if len(v) != 1}
    if mixed:
        raise SealError(f"描画とラベルの規約が本ごとに混ざっている（版の混在・封印しない）: {mixed}")
    (rule, line, bg), = seen["gate"]
    conv = {"gate_rule": rule, "gate_line_db": line, "gate_bg": bg, "mic_heading": next(iter(seen["mic_heading"])),
            "noise_shape": next(iter(seen["noise_shape"])), "masks_header": next(iter(seen["masks_header"])),
            "n_scene": len(scenes), "n_masks": len(masks)}
    return conv


def build_seal(ds: Path, contract_name: str = CONTRACT_NAME) -> dict:
    """封印の記録を作る（書かない）。契約が無い・plan が無い・規約が混ざっていれば SealError。"""
    ds = Path(ds)
    cp = ds / contract_name
    if not cp.is_file():
        raise SealError(f"評価セットの契約が無い: {cp}（生成のジョブの封印のあとに書く）")
    C = json.loads(cp.read_text(encoding="utf-8"))
    plan = {f"plan/{p.name}": sha256(p) for p in sorted((ds / "plan").iterdir()) if p.is_file()} if (ds / "plan").is_dir() else {}
    if not plan:
        raise SealError(f"plan が無い: {ds / 'plan'}")
    gv = ds / "GEN_VERSIONS.txt"
    if not gv.is_file():
        raise SealError(f"GEN_VERSIONS.txt が無い: {gv}")
    conv = conventions(ds)
    if C.get("n_clips") is not None and conv["n_scene"] != C["n_clips"]:
        raise SealError(f"scene.json の数 {conv['n_scene']} が契約の本数 {C['n_clips']} と違う")
    return {"seal_version": SEAL_VERSION, "dataset": ds.name, "written_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "contract": {"name": contract_name, "sha256": sha256(cp), "n_clips": C.get("n_clips"), "n_files": C.get("n_files"), "sealed_at": C.get("sealed_at")},
            "plan": plan, "gen_versions_sha256": sha256(gv),
            "conventions": conv, "conventions_sha256": _canon_sha(conv),
            "rule": "開封のとき _conf_exec_contract.py --seal-json がこの値と実物を比べる（違えば止める）。規約を変えるときは新しい置き場で作り直して新しく封印する"}


def write_seal(ds: Path, out: Path, contract_name: str = CONTRACT_NAME) -> dict:
    out = Path(out)
    if out.exists():
        raise SealError(f"封印の記録が既にある: {out}（上書きしない・作り直すなら新しい置き場で）")
    seal = build_seal(ds, contract_name)
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(seal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    return seal


def check_seal(seal_json: Path, eval_contract: Path) -> dict:
    """開封時: seal.json の値と実物（契約・plan・GEN_VERSIONS）を比べる。違えば SealError。返り値 = 記録（EXEC_CONTRACT に入れる）。"""
    seal_json, eval_contract = Path(seal_json), Path(eval_contract)
    if not seal_json.is_file():
        raise SealError(f"封印の記録が無い: {seal_json}")
    if not eval_contract.is_file():
        raise SealError(f"評価セットの契約が無い: {eval_contract}")
    S = json.loads(seal_json.read_text(encoding="utf-8"))
    if not str(S.get("seal_version", "")).startswith("eval_seal/1"):
        raise SealError(f"封印の記録の版が違う: {S.get('seal_version')}")
    ds = eval_contract.parent
    bad = []
    if S.get("dataset") != ds.name:
        bad.append(f"置き場の名前 {ds.name} ≠ 封印 {S.get('dataset')}")
    if (S.get("contract") or {}).get("name") != eval_contract.name:
        bad.append(f"契約の名前 {eval_contract.name} ≠ 封印 {(S.get('contract') or {}).get('name')}")
    c_sha = sha256(eval_contract)
    if (S.get("contract") or {}).get("sha256") != c_sha:
        bad.append(f"契約の sha256 {c_sha[:12]} ≠ 封印 {str((S.get('contract') or {}).get('sha256'))[:12]}")
    plan_now = {f"plan/{p.name}": sha256(p) for p in sorted((ds / "plan").iterdir()) if p.is_file()} if (ds / "plan").is_dir() else {}
    if plan_now != S.get("plan"):
        diff = sorted(set(plan_now) ^ set(S.get("plan") or {})) + sorted(k for k in set(plan_now) & set(S.get("plan") or {}) if plan_now[k] != S["plan"][k])
        bad.append(f"plan が封印と違う: {diff[:5]}")
    gv = ds / "GEN_VERSIONS.txt"
    if not gv.is_file() or sha256(gv) != S.get("gen_versions_sha256"):
        bad.append("GEN_VERSIONS.txt が封印と違う（か無い）")
    if _canon_sha(S.get("conventions")) != S.get("conventions_sha256"):
        bad.append("封印の記録の規約と、その sha256 が合わない（記録が書き換えられた）")
    if bad:
        raise SealError("封印の記録と実物が違う（開封しない）: " + "／".join(bad))
    return {"path": str(seal_json), "sha256": sha256(seal_json), "contract_sha256": c_sha, "dataset": ds.name,
            "conventions_sha256": S["conventions_sha256"], "gate_line_db": S["conventions"].get("gate_line_db")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="封印時: seal.json を書く")
    w.add_argument("--ds", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--contract-name", default=CONTRACT_NAME)
    c = sub.add_parser("check", help="開封時: seal.json と実物を比べる")
    c.add_argument("--seal", type=Path, required=True)
    c.add_argument("--contract", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            s = write_seal(a.ds, a.out, a.contract_name)
            print(f"SEAL_WRITTEN {a.out} contract {s['contract']['sha256'][:12]} plan {len(s['plan'])} 本・規約 {s['conventions_sha256'][:12]}（線 {s['conventions']['gate_line_db']}）")
        else:
            r = check_seal(a.seal, a.contract)
            print(f"SEAL_OK {r['dataset']} contract {r['contract_sha256'][:12]}")
    except SealError as e:
        print(f"NG {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
