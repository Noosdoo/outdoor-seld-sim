# -*- coding: utf-8 -*-
"""本体 §3-4（契約の 2 段・㉑・R30-10）の受入れ試験（2026-10-26）。架空の評価セット（3 本）で、封印の記録（_eval_seal.py）と開封の照合（_conf_exec_contract.py --seal-json）を確かめる。

  S1 封印時: write が契約の sha256・本数・plan/ の全ファイルの sha256・GEN_VERSIONS の sha256・規約（ゲートの規則・線 −5.0・背景・マイクの向き・背景の形・masks の見出し）と
     規約の sha256 を書く。同じ置き場にもう一度書こうとすると止まる（上書きしない）
  S2 開封時: 手を触れていなければ check_seal が通る（記録に契約の sha・規約の sha・線）
  S3 書き換えを止める: 契約の 1 バイト・plan の 1 バイト・plan にファイルを足す・GEN_VERSIONS・seal の規約だけ書き換え（sha と合わない）・置き場の名前が違う → どれも止まる
  S4 封印しない: 規約が本ごとに混ざる（1 本だけ線 0 dB・1 本だけ masks の見出しが 5 列）・scene の数が契約の本数と違う・契約が無い → write が止まる
  C1 契約の照合（_conf_exec_contract）: eval v4 の契約に --seal-json を渡すと通り、EXEC の記録に eval_seal が入る／eval v4 なのに --seal-json なし → 止まる（終了 3）／
     実物が封印と違う → 止まる（終了 3）／--seal-json だけで --eval-contract なし → 止まる／eval v3 の名前は --seal-json なしで旧のとおり通る
使い方: python scripts/test_eval_seal_v4.py（torch 不要）
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_sde"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import _eval_seal as ES  # noqa: E402
import _conf_exec_contract as CC  # noqa: E402

RESULTS = []
HEADER6 = "frame,class,track,src,snr_a_db,snr_a_bg_db"


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def make_ds(root: Path, name="dataset_outdoor_siren_v2final_v30_eval_v4", n=3, line_db=-5.0, odd_line=None, odd_header=None) -> Path:
    ds = root / name
    for i in range(n):
        c = f"eval_syn_clip{i:05d}"
        (ds / "work" / c).mkdir(parents=True)
        s = {"gate_rule": "v30", "gate_line_db": (odd_line if (odd_line is not None and i == 1) else line_db), "gate_bg": "noise+others",
             "mic": {"mic_heading_deg": 0.0}, "noise": {"dba": 45.0}, "sources": []}
        (ds / "work" / c / "scene.json").write_text(json.dumps(s), encoding="utf-8")
        (ds / "masks").mkdir(exist_ok=True)
        h = odd_header if (odd_header and i == 2) else HEADER6
        (ds / "masks" / f"{c}.csv").write_text(h + "\n10,0,0,0,3.1,8.0\n", encoding="utf-8")
    (ds / "plan").mkdir()
    (ds / "plan" / "assignment_v2final.csv").write_text("clip_id,seed\n" + "".join(f"eval_syn_clip{i:05d},{i}\n" for i in range(n)), encoding="utf-8")
    (ds / "GEN_VERSIONS.txt").write_text("render chain (import order): step11_v9_render.py -> step11_v30_render.py\n", encoding="utf-8")
    (ds / "CONTRACT_eval_v4.json").write_text(json.dumps({"name": name, "n_clips": n, "sealed_at": "2026-11-01T00:00:00", "n_files": 4 * n + 2, "files": {}}, indent=1), encoding="utf-8")
    return ds


def stops(f, *a):
    try:
        f(*a)
        return False
    except ES.SealError:
        return True


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="eval_seal_"))
    try:
        # S1
        ds = make_ds(tmp / "a")
        out = tmp / "eval_v4_seal.json"
        s = ES.write_seal(ds, out)
        S = json.loads(out.read_text(encoding="utf-8"))
        cv = S["conventions"]
        check("S1 封印時: 契約の sha・本数・plan・GEN_VERSIONS・規約（線 −5.0・6 列の見出し）と規約の sha を書く・同じ所にもう一度は止まる",
              S == s and S["seal_version"] == ES.SEAL_VERSION and S["contract"]["sha256"] == ES.sha256(ds / "CONTRACT_eval_v4.json") and S["contract"]["n_clips"] == 3
              and list(S["plan"]) == ["plan/assignment_v2final.csv"] and S["gen_versions_sha256"] == ES.sha256(ds / "GEN_VERSIONS.txt")
              and cv["gate_line_db"] == -5.0 and cv["gate_rule"] == "v30" and cv["gate_bg"] == "noise+others" and cv["masks_header"] == HEADER6
              and cv["mic_heading"] is True and cv["noise_shape"] is False and len(S["conventions_sha256"]) == 64 and stops(ES.write_seal, ds, out),
              json.dumps(cv, ensure_ascii=False))
        # S2
        r = ES.check_seal(out, ds / "CONTRACT_eval_v4.json")
        check("S2 開封時: 手を触れていなければ通る（記録に契約の sha・規約の sha・線）",
              r["contract_sha256"] == S["contract"]["sha256"] and r["conventions_sha256"] == S["conventions_sha256"] and r["gate_line_db"] == -5.0)
        # S3
        bad = {}

        def tamper(name, fn):
            d2 = tmp / f"t_{name}"
            shutil.copytree(ds.parent, d2)
            dd = d2 / ds.name
            seal2 = d2 / "seal.json"
            shutil.copy(out, seal2)
            fn(dd, seal2)
            bad[name] = stops(ES.check_seal, seal2, dd / "CONTRACT_eval_v4.json")

        def app(p, b=b" "):
            with open(p, "ab") as f:
                f.write(b)
        tamper("contract", lambda dd, sj: app(dd / "CONTRACT_eval_v4.json"))
        tamper("plan_byte", lambda dd, sj: app(dd / "plan" / "assignment_v2final.csv", b"x"))
        tamper("plan_add", lambda dd, sj: (dd / "plan" / "extra.csv").write_text("a\n", encoding="utf-8"))
        tamper("gen_versions", lambda dd, sj: app(dd / "GEN_VERSIONS.txt", b"\n"))

        def conv_edit(dd, sj):
            J = json.loads(sj.read_text(encoding="utf-8")); J["conventions"]["gate_line_db"] = 0.0
            sj.write_text(json.dumps(J), encoding="utf-8")
        tamper("seal_conventions", conv_edit)

        def rename(dd, sj):
            dd.rename(dd.parent / "dataset_other_eval_v4")
            bad["dataset_name"] = stops(ES.check_seal, sj, dd.parent / "dataset_other_eval_v4" / "CONTRACT_eval_v4.json")
            (dd.parent / "dataset_other_eval_v4").rename(dd)
        tamper("rename_dummy", rename)
        bad.pop("rename_dummy", None)
        check("S3 書き換えを止める: 契約・plan の 1 バイト・plan に足す・GEN_VERSIONS・seal の規約だけ・置き場の名前", len(bad) == 6 and all(bad.values()), str(bad))
        # S4
        r4 = {"mixed_line": stops(ES.build_seal, make_ds(tmp / "m1", odd_line=0.0)),
              "mixed_header": stops(ES.build_seal, make_ds(tmp / "m2", odd_header="frame,class,track,snr_a_db,snr_a_bg_db"))}
        d3 = make_ds(tmp / "m3")
        J = json.loads((d3 / "CONTRACT_eval_v4.json").read_text(encoding="utf-8")); J["n_clips"] = 4
        (d3 / "CONTRACT_eval_v4.json").write_text(json.dumps(J), encoding="utf-8")
        r4["count"] = stops(ES.build_seal, d3)
        d4 = make_ds(tmp / "m4"); (d4 / "CONTRACT_eval_v4.json").unlink()
        r4["no_contract"] = stops(ES.build_seal, d4)
        check("S4 封印しない: 規約の混在（線・masks の見出し）・scene の数 ≠ 契約の本数・契約が無い", all(r4.values()), str(r4))
        # C1
        ck = tmp / "epoch_099.ckpt"; ck.write_bytes(b"ckpt-bytes")
        sel = tmp / "sel.json"
        sel.write_text(json.dumps({"spec_version": "v2_ckpt_selector/2.1 (x)", "not_for_registration": False, "selected": 99, "conditions": {"height": "table"},
                                   "candidates": {"99": {"verified": True, "ckpt_sha256": CC.sha256(ck), "ckpt": "runs/x_sde_w3_causal/checkpoints/epoch_099.ckpt"}}}), encoding="utf-8")
        dec = tmp / "dec.json"
        dec.write_text(json.dumps({"adopt": False, "verdict": "表なし（不採用）", "inputs": {}}), encoding="utf-8")
        base = ["--select-json", str(sel), "--ckpt", str(ck), "--decide-json", str(dec)]
        c4 = str(ds / "CONTRACT_eval_v4.json")
        rc_ok = CC.main(base + ["--eval-contract", c4, "--seal-json", str(out), "--dry"])
        res = tmp / "opened"
        rc_res = CC.main(base + ["--eval-contract", c4, "--seal-json", str(out), "--reserve-dir", str(res), "--out", str(res / "EXEC_CONTRACT.json"), "--job", "1"])
        rec = json.loads((res / "EXEC_CONTRACT.json").read_text(encoding="utf-8")) if (res / "EXEC_CONTRACT.json").is_file() else {}
        rc_noseal = CC.main(base + ["--eval-contract", c4, "--dry"])
        d5 = tmp / "c_bad"; shutil.copytree(ds.parent, d5); app(d5 / ds.name / "plan" / "assignment_v2final.csv", b"y")
        rc_bad = CC.main(base + ["--eval-contract", str(d5 / ds.name / "CONTRACT_eval_v4.json"), "--seal-json", str(out), "--dry"])
        rc_nocontract = CC.main(base + ["--seal-json", str(out), "--dry"])
        v3 = tmp / "v3" / "dataset_outdoor_siren_v2final_eval_v3"; v3.mkdir(parents=True)
        (v3 / "CONTRACT_eval_v3.json").write_text("{}", encoding="utf-8")
        rc_v3 = CC.main(base + ["--eval-contract", str(v3 / "CONTRACT_eval_v3.json"), "--dry"])
        check("C1 契約の照合: eval v4＋seal → 通る・記録に eval_seal／eval v4 で seal なし → 3／実物が封印と違う → 3／seal だけ → 3／eval v3 は seal なしで旧のとおり",
              rc_ok == 0 and rc_res == 0 and (rec.get("eval_seal") or {}).get("contract_sha256") == S["contract"]["sha256"] and rec["eval_seal"]["sha256"] == ES.sha256(out)
              and rc_noseal == 3 and rc_bad == 3 and rc_nocontract == 3 and rc_v3 == 0,
              str((rc_ok, rc_res, rc_noseal, rc_bad, rc_nocontract, rc_v3)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
