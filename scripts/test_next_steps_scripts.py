# -*- coding: utf-8 -*-
"""次の手順の台本の通し検査（Astra RN20-04／05／06／07／08・2026-10-20・Fable）。GPU もサーバも使わない。

run_selector_v29.sh と run_calib_decide.sh を、手元の架空の推論ディレクトリ（_prediction_integrity の begin_run／publish で来歴を付けた本物の形）から
cp で写す検査モード（SEL_SRC_DIR／CALIB_SRC_DIR）で回し、次を確かめる:
  T1 選択器の台本: 8 候補を写して来歴を照合し、選択器が e099 を選び、selected_ckpt.json にサーバのユーザー名を残さない（~/ に伏せる）
  T2 選択器の台本: 候補の csv が壊れている → 台本が非 0 で止まり selected_ckpt.json を書かない（RN20-07）
  T3 選択器の台本: 種の札 s1 → infer_v2finalft_s1_e<NNN>_selfcausal を読み、label v2final_s1 の置き場に出す（RN20-02 の命名）
  T4 補正表の台本: calib 4,000 本（本物の plan・正解・オラクル予測）と長尺 400 本（本物の長尺 v1・v17b e139 の予測を来歴つきに）で通し → decide.json に採否・入力 10 本の sha・ckpt・表の sha
  T5 補正表の台本: 推論の ckpt が選択器の選んだ ckpt と違う → 採点の前に非 0 で止まる（RN20-06）
  T6 補正表の台本: 長尺の検出の設定が 0.5（登録の 0.7／0.3 でない）→ 止まる（RN20-01／06）
  T7 補正表の台本: 出力の置き場が既にある → 止まる（RN20-07）
使い方: python scripts/test_next_steps_scripts.py   （T4 は統一採点器 2 回と長尺の採点 8 回を回すので数分かかる）
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "server_sde")]
import _prediction_integrity as I  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASH = shutil.which("bash") or r"C:\Program Files\Git\bin\bash.exe"
PY = sys.executable
SEL_SH = ROOT / "server_sde/_handoff_scripts_20261020/run_selector_v29.sh"
CAL_SH = ROOT / "server_sde/_handoff_scripts_20261020/run_calib_decide.sh"
FAKE_HOME = "/home/testuser"          # 架空のサーバのホーム（本物のユーザー名の形は使わない・伏せる道具に引っかかるため）。出力に残らないことを確かめる
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)[:300]) if detail and not cond else ''}")


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def make_run(run: Path, foa: Path, clips: list[str], rows: dict[str, list[str]], *, epoch: int, ckpt_sha: str, ckpt_path: str,
             height: dict, frames: int | None = None, sed=(0.7, 0.3, 20.0)) -> dict:
    """来歴つきの架空の推論ディレクトリ（本物の run_meta.json／val_all_causal.csv／val_all_causal.done.json の形）。"""
    (run / "submissions").mkdir(parents=True, exist_ok=True)
    meta = {"schema": I.SCHEMA, "ckpt": ckpt_path, "ckpt_sha256": ckpt_sha, "ckpt_epoch": epoch, "dataset": "outdoor_siren_v12",
            "foa": str(foa), "glob": "*.flac", "prefix": "", "clips": clips, "audio_sha256": I.audio_manifest(foa, clips), "height": height,
            "code_sha256": {"infer": "c" * 64},
            "settings": {"batch": 25, "sed_threshold": sed[0], "sed_stay": sed[1], "sed_assoc": sed[2], "hysteresis": sed[1] < sed[0]}}
    if frames is not None:
        meta["frames"] = frames
    I.begin_run(run, meta)
    for c in clips:
        (run / "submissions" / (Path(c).stem + ".csv")).write_text("".join(rows.get(Path(c).stem, [])), encoding="utf-8")
    I.publish(run, meta)
    return meta


def run_sh(script: Path, args: list[str], env: dict) -> tuple[int, str]:
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    p = subprocess.run([BASH, str(script)] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=str(ROOT))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def selector_tests(T: Path) -> None:
    foa = T / "sel_foa"; foa.mkdir()
    clips = ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]
    for c in clips:
        (foa / c).write_bytes(c.encode())
    gt = T / "sel_gt"; gt.mkdir()
    (gt / "val_syn_clip00001.csv").write_text("".join(f"{k},4,0,30,0,1.2\n" for k in range(40, 100)), encoding="utf-8")
    (gt / "val_syn_clip00002.csv").write_text("".join(f"{k},4,0,-30,0,4.0\n" for k in range(40, 100)), encoding="utf-8")
    man = T / "sel_manifest.csv"; man.write_text("clip_id\nval_syn_clip00001\nval_syn_clip00002\n", encoding="utf-8")
    ht = T / "sel_height.csv"; ht.write_text("clip_id,mic_z\nval_syn_clip00001,1.5\nval_syn_clip00002,1.5\n", encoding="utf-8")
    height = {"on": True, "table": str(ht), "table_sha256": I.sha256(ht), "offset": 0.0, "default": 1.5}

    def build(src: Path, prefix: str):
        for e in (19, 39, 59, 79, 99, 119, 139, 149):
            d = 1.25 if e == 99 else 1.7                                   # e099 だけ GT 1.2 に近い → 選ばれる
            rows = {"val_syn_clip00001": [f"{k},4,0,30,0,{d}\n" for k in range(40, 100)], "val_syn_clip00002": []}
            make_run(src / f"{prefix}_e{e:03d}_selfcausal", foa, clips, rows, epoch=e, ckpt_sha=sha_text(f"ckpt{prefix}{e}"),
                     ckpt_path=f"{FAKE_HOME}/PSELDNets_logs/outdoor_siren_v12/runs/x/checkpoints/epoch_{e:03d}.ckpt", height=height)
    env = dict(SEL_MANIFEST=man.as_posix(), SEL_GT=gt.as_posix(), SEL_HEIGHT_TABLE=ht.as_posix(), SEL_EXPECTED=2,
               SEL_OUT_ROOT=(T / "sel_out").as_posix(), SEL_CAND_ROOT=(T / "sel_cand").as_posix())
    # T1
    src = T / "sel_src"; build(src, "infer_v2finalft")
    rc, out = run_sh(SEL_SH, [], dict(env, SEL_SRC_DIR=src.as_posix()))
    outs = sorted((T / "sel_out").glob("v2final_*/selected_ckpt.json"))
    rec = json.loads(outs[-1].read_text(encoding="utf-8")) if outs else {}
    sel_txt = "".join(p.read_text(encoding="utf-8") for p in (T / "sel_out").glob("v2final_*/select.*")) if outs else ""
    check("T1 選択器の台本: 8 候補の来歴 OK・e099 を選ぶ・selected_ckpt.json に sha と伏せた ckpt（~/）", rc == 0 and rec.get("selected_epoch") == 99
          and rec.get("ckpt_sha256") == sha_text("ckptinfer_v2finalft99") and str(rec.get("ckpt", "")).startswith("~/") and "testuser" not in json.dumps(rec), out[-800:])
    check("T1b 選択器の JSON／md にもサーバのユーザー名を残さない", outs and "testuser" not in sel_txt and "~/PSELDNets_logs" in sel_txt, sel_txt[:200])
    # T2: 候補の csv を壊す → 来歴の照合で止まる・selected_ckpt.json は書かない
    src2 = T / "sel_src_bad"; build(src2, "infer_v2finalft")
    with open(src2 / "infer_v2finalft_e039_selfcausal/val_all_causal.csv", "a", encoding="utf-8") as h:
        h.write("val_syn_clip00001,50,4,0,30,0,9.9\n")
    before = set((T / "sel_out").glob("v2final_*/selected_ckpt.json"))
    rc2, out2 = run_sh(SEL_SH, [], dict(env, SEL_SRC_DIR=src2.as_posix()))
    after = set((T / "sel_out").glob("v2final_*/selected_ckpt.json"))
    check("T2 候補の csv が壊れている → 非 0 で止まり selected_ckpt.json を書かない（RN20-07）", rc2 != 0 and after == before and "download content mismatch" in out2, out2[-500:])
    # T3: 種の札 s1 → 名前の決まり（RN20-02）
    src3 = T / "sel_src_s1"; build(src3, "infer_v2finalft_s1")
    rc3, out3 = run_sh(SEL_SH, ["s1"], dict(env, SEL_SRC_DIR=src3.as_posix()))
    outs3 = sorted((T / "sel_out").glob("v2final_s1_*/selected_ckpt.json"))
    rec3 = json.loads(outs3[-1].read_text(encoding="utf-8")) if outs3 else {}
    check("T3 種の札 s1: infer_v2finalft_s1_e<NNN>_selfcausal を読み label v2final_s1 に出す", rc3 == 0 and rec3.get("label") == "v2final_s1" and rec3.get("selected_epoch") == 99, out3[-500:])
    rc4, out4 = run_sh(SEL_SH, ["s3"], dict(env, SEL_SRC_DIR=src3.as_posix()))
    check("T3b 種の札 s3 は拒否", rc4 != 0, out4[-200:])


def split_pred(pred_csv: Path, clips: list[str]) -> dict[str, list[str]]:
    rows = {Path(c).stem: [] for c in clips}
    for line in pred_csv.read_text(encoding="utf-8").splitlines():
        p = line.strip().split(",")
        if len(p) == 7 and p[0] in rows:
            rows[p[0]].append(",".join(p[1:]) + "\n")
    return rows


def calib_tests(T: Path) -> None:
    plandir = ROOT / "out/dataset_outdoor_siren_v2final_calib/plan"
    gt = ROOT / "out/_v29_calib_gt/metadata_dist"
    longds = ROOT / "out/dataset_outdoor_long_v1"
    if not (gt.is_dir() and len(list(gt.glob("tune_syn_clip*.csv"))) == 4000 and (longds / "pred_v17b_e139.csv").is_file() and (plandir / "assignment_v2final.csv").is_file()):
        check("T4 前提（calib の正解 4,000・長尺 v1・calib の plan が手元にある）", False, "無い → T4〜T7 は飛ばす"); return
    TAG = "outdoor_siren_v2final_v29_causal_ft_epoch_099"
    ck_sha = sha_text("ckpt-calib-test")
    ck_path = f"{FAKE_HOME}/PSELDNets_logs/outdoor_siren_v12/runs/outdoor_siren_v2final_v29_causal_ft/checkpoints/epoch_099.ckpt"
    # calib 4,000 本（オラクル予測 = 正解をそのまま）・長尺 400 本（v17b e139 の予測）を来歴つきの架空ディレクトリに
    cal_clips = sorted(p.stem + ".flac" for p in gt.glob("tune_syn_clip*.csv"))
    foa_c = T / "cal_foa"; foa_c.mkdir()
    for c in cal_clips:
        (foa_c / c).write_bytes(c.encode())
    orc = T / "oracle_calib.csv"
    r = subprocess.run([PY, str(ROOT / "scripts/_gt_as_pred.py"), "--plan", str(plandir / "assignment_v2final.csv"), "--split", "tune", "--clip-max", "4000", "--meta", str(gt), "--out", str(orc)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    check("T4a オラクル予測（calib 4,000 本）を作れる", r.returncode == 0 and orc.is_file(), (r.stdout + r.stderr)[-400:])
    if not orc.is_file():
        return
    ht = plandir / "assignment_v2final.csv"
    height_c = {"on": True, "table": str(ht), "table_sha256": I.sha256(ht), "offset": 0.0, "default": 1.5}
    height_l = {"on": True, "table": "", "table_sha256": None, "offset": 0.0, "default": 1.5}
    long_clips = sorted(p.stem + ".flac" for p in (longds / "metadata_dist").glob("long_*.csv"))
    foa_l = T / "long_foa"; foa_l.mkdir()
    for c in long_clips:
        (foa_l / c).write_bytes(c.encode())
    long_rows = split_pred(longds / "pred_v17b_e139.csv", long_clips)

    def build(src: Path, *, ck=ck_sha, long_sed=(0.7, 0.3, 20.0)):
        make_run(src / f"infer_calib_{TAG}", foa_c, cal_clips, split_pred(orc, cal_clips), epoch=99, ckpt_sha=ck, ckpt_path=ck_path, height=height_c)
        make_run(src / f"infer_long_v1_{TAG}", foa_l, long_clips, long_rows, epoch=99, ckpt_sha=ck, ckpt_path=ck_path, height=height_l, frames=600, sed=long_sed)
    selck = T / "selected_ckpt.json"
    selck.write_text(json.dumps({"label": "v2final", "selected_epoch": 99, "ckpt": "~/x/epoch_099.ckpt", "ckpt_sha256": ck_sha, "ckpt_epoch": 99}), encoding="utf-8")
    env = dict(CALIB_OUT_ROOT=(T / "cal_out").as_posix(), CALIB_PRED_ROOT=(T / "cal_pred").as_posix())
    # T5: ckpt の不一致（軽いので先に）
    src5 = T / "cal_src_badck"; build(src5, ck=sha_text("other-ckpt"))
    rc5, out5 = run_sh(CAL_SH, [TAG, selck.as_posix()], dict(env, CALIB_SRC_DIR=src5.as_posix(), CALIB_OUT_ROOT=(T / "cal_out5").as_posix(), CALIB_PRED_ROOT=(T / "cal_pred5").as_posix()))
    check("T5 推論の ckpt が選択器の選んだ ckpt と違う → 採点の前に止まる（RN20-06）", rc5 == 6 and "選択器の選んだ ckpt と違う" in out5 and not (T / "cal_out5" / TAG / "dist_table.json").exists(), out5[-500:])
    # T6: 長尺の検出の設定が 0.5 固定（旧 long17 相当）
    src6 = T / "cal_src_sed05"; build(src6, long_sed=(0.5, 0.5, 20.0))
    rc6, out6 = run_sh(CAL_SH, [TAG, selck.as_posix()], dict(env, CALIB_SRC_DIR=src6.as_posix(), CALIB_OUT_ROOT=(T / "cal_out6").as_posix(), CALIB_PRED_ROOT=(T / "cal_pred6").as_posix()))
    check("T6 長尺の検出の設定が 0.5（登録の 0.7／0.3 でない）→ 止まる（RN20-01／06）", rc6 == 6 and "検出の設定が登録の値" in out6, out6[-500:])
    # T4: 通し
    src4 = T / "cal_src"; build(src4)
    rc4, out4 = run_sh(CAL_SH, [TAG, selck.as_posix()], dict(env, CALIB_SRC_DIR=src4.as_posix()))
    O = T / "cal_out" / TAG
    dec = json.loads((O / "decide.json").read_text(encoding="utf-8")) if (O / "decide.json").is_file() else {}
    tbl = json.loads((O / "dist_table.json").read_text(encoding="utf-8")) if (O / "dist_table.json").is_file() else {}
    cal_csv = T / "cal_pred" / TAG / f"infer_calib_{TAG}" / "val_all_causal.csv"
    check("T4 補正表の台本の通し: 来歴 OK → 表（fit 2,000 本・全体 4,000 本で読む）→ 表なし／あり → 長尺 fold40・41 → decide.json",
          rc4 == 0 and type(dec.get("adopt")) is bool and len(dec.get("inputs", {})) == 10 and dec.get("ckpt_sha256") == ck_sha
          and dec.get("table_sha256") == I.sha256(O / "dist_table.json") and dec.get("fit_pred_sha256") == I.sha256(cal_csv) == dec.get("pred_calib_sha256")
          and tbl.get("n_clips") == 2000 and tbl.get("n_clips_all") == 4000 and tbl.get("ckpt_sha256") == ck_sha and len(dec.get("conditions", {})) + len(dec.get("undefined", [])) >= 6, out4[-900:])
    check("T4b オラクル予測の表はほぼ恒等 → 採用にならない（至近捕捉 +5 pt が出ない）", dec.get("adopt") is False, dec.get("verdict"))
    check("T4c inputs_check.json に ckpt・設定・frames の照合の記録", (O / "inputs_check.json").is_file() and json.loads((O / "inputs_check.json").read_text(encoding="utf-8")).get("ok") is True)
    # T7: 出力の置き場が既にある → 止まる
    rc7, out7 = run_sh(CAL_SH, [TAG, selck.as_posix()], dict(env, CALIB_SRC_DIR=src4.as_posix()))
    check("T7 出力の置き場が既にある → 止まる（古い結果と混ぜない・RN20-07）", rc7 == 1 and "既にある" in out7, out7[-300:])


def main() -> int:
    T = Path(tempfile.mkdtemp(prefix="next_steps_"))
    try:
        selector_tests(T)
        calib_tests(T)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    k = sum(c for _, c in results)
    print(f"{k}/{len(results)} {'ALL PASS' if k == len(results) else 'FAILED'}")
    return 0 if k == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
