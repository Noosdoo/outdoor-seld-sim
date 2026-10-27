# -*- coding: utf-8 -*-
"""test_v2final_members.py — 変更票 #73（比べる相手 B・D）・#74（アブレーション 1〜4）の接続の検査（2026-10-21・Fable）。
  M1 実行の契約の自己検査（アブレーション・欠けの理由を含む 30 件）
  M2 凍結 JSON → 環境変数の自己検査（_baselines_env.py）
  M3 _v2final_members.sh の乾式（MEMBER_DRY=1・bash）: 出力名の一覧・v17 の環境（提案法／nohyst／noheight）・D の環境が凍結 JSON の値・B が無ければ 3・停止規則 4・環境が外に漏れない
  M4 確定評価の台本 v2final_conf_v3.sbatch の文面: 契約に --baselines-json／--ablation nohyst・nocausalft／--require 6 つ・停止規則に member_outs・推論の順（v2 → v1 → 3 → 4 → 1 → 2 → 相手）・
     採点（abl・baselines・commonD・D は知らせの表に入れない）・sha の一覧に新しい道具
  M5 実録の台本 field_eval_infer.sbatch の文面: 同じ契約・eval_field_ の点検・予約・同じ顔ぶれ・採点はしない
  M6 凍結 JSON（md/design/baselines_selected.json）が契約の照合に通る形（D の code に 170 行の tsv・ユーザー名なし）
使い方: python scripts/test_v2final_members.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(args, env=None, cwd=None):
    e = dict(os.environ, PYTHONUTF8="1"); e.update(env or {})
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=cwd or ROOT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def bash() -> str | None:
    for c in ("bash", r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if shutil.which(c) or Path(c).is_file():
            return c
    return None


def main() -> int:
    ok = []
    rc, o = run([PY, str(ROOT / "server_sde/_conf_exec_contract.py"), "--selftest"])
    ok.append(("M1 実行の契約の自己検査（アブレーション・欠けの理由・RI73-04 の必須入力と値域を含む）", rc == 0 and "ALL PASS" in o))
    rc, o = run([PY, str(ROOT / "server_sde/_baselines_env.py"), "--selftest"])
    ok.append(("M2 凍結 JSON → 環境変数の自己検査", rc == 0 and "ALL PASS" in o))
    # M3 乾式
    b = bash()
    tmp = Path(tempfile.mkdtemp(prefix="members_"))
    try:
        if b is None:
            ok.append(("M3 members.sh の乾式（bash が無い → 飛ばす）", True))
        else:
            def posix(p: Path) -> str:
                s = str(p).replace("\\", "/")
                return re.sub(r"^([A-Za-z]):/", lambda m: f"/{m.group(1).lower()}/", s)
            ps = tmp / "ps"; runs = tmp / "runs"; job = tmp / "job"; ps.mkdir(); runs.mkdir(); job.mkdir(); (runs / "infer_exist").mkdir()
            script = tmp / "dry.sh"
            script.write_text(f"""set -u
cd {posix(ROOT)}
export PS={posix(ps)} E2E={posix(ROOT)} PY={posix(Path(PY))} RUNS={posix(runs)} N=6000 JOBDIR={posix(job)} MEMBER_DRY=1 PYTHONUTF8=1
export CAUSAL_FOA=/tmp/foa CAUSAL_PREFIX=eval_syn_ CAUSAL_GLOB="eval_syn_*.flac"
source server_sde/_v2final_members.sh
echo "OUTS: $(member_outs infer_x "D B")"
member_v17 infer_x /tmp/e.ckpt 1 /tmp/plan.csv 0.3
member_v17 infer_x_abl_nohyst /tmp/e.ckpt 1 /tmp/plan.csv 0.7
member_v17 infer_x_abl_noheight /tmp/n.ckpt 0 "" 0.3
member_v17 infer_bad /tmp/e.ckpt 1 "" 0.3; echo "BAD_RC=$?"
member_D infer_x_baselineD md/design/baselines_selected.json; echo "D_RC=$?"
member_B infer_x_baselineB md/design/baselines_selected.json; echo "B_RC=$?"
member_absent infer_none infer_exist; echo "ABS_RC=$?"
echo "LEAK: ${{CAUSAL_OUT:-unset}} ${{BASELINE_GAIN_DB:-unset}}"
""", encoding="utf-8", newline="\n")
            rc, o = run([b, posix(script)])
            J = json.loads((ROOT / "md/design/baselines_selected.json").read_text(encoding="utf-8"))
            d = J["baselines"]["D"]["settings"]
            outs_ok = "OUTS: infer_x infer_x_abl_nocausalft infer_x_abl_nohyst infer_x_abl_scratch infer_x_abl_noheight infer_x_baselineD infer_x_baselineB" in o
            main_ok = "OUT=infer_x CKPT=/tmp/e.ckpt HEIGHT_COND=1 HEIGHT_TABLE=/tmp/plan.csv TH=0.7 STAY=0.3 ASSOC=20" in o
            nohyst_ok = "OUT=infer_x_abl_nohyst CKPT=/tmp/e.ckpt HEIGHT_COND=1 HEIGHT_TABLE=/tmp/plan.csv TH=0.7 STAY=0.7" in o
            noh_ok = "OUT=infer_x_abl_noheight CKPT=/tmp/n.ckpt HEIGHT_COND=0 HEIGHT_TABLE= TH=0.7 STAY=0.3" in o
            d_line = next((l for l in o.splitlines() if "OUT=infer_x_baselineD" in l), "")
            d_ok = (f"TH={d['sed_threshold']} STAY={d['sed_stay']} ASSOC={d['sed_assoc']} GAIN={d['gain_db']}" in d_line and "baselines_class_map_v2.json" in d_line
                    and "datasets_pretrain170" in d_line and "DATASET=synth" in d_line and "mACCDOA-HTSAT-0.567.ckpt" in d_line and "D_RC=0" in o)
            ok.append(("M3a 乾式: 出力名 7 つ・提案法 0.7/0.3・nohyst 0.7/0.7・noheight は HEIGHT_COND=0 で表なし", rc == 0 and outs_ok and main_ok and nohyst_ok and noh_ok))
            ok.append(("M3b 乾式: D の環境が凍結 JSON の値（ゲイン・閾値・対応表 v2・170 クラスの置き場・synth）", d_ok))
            b_expected = "B_RC=0" if "B" in J["baselines"] else "B_RC=3"
            ok.append(("M3c 乾式: HEIGHT_COND=1 で表なしは 3・B は凍結の有無どおり・既にある出力は 4・環境は外に漏れない",
                       "BAD_RC=3" in o and b_expected in o and "ABS_RC=4" in o and "LEAK: unset unset" in o))
            if not all(c for _, c in ok[-3:]):
                print(o[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # M4 確定評価の台本の文面
    sb = (ROOT / "server_sde/v2final_conf_v3.sbatch").read_text(encoding="utf-8")
    i_bj = sb.find('--baselines-json "$BJ"'); i_dry = sb.find('_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --dry'); i_abs = sb.find('member_absent $(member_outs "$OUT2" "$NAMES")')
    i_v2 = sb.find("export CAUSAL_OUT=$OUT2"); i_v1 = sb.find("export CAUSAL_OUT=$OUT1")
    i_3 = sb.find('member_v17 "${OUT2}_abl_nocausalft"'); i_4 = sb.find('member_v17 "${OUT2}_abl_nohyst"'); i_1 = sb.find('member_v17 "${OUT2}_abl_scratch"'); i_2 = sb.find('member_v17 "${OUT2}_abl_noheight"')
    i_D = sb.find('member_D "${OUT2}_baselineD"'); i_B = sb.find('member_B "${OUT2}_baselineB"'); i_score = sb.find("unified_eval_v3_v9c.md")
    ok.append(("M4a conf_v3: 契約に相手の凍結・nohyst・nocausalft・--require 6 つ・欠けの理由（V2FINAL_MISSING_OK）・契約の照合が停止規則より前・契約の引数は乾式も予約も配列展開（RI73-01）・相手の環境と入力の点検が契約より前（RI73-04）",
               0 < i_bj < i_dry < i_abs and '--ablation nohyst "$V2FINAL_CKPT"' in sb and '--ablation nocausalft "$BASE_LAST"' in sb
               and "--require B,D,scratch,noheight,nocausalft,nohyst" in sb and "V2FINAL_MISSING_OK" in sb and 'ABL_ARGS+=(--ablation scratch "$SCR" "$SCRJ")' in sb
               and "$CONTRACT_ARGS " not in sb and "$CONTRACT_ARGS\n" not in sb and sb.count('_conf_exec_contract.py "${CONTRACT_ARGS[@]}"') == 2
               and 0 < sb.find('--env "$nm"') < i_dry and "foa_wts" in sb
               and 0 < sb.find("torch.__version__,numpy.__version__") < i_dry < sb.find("--reserve-dir")))      # RF73-02: 版の照合が契約の乾式・予約より前
    ok.append(("M4b conf_v3: 推論の順 = v2 → v1 → 3 nocausalft → 4 nohyst → 1 scratch → 2 noheight → 相手 D・B → 採点・nohyst は保持 0.7・noheight は 0 と表なし",
               0 < i_v2 < i_v1 < i_3 < i_4 < i_1 < i_2 < i_D < i_B < i_score and '"$HT" 0.7' in sb and '"$NOH" 0 "" 0.3' in sb and '[ "$CKE" = "99" ]' in sb))
    abl_lines = [l for l in sb.splitlines() if "unified_eval_v3_v9c_abl.md" in l and "_score_unified.py" in l]
    bl_lines = [l for l in sb.splitlines() if "unified_eval_v3_v9c_baselines.md" in l and "_score_unified.py" in l]
    ok.append(("M4c conf_v3: 採点 = abl（統一・知覚）・baselines（知覚は全員・知らせは D を除く）・commonD（6 クラスに絞った写し・採点器は同じ）・表は v2final だけ＋v2final_notable",
               len(abl_lines) == 1 and "$ABL_ITEMS" in abl_lines[0] and "$NT_ITEM" in abl_lines[0] and "perception_eval_v3_abl.md" in sb
               and len(bl_lines) == 1 and "$BL_NOTIFY" in bl_lines[0] and '[ "$nm" = D ] || BL_NOTIFY=' in sb and "perception_eval_v3_baselines.md" in sb
               and "--classes 0,1,3,4,7,8" in sb and "perception_eval_v3_commonD.md" in sb and "--meta $CU/meta_commonD" in sb and sb.count('--dist-table-for v2final"') == 1))
    ok.append(("M4d conf_v3: sha の一覧に members.sh・_baselines_env.py・相手の台本・_filter_classes.py・凍結 JSON・アブレーションの ckpt",
               all(s in sb for s in ("server_sde/_v2final_members.sh", "server_sde/_baselines_env.py", "server_sde/_baseline_D_pretrain170_infer.py", "server_sde/_baseline_B_infer.py",
                                     "scripts/_filter_classes.py", '"$BJ" ${BASE_LAST:-} ${SCR:-} ${SCRJ:-} ${NOH:-} ${NOHJ:-}'))
               and "source $E2E/server_sde/_v2final_members.sh" in sb))
    # M5 実録の台本
    fb = (ROOT / "server_sde/field_eval_infer.sbatch").read_text(encoding="utf-8")
    j_dry = fb.find('_conf_exec_contract.py "${CONTRACT_ARGS[@]}" --dry'); j_pre = fb.find("[preflight] n="); j_res = fb.find("--reserve-dir"); j_inf = fb.find('member_v17 "$OUT2" "$V2FINAL_CKPT" 1 "$HT" 0.3')
    ok.append(("M5 field: 同じ契約（相手・アブレーション・--require）→ eval_field_ の点検 → 予約 → 同じ顔ぶれ（提案法 → 3 → 4 → 1 → 2 → D・B）・採点の行は無い・原本は置かない",
               0 < j_dry < j_pre < j_res < j_inf and "--require B,D,scratch,noheight,nocausalft,nohyst" in fb and "CAUSAL_PREFIX=eval_field_" in fb
               and 0 < fb.find("torch.__version__,numpy.__version__") < j_dry
               and fb.find('member_v17 "${OUT2}_abl_nocausalft"') < fb.find('member_v17 "${OUT2}_abl_nohyst"') < fb.find('member_v17 "${OUT2}_abl_scratch"') < fb.find('member_v17 "${OUT2}_abl_noheight"') < fb.find('member_D "${OUT2}_baselineD"')
               and not any(("_score_unified.py" in l or "step20_realsmoke" in l or "_perception_eval.py" in l) and not l.lstrip().startswith(("#", "echo")) for l in fb.splitlines())
               and "raw/" in fb and "height_missing" in fb))
    # M6 凍結 JSON の形
    J = json.loads((ROOT / "md/design/baselines_selected.json").read_text(encoding="utf-8"))
    raw = (ROOT / "md/design/baselines_selected.json").read_text(encoding="utf-8")
    D = J["baselines"]["D"]
    ok.append(("M6 凍結 JSON: D の code に 170 行の tsv（sha 64 桁）・重み／対応表／選択の記録のパスは ~/ 始まり・ユーザー名なし",
               any(k.endswith("cls_indices_train_pseldnets170.tsv") and len(v) == 64 for k, v in D["code"].items())
               and all(str(D[k]).startswith("~/") for k in ("weights", "classmap", "selection")) and not re.search(r"s\d{7}", raw)))
    # M8（Astra RI73-01）: 台本と同じ「配列に組んで "${CONTRACT_ARGS[@]}" で渡す」形で本物の契約 CLI を bash から呼び、乾式 → 予約 1 回目 OK → 同じ場所の 2 回目は 4、まで通す
    #   （理由に空白が入る --missing-ok も配列で崩れないこと）。入力は架空（契約の自己検査と同じ形）
    b = bash()
    tmp = Path(tempfile.mkdtemp(prefix="m8_"))
    try:
        if b is None:
            ok.append(("M8 配列で渡した本物の予約（bash が無い → 飛ばす）", True))
        else:
            import hashlib
            def sha(p: Path) -> str:
                return hashlib.sha256(p.read_bytes()).hexdigest()
            ck = tmp / "epoch_099.ckpt"; ck.write_bytes(b"ckpt")
            selj = tmp / "select.json"; selj.write_text(json.dumps({"spec_version": "v2_ckpt_selector/2.1 (t)", "not_for_registration": False, "selected": 99,
                                                                   "conditions": {"height": "table x / default 1.5 m / offset 0"}, "candidates": {"99": {"verified": True, "ckpt_sha256": sha(ck), "ckpt": "runs/x_causal/checkpoints/epoch_099.ckpt"}}}), encoding="utf-8")
            decj = tmp / "decide.json"; decj.write_text(json.dumps({"adopt": False, "verdict": "表なし", "inputs": {}}), encoding="utf-8")
            sys.path.insert(0, str(ROOT / "server_sde"))
            import _conf_exec_contract as CC
            ps = tmp / "ps"; (ps / "ckpts").mkdir(parents=True); (ps / "configs").mkdir(); (ps / "configs/infer.yaml").write_text("a: 1\n", encoding="utf-8")
            w = ps / "ckpts/D.ckpt"; w.write_bytes(b"w")
            codes = {}
            for suf in CC.REQUIRED_CODE["D"]:                            # RR73-01: モデルのソースは ps_dir/src の下・設定の木は configs_tree_sha256
                cf = (ps / suf) if suf.startswith("src/") else (tmp / f"x{suf}")
                cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text("#\n", encoding="utf-8"); codes[cf.as_posix()] = sha(cf)
            cmap = tmp / "cmap.json"; cmap.write_text("{}", encoding="utf-8")
            selg = tmp / "selgain.json"; selg.write_text(json.dumps({"selected": {"gain_db": 40, "th": 0.1, "stay": 0.0429}}), encoding="utf-8")
            bj = tmp / "baselines.json"
            bj.write_text(json.dumps({"baselines": {"D": {"kind": "pretrained_pseldnets_t", "weights": w.as_posix(), "weights_sha256": sha(w), "code": codes, "classmap": cmap.as_posix(), "classmap_sha256": sha(cmap),
                                                           "settings": {"gain_db": 40, "sed_threshold": 0.1, "sed_stay": 0.0429, "sed_assoc": 20, "ps_dir": ps.as_posix(), "configs_tree_sha256": CC.tree_sha256(ps / "configs"),
                                                                        "torch_version": CC.runtime_versions().get("torch", "2.8.0+cu128"), "numpy_version": CC.runtime_versions().get("numpy", "2.4.6")},
                                                           "selection": selg.as_posix(), "selection_sha256": sha(selg)}}}), encoding="utf-8")
            res = tmp / "opened"
            def posix(p: Path) -> str:
                s = str(p).replace("\\", "/")
                return re.sub(r"^([A-Za-z]):/", lambda m: f"/{m.group(1).lower()}/", s)
            script = tmp / "m8.sh"
            script.write_text(f"""set -u
PY={posix(Path(PY))}
MISS_ARGS=(--missing-ok B "学習が間に合わなかった 試験の理由" --missing-ok scratch "候補なし 試験")
CONTRACT_ARGS=(--select-json {posix(selj)} --decide-json {posix(decj)} --ckpt {posix(ck)} --dist-table "" --baselines-json {posix(bj)} --ablation nohyst {posix(ck)} --require B,D,scratch,nohyst "${{MISS_ARGS[@]}}")
$PY {posix(ROOT / 'server_sde/_conf_exec_contract.py')} "${{CONTRACT_ARGS[@]}}" --dry; echo "DRY_RC=$?"
$PY {posix(ROOT / 'server_sde/_conf_exec_contract.py')} "${{CONTRACT_ARGS[@]}}" --reserve-dir {posix(res)} --out {posix(res / 'EXEC_CONTRACT.json')} --job 1; echo "RES1_RC=$?"
$PY {posix(ROOT / 'server_sde/_conf_exec_contract.py')} "${{CONTRACT_ARGS[@]}}" --reserve-dir {posix(res)} --out {posix(res / 'EXEC_CONTRACT.json')} --job 2; echo "RES2_RC=$?"
""", encoding="utf-8", newline="\n")
            rc, o = run([b, posix(script)], env={"PYTHONUTF8": "1"})
            rec = json.loads((res / "EXEC_CONTRACT.json").read_text(encoding="utf-8")) if (res / "EXEC_CONTRACT.json").is_file() else {}
            ok.append(("M8 配列で渡した本物の契約: 乾式 0 → 予約 1 回目 0（記録に相手 D・nohyst・理由つきの欠け B/scratch）→ 2 回目 4",
                       "DRY_RC=0" in o and "RES1_RC=0" in o and "RES2_RC=4" in o and rec.get("job") == "1" and set((rec.get("baselines") or {}).get("items", {})) == {"D"}
                       and set(rec.get("ablations", {})) == {"nohyst"} and (rec.get("required") or {}).get("missing", {}).get("B") == "学習が間に合わなかった 試験の理由"))
            if not ok[-1][1]:
                print(o[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok)
    print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
