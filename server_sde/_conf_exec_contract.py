# -*- coding: utf-8 -*-
"""確定評価（v2final_conf_v3.sbatch）の実行の契約（Astra RN20-03・2026-10-20・Fable）。

封印した eval v3 を開く**前**に、次を 1 つの契約として照合し、開封を原子的に 1 回だけ予約する:
  - 選択器 2.1 の結果（select.json）: selected が決まっている・not_for_registration が false・選んだ候補の ckpt の sha256
  - 渡された ckpt（V2FINAL_CKPT）の sha256 が選んだ候補と同じ
  - 補正表の採否（decide.json・_dist_calib_v2.py decide）: adopt が true なら表（dist_table.json）が要り、その sha256 が decide.json の table_sha256 と同じ・形式が正しい。
    adopt が false なら表を渡してはいけない。decide.json に ckpt_sha256 があれば ckpt と同じこと
  - 予約: --reserve-dir を mkdir（原子的・既にあれば拒否）し、その中に EXEC_CONTRACT.json（全部の sha と時刻・ジョブ）を書く。失敗しても予約の記録は消さない
使い方:
  python _conf_exec_contract.py --select-json <json> --decide-json <json> --ckpt <ckpt> [--dist-table <json>|""] [--eval-contract <CONTRACT_eval_v3.json>] [--code <file> ...]
                                [--baselines-json <json>] [--ablation <札> <ckpt> [<select.json>]]... [--require B,D,scratch,...] [--missing-ok <名前> <理由>]...
                                [--reserve-dir <dir> --out <EXEC_CONTRACT.json> --job <id>] [--dry]
  --dry: 照合だけ（予約しない・書かない）。終了コード: 0 = 通った・3 = 契約が通らない・4 = 既に開封を予約した記録がある
  --selftest
変更票 #73（比べる相手）・#74（アブレーション）・2026-10-21:
  --baselines-json: 相手の凍結（重み・対応表・コードの sha・ゲインと閾値・val の選択の記録）。中の相手は全部照合する。
  --ablation 札 ckpt [select.json]: 札 = scratch／noheight（選択器 2.1 の JSON が要る・選んだ候補の ckpt の sha と一致・候補の run 名に _abl_<札>・
      noheight は選択器の条件が「高さの入力なし」）／nocausalft（基準学習の last.ckpt・run 名が _sde_w3 で終わる = 主 seed・select 不要）／nohyst（主の ckpt と同じ sha）。
  --require: 無いと止まる名前（相手の名前か札）。--missing-ok 名前 理由 で理由つきなら通す（記録に残す = 「作れなかった事実を書く」の約束）。
本体 §3-4（契約の 2 段・㉑・2026-10-26）:
  --seal-json md/design/eval_v4_seal.json: 封印時に書いた記録（_eval_seal.py write）と実物（--eval-contract の契約・plan・GEN_VERSIONS）を比べ、違えば止める。
  契約の名前に eval_v4 が入っているのに --seal-json が無ければ止める。eval v3 の道（--seal-json なし）は旧のまま。
本体 §3-5（検出のしきい値・2026-10-26）:
  --thresholds-json md/design/thresholds_v30.json: 表を threshold_table.load で確かめ（版・クラス・候補・保持 = round(入り × 3/7, 4)・**表を作った ckpt = 主の ckpt**）、
  sha と入り／保持を記録に（thresholds）。省略 = 旧の記録のまま。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SELECTOR_SPEC_PREFIX = "v2_ckpt_selector/2.1"
TABLE_VERSION_PREFIX = "dist_calib/2."
N_BINS = 20
ABL_TAGS = ("scratch", "noheight", "nocausalft", "nohyst")     # 変更票 #74 の 1〜4
ABL_NEEDS_SELECT = ("scratch", "noheight")                      # 学習し直し = 選択器 2.1 で選んだ ckpt
NOHEIGHT_CONDITION = "no height input"                          # 選択器 --height-none の conditions.height の頭


class ContractError(Exception):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path, what: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ContractError(f"{what} が読めない: {path}（{e}）")


def _norm(p: str) -> str:
    """凍結 JSON の中のパスの比較用: ~ を展開し / に統一（存在は見ない）。"""
    return Path(str(p)).expanduser().as_posix()


def runtime_versions() -> dict:
    """この python で読める torch／numpy の版（RF73-02: 予約より前の照合に使う）。読めなければ空の辞書（= 未確認）。"""
    out = {}
    try:
        import numpy as _np
        out["numpy"] = str(_np.__version__)
    except Exception:
        pass
    try:
        import torch as _torch
        out["torch"] = str(_torch.__version__)
    except Exception:
        pass
    return out


def tree_sha256(root: Path, pattern: str = "**/*.yaml") -> str:
    """ディレクトリの中の設定ファイル群（相対パス＋sha）をまとめた 1 つの sha256（RR73-01: D の設定の木を固定する）。"""
    h = hashlib.sha256()
    for p in sorted(root.glob(pattern)):
        if p.is_file():
            h.update(f"{p.relative_to(root).as_posix()}:{sha256(p)}\n".encode())
    return h.hexdigest()


def check_baselines(baselines_json: Path, files: bool = True) -> dict:
    """比べる相手（変更票 #73）の凍結の照合。files=True（サーバ・本番）= 重み・コード・対応表・設定の木の sha が今のファイルと一致するか。
    files=False（手元・凍結の道具）= ファイルを見ずにできる検査だけ（必須項目・値域・選択の記録との一致・**読込先と凍結一覧の一致**）。
    Astra RR73-01: 実際に推論が読むパス（B の標準化 feat_dir/foa_wts・公式コード official_dir/*.py・D の PSELDNets ps_dir の設定とモデルのソース）が、
    凍結一覧（code）の同じパスとして入っていることを確かめる。名前が同じでも別の置き場なら通さない。"""
    B = _load(baselines_json, "比べる相手の凍結 JSON")
    if not isinstance(B.get("baselines"), dict) or not B["baselines"]:
        raise ContractError("比べる相手の凍結 JSON に baselines が無い")
    out = {"json": str(baselines_json), "json_sha256": sha256(baselines_json), "files_checked": bool(files), "items": {}}
    for name, b in B["baselines"].items():
        for key in ("kind", "weights", "weights_sha256", "code", "settings", "selection", "selection_sha256"):
            if key not in b:
                raise ContractError(f"相手 {name}: {key} が無い（RI73-04: 選択の記録とその sha も必須）")
        kind = str(b["kind"]); fam = _kind_family(kind)
        req = REQUIRED_CODE.get(fam)
        if req is None:
            raise ContractError(f"相手 {name}: kind が未対応（{kind!r}）")
        code = b["code"] or {}
        if not isinstance(code, dict) or not code:
            raise ContractError(f"相手 {name}: code が空（必要なコード・設定ファイルを sha つきで列挙する）")
        norm_code = {_norm(k): v for k, v in code.items()}
        for suffix in req:                                        # RI73-04: 相手ごとの必須ファイルが code に揃っているか（名前の末尾で照合）
            if not any(k.endswith(suffix) for k in norm_code):
                raise ContractError(f"相手 {name}: 必須のファイルが code に無い: …{suffix}")
        for h in code.values():
            if not (isinstance(h, str) and len(h) == 64):
                raise ContractError(f"相手 {name}: code の sha256 が 64 桁でない")
        if not (isinstance(b["weights_sha256"], str) and len(b["weights_sha256"]) == 64):
            raise ContractError(f"相手 {name}: weights_sha256 が 64 桁でない")
        s = b["settings"]
        if not isinstance(s, dict):
            raise ContractError(f"相手 {name}: settings が辞書でない")
        for key in ("sed_threshold", "sed_stay", "sed_assoc"):
            if not isinstance(s.get(key), (int, float)) or isinstance(s.get(key), bool) or not math.isfinite(float(s[key])):
                raise ContractError(f"相手 {name}: settings.{key} が有限の数でない")
        th, stay, assoc = float(s["sed_threshold"]), float(s["sed_stay"]), float(s["sed_assoc"])
        gain = s.get("gain_db", 0)
        if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(float(gain)):
            raise ContractError(f"相手 {name}: settings.gain_db が有限の数でない")
        if not (0 < th <= 1 and 0 < stay <= th and 0 < assoc <= 180):     # RI73-04: 値域（入り (0,1]・保持 (0, 入り]・対応幅 (0,180]）
            raise ContractError(f"相手 {name}: 閾値の値域が違う（入り {th}・保持 {stay}・対応幅 {assoc}）")
        if assoc != ASSOC_DEG_FIXED:
            raise ContractError(f"相手 {name}: 対応幅は {ASSOC_DEG_FIXED}° に固定（{assoc}）")
        # ---- RR73-01: 読込先と凍結一覧の一致（ファイルは見ない・パスの同一性）
        if fam == "B":
            fd = s.get("feat_dir"); od = s.get("official_dir")
            if not fd or not od:
                raise ContractError(f"相手 {name}: settings.feat_dir（標準化の置き場）と settings.official_dir（公式コードの置き場）が必須")
            wts = f"{_norm(fd)}/foa_wts"
            if wts not in norm_code:
                raise ContractError(f"相手 {name}: 実際に読む標準化の重み {wts} が凍結一覧（code）に無い（別の置き場を指している）")
            for f in OFFICIAL_FILES_B:
                if f"{_norm(od)}/{f}" not in norm_code:
                    raise ContractError(f"相手 {name}: 実際に読む公式コード {_norm(od)}/{f} が凍結一覧（code）に無い")
            if s.get("gain_db", 0) not in (0, 0.0) and not s.get("gain_decision_json"):   # 本体 §3-6（M11）: v30 の B の倍率は主と同じ決定（decision.json）から
                raise ContractError(f"相手 {name}: B の倍率 {s.get('gain_db')!r} の出どころ（settings.gain_decision_json＝倍率の決定 decision.json と sha）が無い")
            if s.get("dist_encoding", "linear") not in ("linear", "log1p"):
                raise ContractError(f"相手 {name}: settings.dist_encoding は linear か log1p（{s.get('dist_encoding')!r}）")
        elif fam == "D":
            pd_ = s.get("ps_dir")
            if not pd_:
                raise ContractError(f"相手 {name}: settings.ps_dir（PSELDNets の置き場）が必須")
            if not _norm(b["weights"]).startswith(_norm(pd_) + "/"):
                raise ContractError(f"相手 {name}: 重みが ps_dir の下にない（{b['weights']}）")
            if not b.get("classmap") or not b.get("classmap_sha256"):
                raise ContractError(f"相手 {name}: 対応表（classmap・classmap_sha256）が必須")
            n_src = sum(1 for k in norm_code if k.startswith(_norm(pd_) + "/src/"))
            if n_src < 5:
                raise ContractError(f"相手 {name}: PSELDNets のモデル本体のソース（ps_dir/src/…）が凍結一覧に足りない（{n_src} 本）")
            for suffix in req:                                     # Astra RF73-01: 必須ソースは「ps_dir/<相対パス>」そのもの（別の控えの同名ファイルでは通さない）
                if suffix.startswith("src/") and f"{_norm(pd_)}/{suffix}" not in norm_code:
                    raise ContractError(f"相手 {name}: 実際に読む {_norm(pd_)}/{suffix} が凍結一覧（code）に無い（別の置き場の同名ファイルでは代わりにならない）")
            if not (isinstance(s.get("configs_tree_sha256"), str) and len(s["configs_tree_sha256"]) == 64):
                raise ContractError(f"相手 {name}: settings.configs_tree_sha256（ps_dir/configs の yaml 全部の sha）が必須")
        for key in ("torch_version", "numpy_version"):                # 外部ライブラリの版の固定（推論の台本が照合する）
            if not isinstance(s.get(key), str) or not s[key]:
                raise ContractError(f"相手 {name}: settings.{key} が必須（RR73-01・版の固定）")
        # ---- 選択の記録（手元にある）との一致
        sp = Path(b["selection"]).expanduser()
        if not sp.is_file():
            raise ContractError(f"相手 {name}: 選択の記録が無い: {b['selection']}")
        sel_sha = sha256(sp)
        if sel_sha != b["selection_sha256"]:
            raise ContractError(f"相手 {name}: 選択の記録の sha256 が凍結と違う（{sel_sha[:12]} ≠ {str(b['selection_sha256'])[:12]}）")
        S = _load(sp, f"相手 {name} の選択の記録")
        chosen = S.get("selected") or {}
        gain_dec = None
        if fam == "B" and s.get("gain_decision_json"):                # 本体 §3-6（M11）: B の倍率 = 主の倍率の決定（v30_gain_decide の decision.json の final）
            gp = Path(s["gain_decision_json"]).expanduser()
            if not gp.is_file():
                raise ContractError(f"相手 {name}: 倍率の決定が無い: {s['gain_decision_json']}")
            g_sha = sha256(gp)
            if g_sha != s.get("gain_decision_sha256"):
                raise ContractError(f"相手 {name}: 倍率の決定の sha256 が凍結と違う（{g_sha[:12]} ≠ {str(s.get('gain_decision_sha256'))[:12]}）")
            fin = (_load(gp, "倍率の決定").get("decision") or {}).get("final")
            if fin is None or float(fin) != float(gain):
                raise ContractError(f"相手 {name}: B の倍率 {gain} が主の倍率の決定（final = {fin}）と違う")
            gain_dec = {"path": str(s["gain_decision_json"]), "sha256": g_sha, "final": fin}
        elif float(gain) != float(chosen.get("gain_db", 0)):
            raise ContractError(f"相手 {name}: ゲインが val の選択の記録と違う（settings {gain} vs 記録 {chosen.get('gain_db')}）")
        if th != float(chosen.get("th", -1)) or stay != float(chosen.get("stay", -1)):
            raise ContractError(f"相手 {name}: ゲイン・入り・保持が val の選択の記録と違う（settings {gain}/{th}/{stay} vs 記録 {chosen.get('gain_db')}/{chosen.get('th')}/{chosen.get('stay')}）")
        rec = {"kind": kind, "settings": s, "selection_sha256": sel_sha, "n_code": len(code), "files_checked": bool(files)}
        if gain_dec:
            rec["gain_decision"] = gain_dec
        if s.get("thresholds_json"):                                   # 本体 §3-5／§3-6: クラス別の表（定数は旧の記録・推論は表を使う）
            tsha = s.get("thresholds_sha256")
            if not (isinstance(tsha, str) and len(tsha) == 64):
                raise ContractError(f"相手 {name}: settings.thresholds_json があるのに thresholds_sha256（64 桁）が無い")
            rec["thresholds"] = {"path": str(s["thresholds_json"]), "sha256": tsha, "constants_superseded": {"sed_threshold": th, "sed_stay": stay},
                                 "note": "定数は #73 の旧の記録。推論は表（クラス別）を使う"}
        # ---- ファイルの照合（サーバ）
        if files:
            wp = Path(b["weights"]).expanduser()                      # JSON には ~/ で書く（ユーザー名を残さない）
            if not wp.is_file():
                raise ContractError(f"相手 {name}: 重みが無い: {b['weights']}")
            w_sha = sha256(wp)
            if w_sha != b["weights_sha256"]:
                raise ContractError(f"相手 {name}: 重みの sha256 が凍結と違う（{w_sha[:12]} ≠ {str(b['weights_sha256'])[:12]}）")
            for path, h in code.items():
                cp = Path(path).expanduser()
                if not cp.is_file():
                    raise ContractError(f"相手 {name}: コードが無い: {path}")
                if sha256(cp) != h:
                    raise ContractError(f"相手 {name}: コードの sha256 が凍結と違う: {path}")
            if b.get("classmap"):
                mp = Path(b["classmap"]).expanduser()
                if not mp.is_file() or sha256(mp) != b.get("classmap_sha256"):
                    raise ContractError(f"相手 {name}: 対応表の sha256 が凍結と違う")
            if fam == "D":
                cfg_dir = Path(s["ps_dir"]).expanduser() / "configs"
                if not cfg_dir.is_dir():
                    raise ContractError(f"相手 {name}: ps_dir/configs が無い（{cfg_dir}）")
                t = tree_sha256(cfg_dir)
                if t != s["configs_tree_sha256"]:
                    raise ContractError(f"相手 {name}: ps_dir/configs の yaml の木の sha256 が凍結と違う（{t[:12]} ≠ {s['configs_tree_sha256'][:12]}）")
            # Astra RF73-02: 外部ライブラリの版は、この契約を走らせる python（= 推論に使う venv）で読んで、予約より前に照合する（読めない環境では未確認と記録）
            have = runtime_versions()
            if have:
                for key, lib in (("torch_version", "torch"), ("numpy_version", "numpy")):
                    if have.get(lib) is not None and have[lib] != s[key]:
                        raise ContractError(f"相手 {name}: {lib} の版が凍結と違う（凍結 {s[key]}・この python {have[lib]}）")
                rec["runtime_versions"] = have
            else:
                rec["runtime_versions"] = None
            rec["weights_sha256"] = w_sha
            if s.get("thresholds_json"):                               # 本体 §3-5: 表の中身（版・対象クラス・候補・保持の比・表を作った重み = 凍結の重み）
                tp = Path(s["thresholds_json"]).expanduser()
                if not tp.is_file() or sha256(tp) != s["thresholds_sha256"]:
                    raise ContractError(f"相手 {name}: しきい値の表が無いか sha256 が凍結と違う: {s['thresholds_json']}")
                if str(Path(__file__).resolve().parent) not in sys.path:
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                import threshold_table
                cls = None
                if fam == "D":
                    CMJ = _load(Path(b["classmap"]).expanduser(), "対応表")
                    cls = sorted({int(v["ours_index"]) for v in CMJ["D_pseldnets_pretrained"]["map"].values()})
                try:
                    t = threshold_table.load(tp, 9, "hyst", ckpt_sha=b["weights_sha256"], classes=cls)
                except threshold_table.ThresholdTableError as e:
                    raise ContractError(f"相手 {name}: {e}")
                rec["thresholds"].update(threshold_table.record(t))
        out["items"][name] = rec
    return out


OFFICIAL_FILES_B = ("cls_feature_class.py", "seldnet_model.py", "train_seldnet.py", "parameters.py", "cls_compute_seld_results.py", "SELD_evaluation_metrics.py")


ASSOC_DEG_FIXED = 20.0            # 決めた対応幅（ヒステリシスの対応・重複の整理も同じ）
REQUIRED_CODE = {                 # RI73-04／RR73-01: 相手ごとに code に揃っていなければならないファイル（名前の末尾で照合）
    "D": ("_baseline_D_pretrain170_infer.py", "_baseline_merge.py", "hysteresis_patch.py", "_prediction_integrity.py", "cls_indices_train_pseldnets170.tsv",
          "src/models/model_module.py", "src/models/components/htsat.py", "src/models/components/model_module.py", "src/models/multi_accdoa.py",
          "src/utils/config.py", "src/utils/feature.py", "src/utils/datasets.py", "src/utils/data_utilities.py"),
    "B": ("_baseline_B_infer.py", "_baseline_B_common.py", "_baseline_merge.py", "hysteresis_patch.py", "_prediction_integrity.py", "foa_wts", "B_prepare_record.json",
          "cls_feature_class.py", "seldnet_model.py", "train_seldnet.py", "parameters.py", "cls_compute_seld_results.py", "SELD_evaluation_metrics.py"),
}


def _kind_family(kind: str) -> str | None:
    if kind.startswith("pretrained_pseldnets"):
        return "D"
    if kind.startswith("dcase2024"):
        return "B"
    return None


def _check_selected(select_json: Path, ck_sha: str, what: str) -> dict:
    """選択器 2.1 の JSON と ckpt の sha の照合（主の ckpt と、学習し直しのアブレーションで共通）。選んだ候補の記録を返す。"""
    S = _load(select_json, f"{what}の選択器の JSON")
    if not str(S.get("spec_version", "")).startswith(SELECTOR_SPEC_PREFIX):
        raise ContractError(f"{what}: 選択器の版が 2.1 でない: {S.get('spec_version')!r}")
    if S.get("not_for_registration") is not False:
        raise ContractError(f"{what}: 選択器の結果が『登録には使えない』（候補集合か門が仕様と違う試走）")
    sel = S.get("selected")
    if type(sel) is not int:
        raise ContractError(f"{what}: 選択器が ckpt を選んでいない（selected = {sel!r}）")
    cand = (S.get("candidates") or {}).get(str(sel))
    if not cand or cand.get("verified") is not True:
        raise ContractError(f"{what}: 選んだ候補 e{sel:03d} の来歴が選択器の JSON に無い")
    if cand.get("ckpt_sha256") != ck_sha:
        raise ContractError(f"{what}: 渡された ckpt の sha256 が選んだ候補 e{sel:03d} と違う（{ck_sha[:12]} ≠ {str(cand.get('ckpt_sha256'))[:12]}）")
    return {"S": S, "selected": sel, "cand": cand}


def check_ablations(items: list[list[str]], main_ck_sha: str) -> dict:
    """変更票 #74 のアブレーション（1〜4）の ckpt の照合。items = [[札, ckpt, (select.json)], ...]。
    scratch／noheight = 選択器 2.1 で選んだ ckpt（候補の run 名に _abl_<札>・noheight は選択器の条件が「高さの入力なし」・scratch はそうでない）。
    nocausalft = 基準学習の last.ckpt（run 名が _sde_w3 で終わる = 主 seed・因果 ft の run ではない）。nohyst = 主の ckpt と同じ sha（閾値だけ替える）。"""
    out: dict = {}
    for it in items:
        if len(it) not in (2, 3):
            raise ContractError(f"--ablation は 札 ckpt [select.json] の 2〜3 個: {it}")
        tag, ck = it[0], Path(it[1]).expanduser()
        sj = Path(it[2]).expanduser() if len(it) == 3 else None
        if tag not in ABL_TAGS:
            raise ContractError(f"アブレーションの札が違う（{ABL_TAGS} のどれか）: {tag}")
        if tag in out:
            raise ContractError(f"アブレーション {tag} が 2 回渡された")
        if not ck.is_file():
            raise ContractError(f"アブレーション {tag}: ckpt が無い: {ck}")
        c_sha = sha256(ck)
        rec = {"ckpt": str(ck), "ckpt_sha256": c_sha}
        if tag in ABL_NEEDS_SELECT:
            if sj is None:
                raise ContractError(f"アブレーション {tag}: 選択器 2.1 の JSON が要る（学習し直し = 提案法と同じ手順で ckpt を 1 つ選ぶ）")
            r = _check_selected(sj, c_sha, f"アブレーション {tag}")
            cand_ck = str(r["cand"].get("ckpt") or "")
            if f"_abl_{tag}" not in cand_ck:
                raise ContractError(f"アブレーション {tag}: 選んだ候補の run 名に _abl_{tag} が無い（{cand_ck}）")
            hcond = str(((r["S"].get("conditions") or {}).get("height")) or "")
            if tag == "noheight" and not hcond.startswith(NOHEIGHT_CONDITION):
                raise ContractError(f"アブレーション noheight: 選択器の条件が「高さの入力なし」でない（{hcond!r}）")
            if tag == "scratch" and hcond.startswith(NOHEIGHT_CONDITION):
                raise ContractError(f"アブレーション scratch: 選択器の条件が「高さの入力なし」になっている（scratch は高さの入力あり）")
            rec.update(select_json=str(sj), select_json_sha256=sha256(sj), selected_epoch=r["selected"], selector_spec=r["S"].get("spec_version"), candidate_ckpt=cand_ck)
        else:
            if sj is not None:
                raise ContractError(f"アブレーション {tag}: 選択器の JSON は渡さない（推論だけ・ckpt は選ばない）")
            if tag == "nocausalft":
                run = ck.parent.parent.name
                if ck.name != "last.ckpt" or not run.endswith("_sde_w3") or "causal" in run or "_abl_" in run:
                    raise ContractError(f"アブレーション nocausalft: 基準学習（主 seed・run 名が _sde_w3 で終わる）の last.ckpt でない: {ck}")
                rec["run"] = run
            elif tag == "nohyst":
                if c_sha != main_ck_sha:
                    raise ContractError(f"アブレーション nohyst: ckpt が主の ckpt と違う（{c_sha[:12]} ≠ {main_ck_sha[:12]}）。閾値だけを替える")
        out[tag] = rec
    return out


def check_required(required: list[str], present: set[str], missing_ok: list[list[str]]) -> dict:
    """--require の名前（相手の名前・アブレーションの札）が揃っているか。無いものは --missing-ok 名前 理由 があるときだけ通し、理由を記録に残す。"""
    reasons: dict[str, str] = {}
    for it in missing_ok:
        if len(it) != 2 or not it[1].strip():
            raise ContractError(f"--missing-ok は 名前 理由 の 2 個（理由は空でない）: {it}")
        reasons[it[0]] = it[1].strip()
    missing = [r for r in required if r not in present]
    no_reason = [m for m in missing if m not in reasons]
    if no_reason:
        raise ContractError(f"要る相手／アブレーションが無い（理由なし）: {no_reason}。作れなかったなら --missing-ok 名前 理由 で理由を記録する")
    unused = [k for k in reasons if k in present]
    if unused:
        raise ContractError(f"--missing-ok に、実際には揃っている名前がある: {unused}")
    return {"required": list(required), "present": sorted(present), "missing": {m: reasons[m] for m in missing}}


def check_input_gain(ckpt: Path, input_gain_db: float) -> float:
    """本体 §3-2（2026-10-26）: 推論に渡す入力の倍率（SDE_INPUT_GAIN_DB）が ckpt に記録された値（`sde_input_gain_db`・無い ckpt は 0 = 旧の学習）と同じか。違えば止める。"""
    import torch
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    g_ck = float((ck.get("sde_input_gain_db", 0.0) if isinstance(ck, dict) else 0.0) or 0.0)
    if abs(g_ck - float(input_gain_db)) > 1e-9:
        raise ContractError(f"入力の倍率が ckpt の記録と違う（渡す {float(input_gain_db):g} dB ≠ ckpt {g_ck:g} dB・本体 §3-2）")
    return g_ck


def check(select_json: Path, decide_json: Path, ckpt: Path, dist_table: Path | None, eval_contract: Path | None = None, code: list[Path] = (), baselines_json: Path | None = None,
          ablations: list[list[str]] = (), required: list[str] = (), missing_ok: list[list[str]] = (), input_gain_db: float | None = None,
          seal_json: Path | None = None, thresholds_json: Path | None = None) -> dict:
    """契約を照合して記録（dict）を返す。通らなければ ContractError。baselines_json（#73）があれば相手の凍結、ablations（#74）があればアブレーションの ckpt も照合する。"""
    if not Path(ckpt).is_file():
        raise ContractError(f"ckpt が無い: {ckpt}")
    ck_sha = sha256(ckpt)
    r_sel = _check_selected(select_json, ck_sha, "主の ckpt")
    S, sel = r_sel["S"], r_sel["selected"]
    D = _load(decide_json, "補正表の採否の JSON")
    if type(D.get("adopt")) is not bool or "verdict" not in D:
        raise ContractError("採否の JSON に adopt（真偽）と verdict が無い（_dist_calib_v2.py decide の出力か）")
    if D.get("ckpt_sha256") and D["ckpt_sha256"] != ck_sha:
        raise ContractError(f"採否の JSON の ckpt_sha256 が渡された ckpt と違う（{str(D['ckpt_sha256'])[:12]} ≠ {ck_sha[:12]}）")
    table_rec = None
    if D["adopt"]:
        if not dist_table:
            raise ContractError("採否は『表あり』なのに DIST_TABLE が渡されていない")
        if not Path(dist_table).is_file():
            raise ContractError(f"補正表が無い: {dist_table}")
        t_sha = sha256(dist_table)
        if D.get("table_sha256") != t_sha:
            raise ContractError(f"補正表の sha256 が採否の JSON の table_sha256 と違う（{t_sha[:12]} ≠ {str(D.get('table_sha256'))[:12]}）")
        T = _load(dist_table, "補正表")
        vals, cents = T.get("values"), T.get("centers")
        if not str(T.get("version", "")).startswith(TABLE_VERSION_PREFIX) or not isinstance(vals, list) or not isinstance(cents, list) \
                or len(vals) != N_BINS or len(cents) != N_BINS or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals):
            raise ContractError("補正表の形式が違う（version dist_calib/2.*・values／centers 20 個・有限・0 以上）")
        if T.get("fit_pred_sha256") and D.get("fit_pred_sha256") and T["fit_pred_sha256"] != D["fit_pred_sha256"]:
            raise ContractError("補正表を作った予測の sha256 が採否の JSON と違う")
        table_rec = {"path": str(dist_table), "sha256": t_sha, "n_pairs": T.get("n_pairs")}
    else:
        if dist_table:
            raise ContractError("採否は『表なし』なのに DIST_TABLE が渡されている（不採用の表・別の表を掛けない）")
    b_rec = check_baselines(baselines_json) if baselines_json else None
    a_rec = check_ablations(list(ablations), ck_sha) if ablations else {}
    present = set(b_rec["items"]) if b_rec else set()
    present |= set(a_rec)
    req_rec = check_required(list(required), present, list(missing_ok)) if (required or missing_ok) else None
    rec = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "ckpt": str(ckpt), "ckpt_sha256": ck_sha, "selected_epoch": sel,
           "baselines": b_rec, "ablations": a_rec, "required": req_rec,
           "select_json": str(select_json), "select_json_sha256": sha256(select_json), "selector_spec": S.get("spec_version"),
           "decide_json": str(decide_json), "decide_json_sha256": sha256(decide_json), "adopt": D["adopt"], "verdict": D.get("verdict"),
           "dist_table": table_rec, "decide_inputs": D.get("inputs")}
    if eval_contract:
        if not Path(eval_contract).is_file():
            raise ContractError(f"評価セットの契約が無い: {eval_contract}")
        rec["eval_contract"] = {"path": str(eval_contract), "sha256": sha256(eval_contract)}
    if eval_contract and "eval_v4" in Path(eval_contract).name and not seal_json:
        raise ContractError("eval v4 の契約は封印の記録（--seal-json md/design/eval_v4_seal.json）と照合してから開く（本体 §3-4）")
    if seal_json:                                      # 本体 §3-4: 封印の記録と実物を比べる（違えば止める）
        if not eval_contract:
            raise ContractError("--seal-json には --eval-contract（照合する評価セットの契約）が要る")
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _eval_seal
        try:
            rec["eval_seal"] = _eval_seal.check_seal(Path(seal_json), Path(eval_contract))
        except _eval_seal.SealError as e:
            raise ContractError(str(e))
    if thresholds_json:                                # 本体 §3-5: 表を確かめて sha を記録（表を作った ckpt が主の ckpt と違えば止める）
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        import threshold_table
        try:
            rec["thresholds"] = threshold_table.record(threshold_table.load(Path(thresholds_json), 9, "hyst", ckpt_sha=ck_sha))
        except threshold_table.ThresholdTableError as e:
            raise ContractError(str(e))
    if code:
        rec["code_sha256"] = {Path(p).name: sha256(p) for p in code}
    if input_gain_db is not None:                      # 本体 §3-2: 渡したときだけ照合して記録（既定 = 旧の記録のまま）
        rec["input_gain_db"] = check_input_gain(ckpt, input_gain_db)
    return rec


def reserve(reserve_dir: Path, out: Path, rec: dict, job: str | None) -> None:
    """開封の予約（原子的）。既にあれば 4 で止める。予約の記録はあとで消さない（失敗しても『開封を始めた』記録が残る）。"""
    try:
        os.mkdir(reserve_dir)
    except FileExistsError:
        prev = Path(reserve_dir) / out.name
        print(f"NG 既に開封を予約した記録がある: {reserve_dir}" + (f"（{prev.read_text(encoding='utf-8')[:200]}）" if prev.is_file() else "（記録ファイル無し = 以前の予約が途中で止まった）"))
        sys.exit(4)
    rec = dict(rec, job=job, reserve_dir=str(reserve_dir))
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-json", type=Path)
    ap.add_argument("--decide-json", type=Path)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--dist-table", default="")
    ap.add_argument("--eval-contract", type=Path, default=None)
    ap.add_argument("--code", type=Path, nargs="*", default=[])
    ap.add_argument("--baselines-json", type=Path, default=None, help="変更票 #73: 比べる相手の凍結（重み・対応表・コードの sha・ゲインと閾値・val の選択の記録）")
    ap.add_argument("--ablation", nargs="+", action="append", default=[], metavar="X", help="変更票 #74: 札 ckpt [select.json]（繰り返し可）")
    ap.add_argument("--require", default="", help="無いと止まる名前（相手の名前・アブレーションの札・カンマ区切り）")
    ap.add_argument("--missing-ok", nargs=2, action="append", default=[], metavar=("NAME", "REASON"), help="無い名前を理由つきで通す（記録に残す）")
    ap.add_argument("--input-gain-db", type=float, default=None, help="本体 §3-2: 推論に渡す SDE_INPUT_GAIN_DB（ckpt の sde_input_gain_db と照合・省略 = 照合しない＝旧）")
    ap.add_argument("--seal-json", type=Path, default=None, help="本体 §3-4: 封印の記録（md/design/eval_v4_seal.json）。--eval-contract の実物と比べる（eval v4 では必須）")
    ap.add_argument("--thresholds-json", type=Path, default=None, help="本体 §3-5: 検出のしきい値の表（md/design/thresholds_v30.json）。確かめて sha を記録（表を作った ckpt = 主の ckpt）")
    ap.add_argument("--reserve-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.select_json and a.decide_json and a.ckpt):
        ap.error("--select-json・--decide-json・--ckpt が要る")
    try:
        req = [x.strip() for x in a.require.split(",") if x.strip()]
        rec = check(a.select_json, a.decide_json, a.ckpt, Path(a.dist_table) if a.dist_table else None, a.eval_contract, a.code, a.baselines_json,
                    a.ablation, req, a.missing_ok, input_gain_db=a.input_gain_db, seal_json=a.seal_json, thresholds_json=a.thresholds_json)
    except ContractError as e:
        print(f"NG {e}")
        return 3
    extra = ""
    if rec.get("baselines"):
        extra += f"・相手 {','.join(sorted(rec['baselines']['items']))}"
    if rec.get("ablations"):
        extra += f"・アブレーション {','.join(sorted(rec['ablations']))}"
    if rec.get("required") and rec["required"]["missing"]:
        extra += f"・無し（理由つき）{','.join(sorted(rec['required']['missing']))}"
    print(f"OK ckpt e{rec['selected_epoch']:03d} sha {rec['ckpt_sha256'][:12]}・補正表 {'あり ' + rec['dist_table']['sha256'][:12] if rec['dist_table'] else 'なし'}（{rec['verdict']}）{extra}")
    if a.dry:
        return 0
    if not (a.reserve_dir and a.out):
        ap.error("予約するには --reserve-dir と --out が要る（照合だけなら --dry）")
    reserve(a.reserve_dir, a.out, rec, a.job)
    print(f"RESERVED {a.reserve_dir} -> {a.out}")
    return 0


def selftest() -> int:
    import tempfile
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="conf_contract_"))
    ck = tmp / "epoch_099.ckpt"; ck.write_bytes(b"ckpt-bytes")
    ck_sha = sha256(ck)
    table = tmp / "dist_table.json"
    table.write_text(json.dumps({"version": "dist_calib/2.0", "centers": [0.125 + 0.25 * i for i in range(20)], "values": [0.125 + 0.25 * i for i in range(20)], "n_pairs": 100, "fit_pred_sha256": "f" * 64}), encoding="utf-8")
    t_sha = sha256(table)

    def sel(selected=99, nfr=False, cand_sha=ck_sha, spec="v2_ckpt_selector/2.1 (x)", name="sel", cand_ckpt="runs/outdoor_siren_v2final_v29_sde_w3_causal/checkpoints/epoch_099.ckpt",
            height="table assignment_v2final.csv sha256 abcd / default 1.5 m / offset 0"):
        p = tmp / f"{name}.json"
        p.write_text(json.dumps({"spec_version": spec, "not_for_registration": nfr, "selected": selected, "conditions": {"height": height},
                                 "candidates": {"99": {"verified": True, "ckpt_sha256": cand_sha, "ckpt": cand_ckpt}}}), encoding="utf-8"); return p

    def dec(adopt, tsha=t_sha, ck=None, name="dec"):
        p = tmp / f"{name}.json"
        d = {"adopt": adopt, "verdict": "表あり（採用）" if adopt else "表なし（不採用）", "table_sha256": tsha, "inputs": {"x": "y"}}
        if ck:
            d["ckpt_sha256"] = ck
        p.write_text(json.dumps(d), encoding="utf-8"); return p

    def run(*args):
        return main(list(args))

    base = ["--select-json", str(sel()), "--ckpt", str(ck)]
    ok.append(("表なし・契約 OK（乾式）", run(*base, "--decide-json", str(dec(False)), "--dry") == 0))
    ok.append(("表あり・契約 OK（乾式）", run(*base, "--decide-json", str(dec(True)), "--dist-table", str(table), "--dry") == 0))
    ok.append(("表あり の採否なのに表を渡さない → 拒否", run(*base, "--decide-json", str(dec(True)), "--dry") == 3))
    ok.append(("表なし の採否なのに表を渡す → 拒否", run(*base, "--decide-json", str(dec(False)), "--dist-table", str(table), "--dry") == 3))
    ok.append(("表の sha が採否の記録と違う → 拒否", run(*base, "--decide-json", str(dec(True, tsha="0" * 64, name="dec_bad")), "--dist-table", str(table), "--dry") == 3))
    ok.append(("ckpt の sha が選んだ候補と違う → 拒否", run("--select-json", str(sel(cand_sha="1" * 64, name="sel_bad")), "--ckpt", str(ck), "--decide-json", str(dec(False)), "--dry") == 3))
    ok.append(("選択器が『登録には使えない』 → 拒否", run("--select-json", str(sel(nfr=True, name="sel_nfr")), "--ckpt", str(ck), "--decide-json", str(dec(False)), "--dry") == 3))
    ok.append(("選択器が選んでいない（selected = null） → 拒否", run("--select-json", str(sel(selected=None, name="sel_none")), "--ckpt", str(ck), "--decide-json", str(dec(False)), "--dry") == 3))
    ok.append(("採否の JSON の ckpt_sha256 が違う → 拒否", run(*base, "--decide-json", str(dec(False, ck="2" * 64, name="dec_ck")), "--dry") == 3))
    ok.append(("採否の JSON の ckpt_sha256 が同じ → OK", run(*base, "--decide-json", str(dec(False, ck=ck_sha, name="dec_ck2")), "--dry") == 0))
    ok.append(("選択器の版が 2.1 でない → 拒否", run("--select-json", str(sel(spec="v2_ckpt_selector/2.0", name="sel_old")), "--ckpt", str(ck), "--decide-json", str(dec(False)), "--dry") == 3))
    res = tmp / "opened"
    ok.append(("乾式は予約を作らない", not res.exists()))
    rc1 = run(*base, "--decide-json", str(dec(True)), "--dist-table", str(table), "--reserve-dir", str(res), "--out", str(res / "EXEC_CONTRACT.json"), "--job", "123")
    rec = json.loads((res / "EXEC_CONTRACT.json").read_text(encoding="utf-8")) if (res / "EXEC_CONTRACT.json").is_file() else {}
    ok.append(("予約 1 回目 OK・記録に ckpt／表／採否の sha と job", rc1 == 0 and rec.get("ckpt_sha256") == ck_sha and rec.get("dist_table", {}).get("sha256") == t_sha and rec.get("adopt") is True and rec.get("job") == "123"))
    try:
        run(*base, "--decide-json", str(dec(True)), "--dist-table", str(table), "--reserve-dir", str(res), "--out", str(res / "EXEC_CONTRACT.json"))
        rc2 = 0
    except SystemExit as e:
        rc2 = e.code
    ok.append(("予約 2 回目 → 4 で止まる・1 回目の記録は残る", rc2 == 4 and (res / "EXEC_CONTRACT.json").is_file()))
    # 比べる相手の凍結（#73）: 重み・コード・対応表の sha と val の選択の記録の照合
    ps = tmp / "ps"; (ps / "ckpts").mkdir(parents=True); (ps / "configs/experiment").mkdir(parents=True)
    w = ps / "ckpts/bD.ckpt"; w.write_bytes(b"weights-D")
    (ps / "configs/infer.yaml").write_text("a: 1\n", encoding="utf-8"); (ps / "configs/experiment/synth_maccdoa.yaml").write_text("b: 2\n", encoding="utf-8")
    cfg_tree = tree_sha256(ps / "configs")
    codes = {}
    for suf in REQUIRED_CODE["D"]:                                # RI73-04／RR73-01: 相手 D の必須ファイル（名前の末尾で照合・src/… は ps_dir の下）
        cf = (ps / suf) if suf.startswith("src/") else (tmp / f"x{suf}")
        cf.parent.mkdir(parents=True, exist_ok=True); cf.write_text(f"# {suf}\n", encoding="utf-8"); codes[cf.as_posix()] = sha256(cf)
    cmap = tmp / "cmap.json"; cmap.write_text("{}", encoding="utf-8")
    selj = tmp / "sel_gain.json"; selj.write_text(json.dumps({"selected": {"gain_db": 40, "th": 0.3, "stay": round(0.3 * 3 / 7, 4)}}), encoding="utf-8")
    have = runtime_versions()                                     # RF73-02: files=True の検査はこの python の版と照合するので、架空の凍結はその版で書く
    VER = {"torch_version": have.get("torch", "2.8.0+cu128"), "numpy_version": have.get("numpy", "2.4.6")}
    def bjson(name="b", gain=40, th=0.3, stay=None, assoc=20, wsha=None, csha=None, code=None, sel_sha=None, drop=(), ps_dir=None, tree=None, weights=None, ver=None):
        p = tmp / f"{name}.json"
        d = {"kind": "pretrained_pseldnets", "weights": (weights or w.as_posix()), "weights_sha256": wsha or sha256(w), "code": (codes if code is None else code),
             "classmap": str(cmap), "classmap_sha256": csha or sha256(cmap),
             "settings": {"gain_db": gain, "sed_threshold": th, "sed_stay": (round(th * 3 / 7, 4) if stay is None else stay), "sed_assoc": assoc,
                          "ps_dir": (ps.as_posix() if ps_dir is None else ps_dir), "configs_tree_sha256": (cfg_tree if tree is None else tree),
                          **(VER if ver is None else ver)},
             "selection": str(selj), "selection_sha256": sel_sha or sha256(selj)}
        for k in drop:
            d.pop(k, None)
        p.write_text(json.dumps({"baselines": {"D": d}}), encoding="utf-8"); return p
    ok.append(("相手の凍結 OK（重み・必須コード 5 本・対応表・選択の記録と sha・ゲイン／入り／保持・対応幅 20）", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson()), "--dry") == 0))
    ok.append(("相手の重みの sha が違う → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b2", wsha="0" * 64)), "--dry") == 3))
    ok.append(("相手のゲインが val の選択の記録と違う → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b3", gain=50)), "--dry") == 3))
    ok.append(("相手の対応表の sha が違う → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b4", csha="1" * 64)), "--dry") == 3))
    # RI73-04: 保持・対応幅・必須ファイル・選択の記録の sha も縛る
    ok.append(("RI73-04 保持が選択の記録と違う（0.9） → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b5", stay=0.9)), "--dry") == 3))
    ok.append(("RI73-04 保持が選択の記録と違う（0.05・値域は正しい） → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b5b", stay=0.05)), "--dry") == 3))
    ok.append(("RI73-04 対応幅 170 → 拒否（20 に固定）", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b6", assoc=170)), "--dry") == 3))
    part = dict(list(codes.items())[:-1])
    ok.append(("RI73-04 必須ファイル（170 行の tsv）が code に無い → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b7", code=part)), "--dry") == 3))
    ok.append(("RI73-04 code が空 → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b8", code={})), "--dry") == 3))
    ok.append(("RI73-04 選択の記録の sha が違う → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b9", sel_sha="2" * 64)), "--dry") == 3))
    ok.append(("RI73-04 selection_sha256 が無い → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b10", drop=("selection_sha256",))), "--dry") == 3))
    ok.append(("RI73-04 対応表が無い（D） → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b11", drop=("classmap", "classmap_sha256"))), "--dry") == 3))
    # RR73-01: 読込先と凍結一覧の一致（D: ps_dir・設定の木・モデルのソース／B: feat_dir の foa_wts・official_dir の公式 6 本）
    ok.append(("RR73-01 D: ps_dir が無い → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("c1", ps_dir="")), "--dry") == 3))
    ok.append(("RR73-01 D: configs の木の sha が違う → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("c2", tree="3" * 64)), "--dry") == 3))
    other = tmp / "ps_other"; (other / "ckpts").mkdir(parents=True); w2 = other / "ckpts/bD.ckpt"; w2.write_bytes(b"weights-D")
    ok.append(("RR73-01 D: 重みが ps_dir の外（同じ名前・別の置き場） → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("c3", weights=w2.as_posix())), "--dry") == 3))
    few = {k: v for k, v in codes.items() if "/src/" not in k}
    ok.append(("RR73-01 D: モデルのソース（ps_dir/src）が凍結一覧に無い → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("c4", code=few)), "--dry") == 3))
    (ps / "configs/infer.yaml").write_text("a: 2\n", encoding="utf-8")
    ok.append(("RR73-01 D: 設定ファイルの中身を変える → 木の sha で拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("c5")), "--dry") == 3))
    (ps / "configs/infer.yaml").write_text("a: 1\n", encoding="utf-8")
    # B の凍結（構造だけ・files=False でも読込先の一致を見る）
    bo = tmp / "official"; bo.mkdir(); bf = tmp / "feat"; bf.mkdir(); bo2 = tmp / "official2"; bo2.mkdir(); bf2 = tmp / "feat2"; bf2.mkdir()
    bcodes = {}
    for f in OFFICIAL_FILES_B:
        q = bo / f; q.write_text(f"# {f}\n", encoding="utf-8"); bcodes[q.as_posix()] = sha256(q)
        q2 = bo2 / f; q2.write_text(f"# other {f}\n", encoding="utf-8")
    for f in ("_baseline_B_infer.py", "_baseline_B_common.py", "_baseline_merge.py", "hysteresis_patch.py", "_prediction_integrity.py", "B_prepare_record.json"):
        q = tmp / f"y{f}"; q.write_text("#\n", encoding="utf-8"); bcodes[q.as_posix()] = sha256(q)
    wts = bf / "foa_wts"; wts.write_bytes(b"scaler"); bcodes[wts.as_posix()] = sha256(wts); (bf2 / "foa_wts").write_bytes(b"other scaler")
    wb = tmp / "B.h5"; wb.write_bytes(b"B")
    selb = tmp / "selB.json"; selb.write_text(json.dumps({"target": "B", "gains": [0], "selected": {"gain_db": 0, "th": 0.3, "stay": 0.1286}}), encoding="utf-8")
    def bj_B(name, feat_dir=None, official_dir=None, drop_ver=False):
        p = tmp / f"{name}.json"
        s = {"sed_threshold": 0.3, "sed_stay": 0.1286, "sed_assoc": 20, "gain_db": 0, "feat_dir": (bf.as_posix() if feat_dir is None else feat_dir),
             "official_dir": (bo.as_posix() if official_dir is None else official_dir), **VER}
        if drop_ver:
            s.pop("torch_version")
        p.write_text(json.dumps({"baselines": {"B": {"kind": "dcase2024_baseline_trained_v29", "weights": wb.as_posix(), "weights_sha256": sha256(wb), "code": bcodes,
                                                     "settings": s, "selection": selb.as_posix(), "selection_sha256": sha256(selb)}}}), encoding="utf-8"); return p
    ok.append(("RR73-01 B: 正しい一覧は通る（files=True）", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bj_B("d1")), "--dry") == 0))
    ok.append(("RR73-01 B: 標準化の読込先 feat_dir だけ別の置き場（同じ名前 foa_wts・一覧の sha は正しいまま） → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bj_B("d2", feat_dir=bf2.as_posix())), "--dry") == 3))
    ok.append(("RR73-01 B: 公式コードの読込先 official_dir だけ別の置き場 → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bj_B("d3", official_dir=bo2.as_posix())), "--dry") == 3))
    ok.append(("RR73-01 B: 版（torch_version）が無い → 拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bj_B("d4", drop_ver=True)), "--dry") == 3))
    try:
        check_baselines(bj_B("d5", feat_dir=bf2.as_posix()), files=False); rc_s = 0
    except ContractError:
        rc_s = 3
    ok.append(("RR73-02 構造だけの検査（files=False）でも読込先の不一致を拒否", rc_s == 3 and check_baselines(bj_B("d6"), files=False)["files_checked"] is False))
    # RF73-01: 必須ソース 8 本のうち 1 本だけ別の控え（同名・正しい sha）に差し替えた一覧 → 拒否（残り 7 本は本来の ps/src のまま）
    other_co = tmp / "other_checkout/src/models"; other_co.mkdir(parents=True)
    oc = other_co / "model_module.py"; oc.write_text("# other checkout\n", encoding="utf-8")
    swapped = {k: v for k, v in codes.items() if not k.endswith("src/models/model_module.py")}; swapped[oc.as_posix()] = sha256(oc)
    ok.append(("RF73-01 D: 必須ソース 1 本だけ別の控えの同名ファイル（sha は正しい） → 拒否（files=True）", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("e1", code=swapped)), "--dry") == 3))
    try:
        check_baselines(bjson("e2", code=swapped), files=False); rc_e2 = 0
    except ContractError:
        rc_e2 = 3
    ok.append(("RF73-01 D: 同じ差し替えを files=False でも拒否", rc_e2 == 3))
    (ps / "src/models/model_module.py").write_text("# changed\n", encoding="utf-8")
    ok.append(("RF73-01 D: 正しいパスのまま実際のモデルのソースの中身を変える → sha で拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("e3")), "--dry") == 3))
    (ps / "src/models/model_module.py").write_text("# src/models/model_module.py\n", encoding="utf-8")
    # RF73-02: ライブラリの版が凍結と違う → 予約の前（契約・乾式）で拒否（この python で torch か numpy が読めるときだけ検査できる）
    if have:
        badver = dict(VER); badver["numpy_version" if "numpy" in have else "torch_version"] = "0.0.0"
        ok.append(("RF73-02 版が凍結と違う → 乾式（予約の前）で拒否", run(*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("e4", ver=badver)), "--dry") == 3))
    else:
        ok.append(("RF73-02 版の照合（この python では torch／numpy が読めないので未確認 = 飛ばす）", True))
    res2 = tmp / "opened2"; os.mkdir(res2)          # 記録ファイル無しの予約（途中で止まった）も拒否
    try:
        run(*base, "--decide-json", str(dec(False)), "--reserve-dir", str(res2), "--out", str(res2 / "EXEC_CONTRACT.json"))
        rc3 = 0
    except SystemExit as e:
        rc3 = e.code
    ok.append(("記録の無い予約の置き場があっても拒否（途中で止まった開封を消さない）", rc3 == 4))
    # アブレーション（#74）と「無いときの理由」（#73 の約束）
    ab = tmp / "abl"; (ab / "outdoor_siren_v2final_v29_sde_w3/checkpoints").mkdir(parents=True); (ab / "outdoor_siren_v2final_v29_sde_w3_causal/checkpoints").mkdir(parents=True)
    last = ab / "outdoor_siren_v2final_v29_sde_w3/checkpoints/last.ckpt"; last.write_bytes(b"last-99")
    last_c = ab / "outdoor_siren_v2final_v29_sde_w3_causal/checkpoints/last.ckpt"; last_c.write_bytes(b"last-causal")
    cks = ab / "scratch.ckpt"; cks.write_bytes(b"scratch"); ckn = ab / "noheight.ckpt"; ckn.write_bytes(b"noheight")
    s_sel = sel(cand_sha=sha256(cks), name="sel_scratch", cand_ckpt="runs/outdoor_siren_v2final_v29_sde_w3_abl_scratch_causal/checkpoints/epoch_079.ckpt")
    n_sel = sel(cand_sha=sha256(ckn), name="sel_noheight", cand_ckpt="runs/outdoor_siren_v2final_v29_sde_w3_abl_noheight_causal/checkpoints/epoch_119.ckpt", height="no height input (ablation noheight, #74)")
    common = [*base, "--decide-json", str(dec(False)), "--baselines-json", str(bjson("b5"))]
    all4 = ["--ablation", "scratch", str(cks), str(s_sel), "--ablation", "noheight", str(ckn), str(n_sel), "--ablation", "nocausalft", str(last), "--ablation", "nohyst", str(ck)]
    ok.append(("アブレーション 4 本 OK（scratch／noheight は選択器の JSON・nocausalft は基準学習の last.ckpt・nohyst は主の ckpt）", run(*common, *all4, "--require", "D,scratch,noheight,nocausalft,nohyst", "--dry") == 0))
    ok.append(("scratch の ckpt の sha が選択器の候補と違う → 拒否", run(*common, "--ablation", "scratch", str(ckn), str(s_sel), "--dry") == 3))
    ok.append(("noheight なのに選択器の条件が「高さの入力なし」でない → 拒否", run(*common, "--ablation", "noheight", str(ckn), str(sel(cand_sha=sha256(ckn), name="sel_nh_bad", cand_ckpt="runs/x_abl_noheight_causal/checkpoints/epoch_119.ckpt")), "--dry") == 3))
    ok.append(("scratch なのに候補の run 名に _abl_scratch が無い → 拒否", run(*common, "--ablation", "scratch", str(cks), str(sel(cand_sha=sha256(cks), name="sel_sc_bad")), "--dry") == 3))
    ok.append(("scratch に選択器の JSON が無い → 拒否", run(*common, "--ablation", "scratch", str(cks), "--dry") == 3))
    ok.append(("nocausalft が因果 ft の run の last.ckpt → 拒否", run(*common, "--ablation", "nocausalft", str(last_c), "--dry") == 3))
    ok.append(("nohyst の ckpt が主と違う → 拒否", run(*common, "--ablation", "nohyst", str(cks), "--dry") == 3))
    ok.append(("--require にある名前が無い（理由なし） → 拒否", run(*common, "--require", "B,D", "--dry") == 3))
    ok.append(("無い名前に理由があれば通る", run(*common, "--require", "B,D", "--missing-ok", "B", "学習が間に合わなかった（試験）", "--dry") == 0))
    ok.append(("揃っている名前に --missing-ok を付ける → 拒否", run(*common, "--require", "D", "--missing-ok", "D", "x", "--dry") == 3))
    res3 = tmp / "opened3"
    rc4 = run(*common, *all4, "--require", "B,D,scratch,noheight,nocausalft,nohyst", "--missing-ok", "B", "学習が間に合わなかった（試験）", "--reserve-dir", str(res3), "--out", str(res3 / "EXEC_CONTRACT.json"))
    rec3 = json.loads((res3 / "EXEC_CONTRACT.json").read_text(encoding="utf-8")) if (res3 / "EXEC_CONTRACT.json").is_file() else {}
    ok.append(("記録にアブレーション 4 本の sha・選んだ epoch・無い相手の理由が残る",
               rc4 == 0 and set(rec3.get("ablations", {})) == set(ABL_TAGS) and rec3["ablations"]["scratch"]["selected_epoch"] == 99 and rec3["ablations"]["nocausalft"]["run"] == "outdoor_siren_v2final_v29_sde_w3"
               and rec3["ablations"]["nohyst"]["ckpt_sha256"] == ck_sha and rec3.get("required", {}).get("missing") == {"B": "学習が間に合わなかった（試験）"}))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
