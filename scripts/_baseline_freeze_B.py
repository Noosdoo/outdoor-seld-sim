# -*- coding: utf-8 -*-
"""相手 B（DCASE 2024 公式ベースライン・v29 で学習）を凍結 JSON（md/design/baselines_selected.json）に足す（変更票 #73・2026-10-21 Fable）。
D のとき（dsel_freeze.py）と同じ考え: パスは ~/ で書く（ユーザー名を残さない）・重み／標準化の重み／コード／対応表の sha を記録・閾値は val で選んだ記録（baselines_B_selection.json）と同じ値。
`_baselines_env.py --env B` が読む形（kind dcase2024_*・weights = .h5・settings.feat_dir・official_commit・sed_threshold／sed_stay／sed_assoc）。

使い方:
  python scripts/_baseline_freeze_B.py --selection md/design/baselines_B_selection.json --weights-sha <学習した .h5 の sha256（サーバで計算）> \
      --wts-sha <標準化の重み foa_wts の sha256（サーバで計算）> --prepare-sha <B_prepare_record.json の sha256> --official-commit <公式コードの commit>
      [--weights ~/research/_external/B_models_v29/custom_v29_dev_split0_multiaccdoa_foa_model.h5] [--feat-dir ~/research/_external/B_feat_v29] [--nb-epochs 250]
      [--json md/design/baselines_selected.json]
  python scripts/_baseline_freeze_B.py --selftest
決まり: 既に B があれば上書きしない（--force で置き換え・理由を note に残す）。D の項目は触らない。書いたあと _conf_exec_contract.check_baselines の形の検査（重みの有無以外）を通す。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
E = "~/research/outdoor_seld_e2e"
CODE_FILES = ("server_sde/_baseline_B_infer.py", "server_sde/_baseline_B_common.py", "server_sde/_baseline_merge.py", "server_sde/hysteresis_patch.py", "server_sde/_prediction_integrity.py")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hex64(s: str, what: str) -> str:
    s = s.strip().lower()
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise SystemExit(f"{what} の sha256 が 64 桁の 16 進でない: {s[:20]}…")
    return s


OFFICIAL_FILES = ("cls_feature_class.py", "seldnet_model.py", "train_seldnet.py", "parameters.py", "cls_compute_seld_results.py", "SELD_evaluation_metrics.py")
OFFICIAL_DIR = "~/research/_external/DCASE2024_seld_baseline"


def build_entry(sel: dict, weights: str, weights_sha: str, feat_dir: str, wts_sha: str, prepare_sha: str, commit: str, nb_epochs: int, selection_path: str, selection_sha: str,
                official_shas: dict, root: Path = ROOT, official_dir: str = OFFICIAL_DIR, torch_version: str = "", numpy_version: str = "") -> dict:
    s = sel.get("selected") or {}
    if s.get("gain_db", 0) != 0 or sel.get("target") != "B" or sel.get("gains") != [0]:
        raise SystemExit(f"選択の記録が B のもの（target B・gains [0]・ゲイン 0）でない: target={sel.get('target')} gains={sel.get('gains')} gain_db={s.get('gain_db')}")
    for k in ("th", "stay"):
        if not isinstance(s.get(k), (int, float)):
            raise SystemExit(f"選択の記録に selected.{k} が無い")
    missing = [f for f in OFFICIAL_FILES if f not in official_shas]
    if missing:
        raise SystemExit(f"公式コードの sha が足りない（--official-shas の JSON）: {missing}")
    code = {f"{E}/{f}": sha(root / f) for f in CODE_FILES}
    code[f"{feat_dir}/foa_wts"] = wts_sha                                   # 標準化の重み（train だけで fit）も凍結
    code["~/research/_external/B_dataset2024_v29/B_prepare_record.json"] = prepare_sha   # 準備の記録（層・特徴・標準化・ラベルの検査 ok）
    for f in OFFICIAL_FILES:                                                # RI73-04／RR73-01: 使う公式ソースも固定（置き場 official_dir と同じパスで = 推論が読む所）
        code[f"{official_dir}/{f}"] = _hex64(official_shas[f], f)
    if not torch_version or not numpy_version:
        raise SystemExit("--torch-version と --numpy-version が要る（サーバの venv の版・RR73-01）")
    return {"kind": "dcase2024_baseline_trained_v29", "weights": weights, "weights_sha256": weights_sha, "code": code, "selection_sha256": selection_sha,
            "settings": {"sed_threshold": s["th"], "sed_stay": s["stay"], "sed_assoc": 20, "feat_dir": feat_dir, "official_dir": official_dir, "official_commit": commit,
                         "official_repo": "partha2409/DCASE2024_seld_baseline (MIT)", "torch_version": torch_version, "numpy_version": numpy_version,
                         "nb_epochs": nb_epochs, "train_split": "fold3 = v29 train 49,682", "val_split": "fold4 = v29 val 12,420", "height": "none", "distance": "m (official cm/100)",
                         "gain_db": 0},
            "selection": selection_path,
            "selection_summary": {"mean_f_select": s.get("mean_f"), "default_mean_f": (sel.get("default") or {}).get("mean_f"), "f_per_class": s.get("f_per_class"),
                                  "check": sel.get("check_at_selected"), "check_note": sel.get("check_note")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, default=ROOT / "md/design/baselines_B_selection.json")
    ap.add_argument("--weights", default="~/research/_external/B_models_v29/custom_v29c_dev_split0_multiaccdoa_foa_model.h5")   # v29c = top_db なしの特徴で学習（RI73-02）
    ap.add_argument("--weights-sha", default=None); ap.add_argument("--wts-sha", default=None); ap.add_argument("--prepare-sha", default=None)
    ap.add_argument("--feat-dir", default="~/research/_external/B_feat_v29c"); ap.add_argument("--official-commit", default=None); ap.add_argument("--nb-epochs", type=int, default=250)
    ap.add_argument("--json", type=Path, default=ROOT / "md/design/baselines_selected.json"); ap.add_argument("--force", action="store_true"); ap.add_argument("--note", default="")
    ap.add_argument("--official-shas", type=Path, default=None, help="公式コードの sha256 の JSON {ファイル名: sha}（サーバで sha256sum した 6 本・RI73-04）")
    ap.add_argument("--official-dir", default=OFFICIAL_DIR, help="公式コードの置き場（推論が import する所・RR73-01・~/ で書く）")
    ap.add_argument("--torch-version", default=""); ap.add_argument("--numpy-version", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.weights_sha and a.wts_sha and a.prepare_sha and a.official_commit and a.official_shas):
        ap.error("--weights-sha・--wts-sha・--prepare-sha・--official-commit・--official-shas が要る（サーバで sha256sum）")
    official = json.loads(a.official_shas.read_text(encoding="utf-8"))
    sel = json.loads(a.selection.read_text(encoding="utf-8"))
    J = json.loads(a.json.read_text(encoding="utf-8"))
    if "B" in J["baselines"] and not a.force:
        print("NG 既に B がある（置き換えるなら --force と --note 理由）"); return 3
    if a.force and not a.note.strip():
        print("NG --force には --note 理由 が要る"); return 3
    rel_sel = a.selection.resolve().relative_to(ROOT.resolve()).as_posix()
    J["baselines"]["B"] = build_entry(sel, a.weights, _hex64(a.weights_sha, "weights"), a.feat_dir, _hex64(a.wts_sha, "foa_wts"), _hex64(a.prepare_sha, "B_prepare_record"),
                                      a.official_commit, a.nb_epochs, f"{E}/{rel_sel}", sha(a.selection), official,
                                      official_dir=a.official_dir, torch_version=a.torch_version, numpy_version=a.numpy_version)
    J["note"] = (J.get("note") or "").replace("B は学習のあとに足す", "B は学習のあとに足した") + (f"／B 置き換え: {a.note}" if a.force else "")
    # Astra RR73-02: 書く前に、ファイルを見ずにできる検査を**全部の相手**に掛ける（必須項目・有限・値域・選択の記録との一致・読込先と凍結一覧の一致）。
    #   通らなければ既存の JSON は変えない。サーバのファイル照合（重み・コード・設定の木の sha）はここではできないので「未確認」と明記する。
    sys.path.insert(0, str(ROOT / "server_sde"))
    import _conf_exec_contract as C  # noqa: E402
    tmp = a.json.with_suffix(".part")
    tmp.write_text(json.dumps(J, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    try:
        rec = C.check_baselines(tmp, files=False)
    except C.ContractError as e:
        tmp.unlink(missing_ok=True)
        print(f"NG 構造の検査で止まる（JSON は変えていない）: {e}"); return 3
    import os
    os.replace(tmp, a.json)
    print(f"構造の検査（手元・files=False）: 通った（相手 {sorted(rec['items'])}）。**サーバのファイル照合は未確認** → 配備後に `_conf_exec_contract.py --baselines-json` か check_baselines(files=True) で")
    print(f"OK B を書いた → {a.json}（閾値 {sel['selected']['th']}／保持 {sel['selected']['stay']}・epoch {a.nb_epochs}・commit {a.official_commit}・公式の置き場 {a.official_dir}）")
    return 0


def selftest() -> int:
    import tempfile
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="freezeB_"))
    import hashlib as _h
    sys.path.insert(0, str(ROOT / "server_sde"))
    import _conf_exec_contract as C  # noqa: E402
    sel = {"target": "B", "gains": [0], "selected": {"gain_db": 0, "th": 0.3, "stay": 0.1286, "mean_f": 0.41, "f_per_class": {"0": 0.5}}, "default": {"mean_f": 0.2}, "check_at_selected": {"mean_f": 0.4, "n_missing": 0}, "check_note": None}
    selp = ROOT / "md/design" / f"_selftest_B_selection_{tmp.name}.json"; selp.write_text(json.dumps(sel), encoding="utf-8")
    # 相手 D の架空の凍結（構造の検査が通る形・重みなどのファイルは手元に無い = 本番と同じ状況）
    ps = tmp / "ps"; (ps / "configs").mkdir(parents=True); (ps / "configs/infer.yaml").write_text("a: 1\n", encoding="utf-8")
    dsel = tmp / "D_selection.json"; dsel.write_text(json.dumps({"selected": {"gain_db": 40, "th": 0.1, "stay": 0.0429}}), encoding="utf-8")
    dcode = {f"{ps.as_posix()}/{suf}" if suf.startswith("src/") else f"{tmp.as_posix()}/x{suf}": "f" * 64 for suf in C.REQUIRED_CODE["D"]}
    D0 = {"kind": "pretrained_pseldnets_no_outdoor", "weights": f"{ps.as_posix()}/ckpts/x.ckpt", "weights_sha256": "a" * 64, "code": dcode, "classmap": "~/x/cmap.json", "classmap_sha256": "9" * 64,
          "settings": {"gain_db": 40, "sed_threshold": 0.1, "sed_stay": 0.0429, "sed_assoc": 20, "ps_dir": ps.as_posix(), "configs_tree_sha256": C.tree_sha256(ps / "configs"), "torch_version": "2.8.0+cu128", "numpy_version": "2.4.6"},
          "selection": dsel.as_posix(), "selection_sha256": _h.sha256(dsel.read_bytes()).hexdigest()}
    J0 = {"version": "baselines_selected/1", "note": "B は学習のあとに足す", "baselines": {"D": D0}}
    jp = tmp / "sel.json"; jp.write_text(json.dumps(J0), encoding="utf-8")
    offj = tmp / "official_shas.json"; offj.write_text(json.dumps({f: "e" * 64 for f in OFFICIAL_FILES}), encoding="utf-8")
    args = ["--selection", str(selp), "--json", str(jp), "--weights-sha", "b" * 64, "--wts-sha", "c" * 64, "--prepare-sha", "d" * 64, "--official-commit", "6eef450", "--official-shas", str(offj),
            "--torch-version", "2.8.0+cu128", "--numpy-version", "2.4.6"]
    try:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(args)
        o = buf.getvalue()
        J = json.loads(jp.read_text(encoding="utf-8")); B = J["baselines"].get("B", {})
        ok.append(("B を足す: kind・重み sha・閾値（選択の記録と同じ）・feat_dir・official_dir・版・commit・コード 5 本＋wts＋準備の記録＋公式 6 本・選択の記録の sha・D はそのまま・note を更新",
                   rc == 0 and B.get("kind") == "dcase2024_baseline_trained_v29" and B.get("weights_sha256") == "b" * 64 and B["settings"]["sed_threshold"] == 0.3 and B["settings"]["sed_stay"] == 0.1286
                   and B["settings"]["feat_dir"] == "~/research/_external/B_feat_v29c" and B["settings"]["official_dir"] == OFFICIAL_DIR and B["settings"]["torch_version"] == "2.8.0+cu128"
                   and B["settings"]["official_commit"] == "6eef450" and len(B["code"]) == len(CODE_FILES) + 2 + len(OFFICIAL_FILES)
                   and B["code"].get("~/research/_external/B_feat_v29c/foa_wts") == "c" * 64 and B["code"].get("~/research/_external/B_dataset2024_v29/B_prepare_record.json") == "d" * 64
                   and B["code"].get(f"{OFFICIAL_DIR}/cls_feature_class.py") == "e" * 64 and B.get("selection_sha256") == _h.sha256(selp.read_bytes()).hexdigest()
                   and J["baselines"]["D"] == D0 and "足した" in J["note"] and B["weights"].startswith("~/") and B["selection"].startswith("~/")))
        ok.append(("RR73-02: 表示は『構造の検査は通った・サーバのファイル照合は未確認』で、例外の文で合格扱いにしない", "サーバのファイル照合は未確認" in o and "重みなどが手元に無い以外は通る" not in o))
        rc2 = main(args)
        ok.append(("既に B があれば上書きしない（3）", rc2 == 3))
        rc3 = main(args + ["--force"])
        ok.append(("--force に理由が無ければ 3", rc3 == 3))
        rc4 = main(args + ["--force", "--note", "試験"])
        ok.append(("--force と理由で置き換え", rc4 == 0 and "試験" in json.loads(jp.read_text(encoding="utf-8"))["note"]))
        # RR73-02: 不正な値（入り 2・保持 0.9）は、D の重みが手元に無くても拒否し、既存の JSON を変えない
        before = jp.read_text(encoding="utf-8")
        badv = dict(sel); badv["selected"] = dict(sel["selected"], th=2, stay=0.9); selp.write_text(json.dumps(badv), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc6 = main(args + ["--force", "--note", "x"])
        ok.append(("RR73-02: 入り 2・保持 0.9 の選択の記録 → 3 で止まり JSON は変わらない・.part も残らない", rc6 == 3 and jp.read_text(encoding="utf-8") == before and not jp.with_suffix(".part").exists()))
        bad = dict(sel); bad["gains"] = [0, 20]; selp.write_text(json.dumps(bad), encoding="utf-8")
        try:
            rc5 = main(args + ["--force", "--note", "x"]); ok.append(("選択の記録が B の形でない → 止まる", rc5 != 0))
        except SystemExit as e:
            ok.append(("選択の記録が B の形でない → 止まる", e.code not in (0, None)))
        import _baselines_env as BE  # noqa: E402
        selp.write_text(json.dumps(sel), encoding="utf-8"); main(args + ["--force", "--note", "x"])
        lines = BE.env_lines(json.loads(jp.read_text(encoding="utf-8")), "B")
        ok.append(("_baselines_env.py --env B が読める（B_MODEL・B_FEAT_DIR・B_OFFICIAL・閾値）", any(l.startswith("export B_MODEL=") for l in lines) and "export CAUSAL_SED_TH=0.3" in lines
                   and any(l.startswith("export B_FEAT_DIR=") for l in lines) and any(l.startswith("export B_OFFICIAL=") for l in lines)))
    finally:
        selp.unlink(missing_ok=True)
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
