# -*- coding: utf-8 -*-
"""比べる相手の凍結 JSON（md/design/baselines_selected.json・変更票 #73）から、推論の台本に渡す環境変数を作る（Fable・2026-10-21）。
台本（_v2final_members.sh）はこの出力を eval して相手の推論を回す = 凍結した値（重み・対応表・ゲイン・閾値）以外を台本に書かない。

使い方:
  python _baselines_env.py <baselines_selected.json> --list          相手の名前を JSON の順に 1 行（空白区切り）
  python _baselines_env.py <baselines_selected.json> --env <名前>     その相手の export 行（シェルの引用つき・~/ は展開）
  python _baselines_env.py --selftest
相手の種類（kind）と出す変数:
  pretrained_pseldnets_*（D）: CAUSAL_CKPT・BASELINE_GAIN_DB・CAUSAL_SED_TH／STAY／ASSOC・BASELINE_CLASSMAP・BASELINE_EXPERIMENT・CAUSAL_DATASET・
      BASELINE_TSV170（code の中の cls_indices_train_pseldnets170.tsv・無ければ止める）・BASELINE_KIND
  dcase2024_*（B）: B_MODEL（学習した .h5）・B_FEAT_DIR（標準化の重み wts の置き場）・B_OFFICIAL_COMMIT・CAUSAL_SED_TH／STAY／ASSOC・BASELINE_KIND
本体 §3-6（2026-10-26・M11）:
  - B の倍率: settings.gain_db が 0 以外なら B_INPUT_GAIN_DB を出す（v30 の B は主と同じ決めた倍率・「B にゲインは無い」は v29 までの話）。0 か無ければ出さない（旧のまま）。
    settings.dist_encoding = log1p なら B_DIST_ENCODING=log1p（linear か無ければ出さない）。
  - クラス別のしきい値（B・D・§3-5）: settings.thresholds_json と thresholds_sha256 があれば CAUSAL_THRESHOLDS_JSON・BASELINE_THRESHOLDS_SHA256 を出し、
    定数の CAUSAL_SED_TH／STAY は出さない（D の #73 の定数〔0.1／0.0429〕は旧の記録として JSON に残る・推論は表を使う）。CAUSAL_SED_ASSOC は出す。
終了コード: 0 = OK・3 = JSON の中身が足りない／名前が無い
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TSV170_SUFFIX = "cls_indices_train_pseldnets170.tsv"


class EnvError(Exception):
    pass


def _num(s: dict, key: str, name: str) -> str:
    v = s.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise EnvError(f"相手 {name}: settings.{key} が数でない（{v!r}）")
    return repr(float(v)) if isinstance(v, float) else str(v)


def env_lines(B: dict, name: str) -> list[str]:
    items = B.get("baselines") or {}
    if name not in items:
        raise EnvError(f"相手 {name} が凍結 JSON に無い（ある相手: {list(items)}）")
    b = items[name]
    kind = str(b.get("kind") or "")
    s = b.get("settings") or {}
    w = str(Path(str(b.get("weights") or "")).expanduser())
    if not b.get("weights"):
        raise EnvError(f"相手 {name}: weights が無い")
    out = {"BASELINE_KIND": kind, "CAUSAL_SED_TH": _num(s, "sed_threshold", name), "CAUSAL_SED_STAY": _num(s, "sed_stay", name), "CAUSAL_SED_ASSOC": _num(s, "sed_assoc", name)}
    th, stay, assoc = float(s["sed_threshold"]), float(s["sed_stay"]), float(s["sed_assoc"])   # RI73-04: 環境変数への変換でも値域を見る（契約と二重）
    if not (0 < th <= 1 and 0 < stay <= th):
        raise EnvError(f"相手 {name}: 入り {th}・保持 {stay} の値域が違う（入り (0,1]・保持 (0, 入り]）")
    if assoc != 20:
        raise EnvError(f"相手 {name}: 対応幅は 20 に固定（{assoc}）")
    for key in ("torch_version", "numpy_version"):                      # RR73-01: 外部ライブラリの版を固定（推論の台本が照合する）
        if not isinstance(s.get(key), str) or not s[key]:
            raise EnvError(f"相手 {name}: settings.{key} が無い")
    out["BASELINE_TORCH_VERSION"] = str(s["torch_version"]); out["BASELINE_NUMPY_VERSION"] = str(s["numpy_version"])
    if s.get("thresholds_json"):                                          # 本体 §3-5／§3-6: クラス別の表（定数は旧の記録として残すが出さない）
        tsha = s.get("thresholds_sha256")
        if not (isinstance(tsha, str) and len(tsha) == 64):
            raise EnvError(f"相手 {name}: settings.thresholds_json があるのに thresholds_sha256（64 桁）が無い")
        out.pop("CAUSAL_SED_TH"); out.pop("CAUSAL_SED_STAY")
        out["CAUSAL_THRESHOLDS_JSON"] = str(Path(str(s["thresholds_json"])).expanduser())
        out["BASELINE_THRESHOLDS_SHA256"] = tsha
    if B.get("_json_path"):                                               # RF73-01: 推論の台本が実行時に、実際に読んだファイルの sha を凍結一覧と照合するための JSON の場所
        out["BASELINE_FROZEN_JSON"] = str(Path(str(B["_json_path"])).expanduser())
    if kind.startswith("pretrained_pseldnets"):
        out["CAUSAL_CKPT"] = w
        out["BASELINE_GAIN_DB"] = _num(s, "gain_db", name)
        if not b.get("classmap"):
            raise EnvError(f"相手 {name}: classmap が無い")
        out["BASELINE_CLASSMAP"] = str(Path(str(b["classmap"])).expanduser())
        out["BASELINE_EXPERIMENT"] = str(s.get("experiment") or "synth_maccdoa")
        out["CAUSAL_DATASET"] = str(s.get("dataset") or "synth")
        tsv = [k for k in (b.get("code") or {}) if str(k).endswith(TSV170_SUFFIX)]
        if len(tsv) != 1:
            raise EnvError(f"相手 {name}: code に 170 行の tsv（{TSV170_SUFFIX}）が 1 つ要る（{len(tsv)} 個）")
        out["BASELINE_TSV170"] = str(Path(tsv[0]).expanduser())
        if not s.get("ps_dir"):
            raise EnvError(f"相手 {name}: settings.ps_dir（PSELDNets の置き場・RR73-01）が無い")
        out["PS"] = str(Path(str(s["ps_dir"])).expanduser())           # 推論の台本はこの PS から設定（configs）とモデルのソース（src）を読む
    elif kind.startswith("dcase2024"):
        out["B_MODEL"] = w
        if not s.get("feat_dir"):
            raise EnvError(f"相手 {name}: settings.feat_dir（標準化の重みの置き場）が無い")
        out["B_FEAT_DIR"] = str(Path(str(s["feat_dir"])).expanduser())
        if not s.get("official_dir"):
            raise EnvError(f"相手 {name}: settings.official_dir（公式コードの置き場・RR73-01）が無い")
        out["B_OFFICIAL"] = str(Path(str(s["official_dir"])).expanduser())   # 推論の台本はこの置き場の公式コードを import する
        out["B_OFFICIAL_COMMIT"] = str(s.get("official_commit") or "")
        if s.get("gain_db") not in (None, 0, 0.0):                        # 本体 §3-6（M11）: v30 の B は主と同じ決めた倍率を特徴の前に掛ける
            out["B_INPUT_GAIN_DB"] = _num(s, "gain_db", name)
        enc = s.get("dist_encoding")
        if enc not in (None, "linear", "log1p"):
            raise EnvError(f"相手 {name}: settings.dist_encoding は linear か log1p（{enc!r}）")
        if enc == "log1p":
            out["B_DIST_ENCODING"] = "log1p"
    else:
        raise EnvError(f"相手 {name}: kind が未対応（{kind!r}・pretrained_pseldnets_* か dcase2024_*）")
    return [f"export {k}={shlex.quote(v)}" for k, v in out.items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--env", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.json or not a.json.is_file():
        print(f"NG 凍結 JSON が無い: {a.json}")
        return 3
    try:
        B = json.loads(a.json.read_text(encoding="utf-8"))
        if not isinstance(B.get("baselines"), dict) or not B["baselines"]:
            raise EnvError("凍結 JSON に baselines が無い")
        B["_json_path"] = str(a.json.resolve())                       # RF73-01: 推論の台本に凍結 JSON の場所を渡す（実行時の sha 照合用）
        if a.list:
            print(" ".join(B["baselines"].keys()))
            return 0
        if a.env:
            print("\n".join(env_lines(B, a.env)))
            return 0
        ap.error("--list か --env <名前>")
    except EnvError as e:
        print(f"NG {e}")
        return 3
    return 0


def selftest() -> int:
    import io
    import tempfile
    from contextlib import redirect_stdout
    ok = []
    tmp = Path(tempfile.mkdtemp(prefix="bl_env_"))
    VER = {"torch_version": "2.8.0+cu128", "numpy_version": "2.4.6"}
    J = {"baselines": {
        "D": {"kind": "pretrained_pseldnets_no_outdoor", "weights": "~/x/mACCDOA.ckpt", "weights_sha256": "a" * 64,
              "code": {"~/x/_baseline_D_pretrain170_infer.py": "b" * 64, "~/x/out/_baselines/cls_indices_train_pseldnets170.tsv": "c" * 64},
              "classmap": "~/x/baselines_class_map_v2.json", "classmap_sha256": "d" * 64,
              "settings": {"gain_db": 40, "sed_threshold": 0.1, "sed_stay": 0.0429, "sed_assoc": 20, "experiment": "synth_maccdoa", "dataset": "synth", "ps_dir": "~/x/PSELDNets", **VER}},
        "B": {"kind": "dcase2024_baseline_trained_v29", "weights": "~/y/custom_v29_dev_split0_multiaccdoa_foa_model.h5", "weights_sha256": "e" * 64,
              "code": {}, "settings": {"sed_threshold": 0.3, "sed_stay": 0.1286, "sed_assoc": 20, "feat_dir": "~/y/feat", "official_dir": "~/y/official", "official_commit": "6eef450", **VER}}}}
    p = tmp / "b.json"; p.write_text(json.dumps(J), encoding="utf-8")
    home = str(Path.home())

    def run(*args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(list(args))
        return rc, buf.getvalue()

    rc, o = run(str(p), "--list")
    ok.append(("--list は JSON の順に名前", rc == 0 and o.strip() == "D B"))
    rc, o = run(str(p), "--env", "D")
    lines = o.strip().splitlines()
    ok.append(("D: ゲイン・閾値・対応表・tsv・ckpt を export（~/ は展開・引用つき）",
               rc == 0 and "export BASELINE_GAIN_DB=40" in lines and "export CAUSAL_SED_TH=0.1" in lines and "export CAUSAL_SED_STAY=0.0429" in lines
               and any(l.startswith("export CAUSAL_CKPT=") and home.replace("\\", "/") in l.replace("\\", "/") and "~" not in l for l in lines)
               and any(l.startswith("export BASELINE_TSV170=") and l.endswith(shlex.quote(str(Path("~/x/out/_baselines/cls_indices_train_pseldnets170.tsv").expanduser()))) for l in lines)
               and "export CAUSAL_DATASET=synth" in lines and "export BASELINE_EXPERIMENT=synth_maccdoa" in lines))
    rc, o = run(str(p), "--env", "B")
    lines = o.strip().splitlines()
    ok.append(("B: B_MODEL・B_FEAT_DIR・B_OFFICIAL（読込先）・閾値・公式の commit・版を export（ゲインは出さない）",
               rc == 0 and any(l.startswith("export B_MODEL=") for l in lines) and any(l.startswith("export B_FEAT_DIR=") for l in lines)
               and any(l.startswith("export B_OFFICIAL=") and l.endswith(shlex.quote(str(Path("~/y/official").expanduser()))) for l in lines)
               and "export CAUSAL_SED_TH=0.3" in lines and "export B_OFFICIAL_COMMIT=6eef450" in lines and "export BASELINE_TORCH_VERSION=2.8.0+cu128" in lines
               and not any("GAIN" in l for l in lines)))
    rc, o = run(str(p), "--env", "D")
    ok.append(("D: PS（PSELDNets の読込先・ps_dir）と版を export", rc == 0 and any(l.startswith("export PS=") and l.endswith(shlex.quote(str(Path("~/x/PSELDNets").expanduser()))) for l in o.splitlines())
               and "export BASELINE_NUMPY_VERSION=2.4.6" in o.splitlines()))
    J5 = json.loads(json.dumps(J)); del J5["baselines"]["B"]["settings"]["official_dir"]
    p5 = tmp / "b5.json"; p5.write_text(json.dumps(J5), encoding="utf-8")
    rc, o = run(str(p5), "--env", "B")
    ok.append(("B に official_dir（公式コードの読込先）が無い → 3", rc == 3))
    J6 = json.loads(json.dumps(J)); del J6["baselines"]["D"]["settings"]["ps_dir"]
    p6 = tmp / "b6.json"; p6.write_text(json.dumps(J6), encoding="utf-8")
    rc, o = run(str(p6), "--env", "D")
    ok.append(("D に ps_dir が無い → 3", rc == 3))
    rc, o = run(str(p), "--env", "A")
    ok.append(("無い名前 → 3", rc == 3))
    J2 = json.loads(json.dumps(J)); del J2["baselines"]["D"]["code"]["~/x/out/_baselines/cls_indices_train_pseldnets170.tsv"]
    p2 = tmp / "b2.json"; p2.write_text(json.dumps(J2), encoding="utf-8")
    rc, o = run(str(p2), "--env", "D")
    ok.append(("D の code に 170 行の tsv が無い → 3", rc == 3))
    J3 = json.loads(json.dumps(J)); J3["baselines"]["B"]["settings"]["gain_db"] = 20
    p3 = tmp / "b3.json"; p3.write_text(json.dumps(J3), encoding="utf-8")
    rc, o = run(str(p3), "--env", "B")
    ok.append(("B にゲイン 20（v30・本体 §3-6・M11）→ B_INPUT_GAIN_DB=20 を出す（v29 の B〔ゲインなし〕は出さない＝上の B の行）", rc == 0 and "export B_INPUT_GAIN_DB=20" in o.splitlines()))
    J7 = json.loads(json.dumps(J)); J7["baselines"]["D"]["settings"].update(thresholds_json="~/x/thresholds_v30_D.json", thresholds_sha256="f" * 64)
    J7["baselines"]["B"]["settings"].update(dist_encoding="log1p")
    p7 = tmp / "b7.json"; p7.write_text(json.dumps(J7), encoding="utf-8")
    rc7, o7 = run(str(p7), "--env", "D")
    rc7b, o7b = run(str(p7), "--env", "B")
    ok.append(("表あり（D）→ CAUSAL_THRESHOLDS_JSON と sha を出し、定数の CAUSAL_SED_TH／STAY は出さない（ASSOC は出す）・B の log1p → B_DIST_ENCODING",
               rc7 == 0 and any(l.startswith("export CAUSAL_THRESHOLDS_JSON=") and "~" not in l for l in o7.splitlines()) and f"export BASELINE_THRESHOLDS_SHA256={'f' * 64}" in o7.splitlines()
               and not any(l.startswith(("export CAUSAL_SED_TH=", "export CAUSAL_SED_STAY=")) for l in o7.splitlines()) and "export CAUSAL_SED_ASSOC=20" in o7.splitlines()
               and rc7b == 0 and "export B_DIST_ENCODING=log1p" in o7b.splitlines()))
    J8 = json.loads(json.dumps(J)); J8["baselines"]["D"]["settings"].update(thresholds_json="~/x/t.json")
    p8 = tmp / "b8.json"; p8.write_text(json.dumps(J8), encoding="utf-8")
    ok.append(("表があるのに thresholds_sha256 が無い → 3", run(str(p8), "--env", "D")[0] == 3))
    J4 = json.loads(json.dumps(J)); J4["baselines"]["D"]["settings"]["sed_stay"] = "0.04"
    p4 = tmp / "b4.json"; p4.write_text(json.dumps(J4), encoding="utf-8")
    rc, o = run(str(p4), "--env", "D")
    ok.append(("閾値が数でない → 3", rc == 3))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
