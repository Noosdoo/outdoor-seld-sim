# -*- coding: utf-8 -*-
"""本体 §3-6（相手 B・§9-1 の 6・§9-4 の 3・M11）と §3-5 の続き（凍結の口）の受入れ試験（2026-10-26）。
公式ベースライン（DCASE2024_seld_baseline）はサーバにだけあるので、同じ形の架空の公式モジュール（seldnet_model・train_seldnet・cls_feature_class）を一時フォルダに作り、
B_OFFICIAL をそこへ向けた別プロセスで _baseline_B_common を通す（環境変数は import のときに読むため）。

  R1 判定: 比率 0.49 → 支配でない・0.50（ちょうど）→ 支配でない・0.51 → 支配・非有限／全体 0 → 止まる
  R2 対数と逆変換: u = log1p(d)/log1p(100)・d = expm1(max(u,0)·log1p(100))・往復（0〜100 m）・u < 0 → 0・sde_patch と同じ式・torch でも同じ
  R3 1 epoch の損失の内訳（patch_loss_breakdown）: 架空の MSELoss_ADPIT（順列 3 通りの最小）で、xyz の二乗誤差 1・距離 4 の組が選ばれる場面 → 比率 = 12/21・支配・
     学習の損失の値は包む前と同じ・JSON に比率とバッチ数・公式の _each_calc が「軸 2 の平均」でない → 止まる・公式の損失に別の項がある → 止まる
  R4 ラベルの距離の符号化（log1p）: 探り当てた欄だけ u に替わる（距離が最後の欄でも最初の欄でも）・linear は何もしない・倍率 20 dB は _load_audio の波形が 10 倍
  R5 出力の復号（patch_dist_decode）: log1p のとき dist0／1／2 が m に戻る・ほかの返り値は同じ・linear は包まない・返り値が 9 個でなければ止まる
  R6 特徴の置き場の印: 書く → 同じ環境は通る・倍率や符号化が違えば止まる・印の無い旧の置き場は 0 dB・linear と読む（倍率 20 なら止まる）
  R7 文面: B の推論（倍率は 2 つの特徴の関数で eps の前・印の照合・復号・run_meta の input_gain_db／dist_encoding）・学習（印・復号・内訳）・準備（印・借りる・u の往復検査）・
     台本（B の置き場は環境で・既定 v29・候補の推論は v30 の置き場の plan を v29 と sha で照合）・B／D の推論の BASELINE_THRESHOLDS_SHA256
  R8 契約（check_baselines・ファイルを見ない検査）: B の倍率 20 は倍率の決定（decision.json の final と sha）が要る・一致で通る・final 違い／sha 違い → 止まる・
     表あり → 記録に thresholds と旧の定数（constants_superseded）・表の sha が無い → 止まる・dist_encoding が linear／log1p 以外 → 止まる
使い方: PSELDNet/PSELDNets/.venv の python（torch が要る）で python scripts/test_baseline_B_v30.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_sde"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RESULTS = []

FAKE_SELDNET = '''
import os, torch
class MSELoss_ADPIT(object):
    def __init__(self):
        self._each_loss = torch.nn.MSELoss(reduction="none")
    def _each_calc(self, output, target):
        v = self._each_loss(output, target).mean(dim=(2))
        return v * 2.0 if os.environ.get("FAKE_BAD_EACH") == "1" else v
    def __call__(self, output, target):
        out = output.reshape(output.shape[0], output.shape[1], 12, -1)
        losses = [self._each_calc(out, target[:, :, k]) for k in range(target.shape[2])]
        idx = torch.stack(losses).min(dim=0).indices
        loss = sum(l * (idx == n) for n, l in enumerate(losses)).mean()
        return loss + 0.1 if os.environ.get("FAKE_EXTRA_TERM") == "1" else loss
'''
FAKE_TRAIN = '''
import numpy as np
def get_multi_accdoa_labels(accdoa_in, nb_classes):
    if accdoa_in == "bad":
        return (1, 2, 3)
    T = 4
    out = []
    for t in range(3):
        out += [np.ones((1, T, nb_classes)), np.full((1, T, 3 * nb_classes), 0.5), np.full((1, T, nb_classes), [0.0, 0.15, 0.2381, 1.0, -0.2, 0.5, 0.7, 0.9, 0.05][:nb_classes])]
    return tuple(out)
def train_epoch(batches, criterion):
    tot, n = 0.0, 0
    for out, tgt in batches:
        loss = criterion(out, tgt)
        tot += float(loss); n += 1
    return tot / max(n, 1)
'''
FAKE_FEAT = '''
import os, numpy as np
class FeatureClass:
    def __init__(self, params=None, is_eval=False):
        self._eps = 1e-8
        self._nb_channels = 4
        self._nb_unique_classes = 9
    def load_output_format_file(self, path):
        d = {}
        for line in open(path, encoding="utf-8"):
            w = line.strip().split(",")
            if len(w) < 6:
                continue
            d.setdefault(int(w[0]), []).append([int(w[1]), int(w[2]), float(w[3]), float(w[4]), float(w[5])])
        return d
    def convert_output_format_polar_to_cartesian(self, d):
        out = {}
        for k, rows in d.items():
            out[k] = []
            for c, t, az, el, cm in rows:
                a, e = np.radians(az), np.radians(el)
                out[k].append([c, t, np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e), cm / 100.0])
        return out
    def get_adpit_labels_for_file(self, desc, nb):
        lab = np.zeros((nb, 6, 5, self._nb_unique_classes))
        first = os.environ.get("FAKE_DIST_FIRST") == "1"
        for k, rows in desc.items():
            if k >= nb:
                continue
            seen = {}
            for c, t, x, y, z, d in rows:
                j = seen.get(c, 0); seen[c] = j + 1
                lab[k, j, :, c] = [d, 1.0, x, y, z] if first else [1.0, x, y, z, d]
        return lab
'''


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)


def run_driver(tmp: Path, code: str, env: dict) -> dict:
    drv = tmp / f"drv_{abs(hash(code)) % 10**8}.py"
    drv.write_text(textwrap.dedent(code), encoding="utf-8")
    e = dict(os.environ)
    for k in ("B_DIST_ENCODING", "B_INPUT_GAIN_DB", "FAKE_BAD_EACH", "FAKE_EXTRA_TERM", "FAKE_DIST_FIRST"):
        e.pop(k, None)
    e.update({"B_OFFICIAL": str(tmp / "official"), "B_VENDOR": str(tmp / "vendor"), "PYTHONIOENCODING": "utf-8", "SERVER_SDE": str(ROOT / "server_sde"), **env})
    r = subprocess.run([sys.executable, str(drv)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=e)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        return {"_rc": r.returncode, "_err": (r.stderr or r.stdout)[-600:]}
    return dict(json.loads(lines[-1][7:]), _rc=r.returncode)


HEAD = '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["SERVER_SDE"])
import numpy as np
import _baseline_B_common as C
C.setup_path()
def stops(f, *a, **k):
    try:
        f(*a, **k); return False
    except (SystemExit, RuntimeError, ValueError):
        return True
'''


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="B_v30_"))
    (tmp / "official").mkdir(); (tmp / "vendor").mkdir()
    (tmp / "official/seldnet_model.py").write_text(FAKE_SELDNET, encoding="utf-8")
    (tmp / "official/train_seldnet.py").write_text(FAKE_TRAIN, encoding="utf-8")
    (tmp / "official/cls_feature_class.py").write_text(FAKE_FEAT, encoding="utf-8")
    for k in ("B_DIST_ENCODING", "B_INPUT_GAIN_DB"):         # R1・R2 は同じプロセス・既定の環境
        os.environ.pop(k, None)
    return _main2(tmp)


def _main2(tmp: Path) -> int:
    import numpy as np
    try:
        import _baseline_B_common as C
        v = C.loss_ratio_verdict
        def stops(f, *a):
            try:
                f(*a); return False
            except ValueError:
                return True
        r49, r50, r51 = v(0.49, 1.0), v(0.5, 1.0), v(0.51, 1.0)
        check("R1 判定: 0.49 → 支配でない・0.50 ちょうど → 支配でない・0.51 → 支配・非有限／全体 0／負 → 止まる",
              r49[1] is False and r50 == (0.5, False) and r51[1] is True and stops(v, float("nan"), 1.0) and stops(v, 1.0, 0.0) and stops(v, float("inf"), 1.0) and stops(v, -0.1, 1.0),
              str((r49, r50, r51)))
        ds = np.array([0.0, 0.3, 1.0, 2.0, 5.0, 37.5, 100.0])
        u = C.dist_encode_log(ds)
        back = C.dist_decode_log(u)
        same_sde = np.allclose(u, np.log1p(ds) / np.log1p(100.0), rtol=0, atol=1e-15)
        import torch
        tback = C.dist_decode_log(torch.tensor(u)).numpy()
        check("R2 対数と逆変換: 往復（0〜100 m）・u < 0 → 0・d < 0 → 0・sde_patch と同じ式・torch でも同じ",
              np.allclose(back, ds, rtol=1e-12, atol=1e-12) and float(C.dist_decode_log(-0.1)) == 0.0 and float(C.dist_encode_log(-1.0)) == 0.0 and same_sde
              and np.allclose(tback, ds, rtol=1e-12, atol=1e-12) and abs(float(C.dist_encode_log(2.0)) - np.log1p(2.0) / np.log1p(100.0)) < 1e-15, str(np.round(u, 4)))
        # R3 損失の内訳（別プロセス・架空の公式）
        r3code = HEAD + '''
import torch
import seldnet_model, train_seldnet
crit = seldnet_model.MSELoss_ADPIT()
B, T, Cn = 2, 3, 9
out = torch.zeros(B, T, 12 * Cn)
t0 = torch.ones(B, T, 12, Cn); t0[:, :, 3::4, :] = 2.0              # 選ばれる組: xyz の誤差 1・距離の誤差 2（二乗 4）
t1 = torch.full((B, T, 12, Cn), 3.0)
tgt = torch.stack([t1, t0], dim=2)
ref = float(crit(out, tgt))
st = C.patch_loss_breakdown(Path(os.environ["OUTJ"]), {"dist_scale": 1})
try:
    m = train_seldnet.train_epoch([(out, tgt), (out, tgt)], crit)
    J = json.loads(Path(os.environ["OUTJ"]).read_text(encoding="utf-8"))
    print("RESULT " + json.dumps({"ref": ref, "m": m, "ratio": J.get("ratio"), "dominant": J.get("dominant"), "n": J.get("n_batches"), "params": J.get("official_dist_params"), "err": None}))
except RuntimeError as e:
    print("RESULT " + json.dumps({"err": str(e)}))
'''
        a = run_driver(tmp, r3code, {"OUTJ": str(tmp / "lb.json")})
        bad1 = run_driver(tmp, r3code, {"OUTJ": str(tmp / "lb1.json"), "FAKE_BAD_EACH": "1"})
        bad2 = run_driver(tmp, r3code, {"OUTJ": str(tmp / "lb2.json"), "FAKE_EXTRA_TERM": "1"})
        check("R3 損失の内訳: 比率 = 12/21（xyz 1・距離 4 の組）・支配・学習の損失の値は包む前と同じ・JSON にバッチ数・_each_calc が平均でない → 止まる・別の項 → 止まる",
              a.get("err") is None and abs((a.get("ratio") or 0) - 12 / 21) < 1e-6 and a.get("dominant") is True and abs(a["m"] - a["ref"]) < 1e-9 and a.get("n") == 2
              and a.get("params") == {"dist_scale": 1} and "軸 2 の平均」でない" in (bad1.get("err") or "") and "再構成した損失" in (bad2.get("err") or "")
              and "軸 2 の平均」でない" in json.loads((tmp / "lb1.json").read_text(encoding="utf-8")).get("error", ""),
              str({k: a.get(k) for k in ("ratio", "dominant", "n", "_err")})[:300] + f"・止め {str(bad1.get('err'))[:30]}／{str(bad2.get('err'))[:30]}")
        # R4 ラベルの符号化と倍率
        r4code = HEAD + '''
import soundfile as sf
fc = C.feature_class({})
p = Path(os.environ["TMPD"]) / "d.csv"
p.write_text("".join(f"{k},4,0,30,0,200\\n" for k in range(10, 20)), encoding="utf-8")
cart = fc.convert_output_format_polar_to_cartesian(fc.load_output_format_file(str(p)))
lab = fc.get_adpit_labels_for_file(cart, 100)
first = os.environ.get("FAKE_DIST_FIRST") == "1"
di, ai = (0, 1) if first else (4, 0)
w = Path(os.environ["TMPD"]) / "a.wav"
sf.write(str(w), np.full((240, 4), 0.01, dtype=np.float32), 24000, subtype="FLOAT")
au, _ = fc._load_audio(str(w))
print("RESULT " + json.dumps({"dist": float(lab[15, 0, di, 4]), "act": float(lab[15, 0, ai, 4]), "xyz": [float(v) for v in lab[15, 0, (2 if first else 1):(5 if first else 4), 4]],
                              "audio": float(au[0, 0]), "probe": list(C._PROBE.get("axis", ()))}))
'''
        u2 = float(np.log1p(2.0) / np.log1p(100.0))
        l_log = run_driver(tmp, r4code, {"TMPD": str(tmp), "B_DIST_ENCODING": "log1p"})
        l_first = run_driver(tmp, r4code, {"TMPD": str(tmp), "B_DIST_ENCODING": "log1p", "FAKE_DIST_FIRST": "1"})
        l_lin = run_driver(tmp, r4code, {"TMPD": str(tmp)})
        l_gain = run_driver(tmp, r4code, {"TMPD": str(tmp), "B_INPUT_GAIN_DB": "20"})
        xyz = [float(np.cos(np.radians(30))), float(np.sin(np.radians(30))), 0.0]
        check("R4 ラベルの符号化: log1p は探り当てた欄だけ u(2.0)（最後の欄でも最初の欄でも）・act と xyz は同じ・linear は 2.0 のまま・倍率 20 dB は波形が 10 倍（eps の前）",
              abs(l_log.get("dist", 0) - u2) < 1e-9 and l_log.get("act") == 1.0 and np.allclose(l_log.get("xyz", [9]), xyz) and l_log.get("probe") == [2, 4]
              and abs(l_first.get("dist", 0) - u2) < 1e-9 and l_first.get("probe") == [2, 0] and l_lin.get("dist") == 2.0 and l_lin.get("probe") == []
              and abs(l_gain.get("audio", 0) - (0.01 * 10 + 1e-8)) < 1e-6 and abs(l_lin.get("audio", 0) - (0.01 + 1e-8)) < 1e-7,
              str([l_log.get("dist"), l_first.get("probe"), l_lin.get("dist"), l_gain.get("audio"), l_log.get("_err")]))
        # R5 出力の復号
        r5code = HEAD + '''
import train_seldnet
orig = train_seldnet.get_multi_accdoa_labels
C.patch_dist_decode()
r = train_seldnet.get_multi_accdoa_labels(None, 9)
bad = stops(train_seldnet.get_multi_accdoa_labels, "bad", 9)
print("RESULT " + json.dumps({"wrapped": train_seldnet.get_multi_accdoa_labels is not orig, "d0": [float(v) for v in np.asarray(r[2])[0, 0]], "sed": float(np.asarray(r[0]).sum()),
                              "doa": float(np.asarray(r[1]).mean()), "bad": bad}))
'''
        d_log = run_driver(tmp, r5code, {"B_DIST_ENCODING": "log1p"})
        d_lin = run_driver(tmp, r5code, {})
        raw_u = np.array([0.0, 0.15, 0.2381, 1.0, -0.2, 0.5, 0.7, 0.9, 0.05])
        check("R5 出力の復号: log1p は dist0 が m（u < 0 → 0・u = 1 → 100 m）・ほかの返り値は同じ・返り値が 9 個でない → 止まる・linear は包まない",
              d_log.get("wrapped") is True and np.allclose(d_log.get("d0", [0]), np.expm1(np.maximum(raw_u, 0) * np.log1p(100.0))) and d_log.get("sed") == 36.0 and d_log.get("doa") == 0.5
              and d_log.get("bad") is True and d_lin.get("wrapped") is False and np.allclose(d_lin.get("d0", [9]), raw_u),
              str([round(v, 3) for v in d_log.get("d0", [])][:5]) + str(d_log.get("_err", ""))[:200])
        # R6 印
        r6code = HEAD + '''
fd = Path(os.environ["FD"])
mode = os.environ["MODE"]
if mode == "write":
    C.check_feat_marker(fd, write=True)
    print("RESULT " + json.dumps({"marker": json.loads((fd / C.FEAT_MARKER).read_text(encoding="utf-8"))}))
else:
    print("RESULT " + json.dumps({"stops": stops(C.check_feat_marker, fd)}))
'''
        fd = tmp / "feat_g20"; fd.mkdir()
        w = run_driver(tmp, r6code, {"FD": str(fd), "MODE": "write", "B_INPUT_GAIN_DB": "20", "B_DIST_ENCODING": "log1p"})
        same = run_driver(tmp, r6code, {"FD": str(fd), "MODE": "check", "B_INPUT_GAIN_DB": "20", "B_DIST_ENCODING": "log1p"})
        g0 = run_driver(tmp, r6code, {"FD": str(fd), "MODE": "check", "B_DIST_ENCODING": "log1p"})
        lin = run_driver(tmp, r6code, {"FD": str(fd), "MODE": "check", "B_INPUT_GAIN_DB": "20"})
        old = tmp / "feat_old"; old.mkdir()
        old0 = run_driver(tmp, r6code, {"FD": str(old), "MODE": "check"})
        old20 = run_driver(tmp, r6code, {"FD": str(old), "MODE": "check", "B_INPUT_GAIN_DB": "20"})
        check("R6 特徴の置き場の印: 書く → 同じ環境は通る・倍率違い／符号化違いは止まる・印の無い旧の置き場は 0 dB・linear（倍率 20 なら止まる）",
              w.get("marker") == {"input_gain_db": 20.0, "dist_encoding": "log1p"} and same.get("stops") is False and g0.get("stops") is True and lin.get("stops") is True
              and old0.get("stops") is False and old20.get("stops") is True, str([w, same, g0, lin, old0, old20])[:300])
        # R7 文面
        S = {n: (ROOT / n).read_text(encoding="utf-8").replace("\r\n", "\n") for n in ("server_sde/_baseline_B_infer.py", "server_sde/_baseline_B_train.py", "server_sde/_baseline_B_prepare.py",
                                                                                  "server_sde/baseline_B_train.sbatch", "server_sde/baseline_B_prepare.sbatch", "server_sde/v2final_candidates_infer.sbatch",
                                                                                  "server_sde/_baseline_D_pretrain170_infer.py")}
        need = {"server_sde/_baseline_B_infer.py": ["C.check_feat_marker(C.FEAT)", "C.patch_dist_decode()", 'META["settings"]["input_gain_db"] = C.INPUT_GAIN_DB',
                                                    'META["settings"]["dist_encoding"] = C.DIST_ENCODING', "BASELINE_THRESHOLDS_SHA256"],
                "server_sde/_baseline_B_train.py": ["C.check_feat_marker(C.FEAT)", "C.patch_dist_decode()", "C.patch_loss_breakdown(LOSS_BD, P)", 'os.environ.get("B_LOSS_BREAKDOWN") == "1"'],
                "server_sde/_baseline_B_prepare.py": ["C.check_feat_marker(C.FEAT, write=True)", "def link_feat_from", "C.dist_decode_log(want)", '"dist_encoding": C.DIST_ENCODING'],
                "server_sde/baseline_B_train.sbatch": ["B_DATASET_DIR=${B_DATASET_DIR:-$HOME/research/_external/B_dataset2024_v29}", "B_FEAT_DIR=${B_FEAT_DIR:-$HOME/research/_external/B_feat_v29c}", "loss_breakdown_"],
                "server_sde/baseline_B_prepare.sbatch": ["B_SRC_BUNDLE=${B_SRC_BUNDLE:-$E2E/out/dataset_outdoor_siren_v2final_v29}", "$B_SRC_BUNDLE/plan/assignment_v2final.csv", "--link-feat-from"],
                "server_sde/v2final_candidates_infer.sbatch": ["PLAN30=$E2E/out/${V2FINAL_DS_NAME:-dataset_outdoor_siren_v2final_v30}/plan", "が v29 の plan と違う", "export HEIGHT_TABLE=$PLAN_DS/assignment_v2final.csv",
                                                               "N_VAL=$(($(wc -l < $PLAN_DS/manifest_val_syn_v2final.csv) - 1))"],
                "server_sde/_baseline_D_pretrain170_infer.py": ["BASELINE_THRESHOLDS_SHA256"]}
        miss = {n: [x for x in xs if x not in S[n]] for n, xs in need.items()}
        miss = {n: m for n, m in miss.items() if m}
        two = S["server_sde/_baseline_B_infer.py"].count("wav.astype(np.float64) * C.GAIN if C.INPUT_GAIN_DB != 0.0 else wav.astype(np.float64)) + fc._eps") == 2
        check("R7 文面: B の推論・学習・準備・台本（既定 v29）・候補の推論の plan の照合・B／D の表の sha の照合・倍率は 2 つの特徴の関数で eps の前", not miss and two, str(miss))
        # R8 契約
        import _conf_exec_contract as CC
        bd = tmp / "bl"; bd.mkdir()
        selb = bd / "selB.json"; selb.write_text(json.dumps({"target": "B", "gains": [0], "selected": {"gain_db": 0, "th": 0.3, "stay": 0.1286}}), encoding="utf-8")
        dec = bd / "decision.json"; dec.write_text(json.dumps({"decision": {"final": 20.0}}), encoding="utf-8")
        dec30 = bd / "decision30.json"; dec30.write_text(json.dumps({"decision": {"final": 30.0}}), encoding="utf-8")
        fdir, odir = "~/rx/B_feat", "~/rx/official"
        code = {f"{fdir}/foa_wts": "1" * 64, "~/rx/B_dataset/B_prepare_record.json": "2" * 64}
        for f in ("_baseline_B_infer.py", "_baseline_B_common.py", "_baseline_merge.py", "hysteresis_patch.py", "_prediction_integrity.py"):
            code[f"~/rx/e2e/server_sde/{f}"] = "3" * 64
        for f in CC.OFFICIAL_FILES_B:
            code[f"{odir}/{f}"] = "4" * 64

        def bj(name, **upd):
            s = {"sed_threshold": 0.3, "sed_stay": 0.1286, "sed_assoc": 20, "gain_db": 0, "feat_dir": fdir, "official_dir": odir, "torch_version": "2.8.0", "numpy_version": "2.4.6"}
            s.update(upd)
            J = {"baselines": {"B": {"kind": "dcase2024_baseline_trained_v30", "weights": "~/rx/B.h5", "weights_sha256": "5" * 64, "code": code, "settings": s,
                                     "selection": str(selb), "selection_sha256": CC.sha256(selb)}}}
            p = bd / f"{name}.json"; p.write_text(json.dumps(J), encoding="utf-8"); return p

        def cstops(p):
            try:
                CC.check_baselines(p, files=False); return False
            except CC.ContractError:
                return True
        ok_g = CC.check_baselines(bj("g20", gain_db=20, gain_decision_json=str(dec), gain_decision_sha256=CC.sha256(dec)), files=False)
        ok_t = CC.check_baselines(bj("t", thresholds_json="~/rx/thresholds_v30_B.json", thresholds_sha256="6" * 64), files=False)
        r8 = {"no_decision": cstops(bj("g20n", gain_db=20)),
              "final_differs": cstops(bj("g20f", gain_db=20, gain_decision_json=str(dec30), gain_decision_sha256=CC.sha256(dec30))),
              "sha_differs": cstops(bj("g20s", gain_db=20, gain_decision_json=str(dec), gain_decision_sha256="0" * 64)),
              "th_no_sha": cstops(bj("tn", thresholds_json="~/rx/t.json")),
              "bad_enc": cstops(bj("be", dist_encoding="sqrt")),
              "v29_ok": not cstops(bj("v29"))}
        gi = ok_g["items"]["B"]; ti = ok_t["items"]["B"]
        check("R8 契約: B の倍率 20 は倍率の決定と一致で通る（記録に gain_decision）・決定なし／final 違い／sha 違い → 止まる・表あり → thresholds と旧の定数・表の sha なし → 止まる・符号化は linear／log1p だけ・v29 の B は旧のとおり",
              all(r8.values()) and gi.get("gain_decision", {}).get("final") == 20.0 and ti.get("thresholds", {}).get("constants_superseded") == {"sed_threshold": 0.3, "sed_stay": 0.1286},
              str(r8))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ng = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(ng)}/{len(RESULTS)} {'ALL PASS' if not ng else 'FAILED'}")
    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
