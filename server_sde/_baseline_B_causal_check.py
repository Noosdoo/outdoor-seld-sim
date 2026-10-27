# -*- coding: utf-8 -*-
"""相手 B の因果推論の検査（Astra RI73-02・2026-10-21）。サーバで走らせる（公式コード・librosa・標準化の重みが要る）。
  C1 未来を変えても過去のモデル入力が一致する: 最初の e_k 秒が同じで、そのあとだけ違う 2 本の音で、ラベルフレーム k までの窓の入力（causal_windows の出力）が完全に一致する。
     本全体の STFT から作る従来の方法（feat_of_audio → 窓）では一致しない（top_db=80 でなくても最後のフレームが 32 サンプル未来を見る）ことも示す。
  C2 top_db=None: 本の後半を大きくしても、前半の log-mel の値が変わらない（公式の top_db=80 だと変わる）。
  C3 判定時刻までの音を切った本（audio[:e_k]）に本全体の方法を掛けた結果と、feat_causal の窓 k の入力が一致する（= 定義どおり「その時刻までの音だけ」）。
使い方: B_FEAT_DIR=<標準化の重みのある置き場> python server_sde/_baseline_B_causal_check.py [--clips 3]（CAUSAL_FOA の val の本を使う・無ければ乱数の音）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("B_MODEL", "/nonexistent.h5")          # _baseline_B_infer は import 時に B_MODEL を読む → ここでは特徴の関数だけ使うので架空でよい
os.environ.setdefault("CAUSAL_FOA", "/nonexistent")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--clips", type=int, default=3); ap.add_argument("--ks", default="9,49,99")
    a = ap.parse_args()
    import _baseline_B_common as C
    C.setup_path()
    import librosa
    # 推論器の関数を、モデルを読まずに使う（モジュールの先頭でモデルを読む所より前の関数だけ要る）→ ソースから該当部分を取り出す代わりに、同じ定義をここで再現しない:
    # _baseline_B_infer を import すると B_MODEL の読み込みで止まるので、必要な関数を exec で切り出す
    src = (HERE / "_baseline_B_infer.py").read_text(encoding="utf-8")
    head = src.split("foa = Path(os.environ[\"CAUSAL_FOA\"])")[0]
    head = head.replace("MODEL = Path(os.environ[\"B_MODEL\"])", "MODEL = Path('/nonexistent.h5')")
    ns: dict = {"__name__": "_b_infer_head", "__file__": str(HERE / "_baseline_B_infer.py")}
    exec(compile(head, str(HERE / "_baseline_B_infer.py"), "exec"), ns)
    feat_of_audio, feat_causal, causal_windows = ns["feat_of_audio"], ns["feat_causal"], ns["causal_windows"]
    fc, N, SR = ns["fc"], ns["N"], ns["SR"]
    rng = np.random.default_rng(0)
    foa = Path(os.environ.get("CAUSAL_FOA", "/nonexistent"))
    wavs = []
    if foa.is_dir():
        import soundfile as sf
        for p in sorted(foa.glob("val_syn_*.flac"))[: a.clips]:
            w, sr = sf.read(str(p), dtype="float32"); assert sr == SR and w.shape[0] == N; wavs.append(w)
    while len(wavs) < a.clips:
        wavs.append((rng.standard_normal((N, 4)) * 1e-3).astype(np.float32))
    ks = [int(x) for x in a.ks.split(",")]
    ok = []
    hop = int(fc._hop_len)
    for ci, w in enumerate(wavs):
        for k in ks:
            e = 5 * (k + 1) * hop
            w2 = w.copy(); w2[e:] = (rng.standard_normal((N - e, 4)) * 0.3).astype(np.float32) if e < N else w2[e:]   # 未来だけ大きく変える
            f1, l1 = feat_causal(w); f2, l2 = feat_causal(w2)
            x1 = causal_windows(f1, l1)[k]; x2 = causal_windows(f2, l2)[k]
            same = np.array_equal(x1, x2)
            ok.append((f"C1 clip{ci} k={k}: 未来を変えても窓 k の入力が一致（feat_causal）", same or e >= N))
            if e < N:
                y1 = causal_windows(feat_of_audio(w))[k]; y2 = causal_windows(feat_of_audio(w2))[k]
                # 報告だけ（合否にしない）: 本全体の STFT の最後のフレームは 32 サンプル（Hann の端・重み ≤ 1%）だけ未来を見る。実測では標準化後の float32 で差が出ないことが多い
                print(f"[info] clip{ci} k={k}: 本全体の方法の窓 k の最大差（未来を変えたとき）= {float(np.abs(y1 - y2).max()):.3g}（feat_causal は 0）")
                # C3: 切った音に本全体の方法 → 窓 k の入力と一致
                cut = w[:e]
                fcut = feat_of_audio(cut)                            # (5(k+1), 448)
                ok.append((f"C3 clip{ci} k={k}: 判定時刻で切った音の特徴 = feat_causal の窓 k", np.allclose(causal_windows(np.concatenate([fcut, np.zeros((f1.shape[0] - fcut.shape[0], fcut.shape[1]), np.float32)]))[k], x1, atol=1e-5)))
        # C2: top_db
        audio = w.astype(np.float64) + fc._eps
        sp = fc._spectrogram(audio, int(len(audio) / hop))
        mel_none = fc._get_mel_spectrogram(sp)                      # 子クラス（top_db=None）
        w_loud = w.copy(); w_loud[N // 2:] *= 300.0
        sp2 = fc._spectrogram(w_loud.astype(np.float64) + fc._eps, int(len(audio) / hop))
        mel_none2 = fc._get_mel_spectrogram(sp2)
        half = mel_none.shape[0] // 2 - 2
        ok.append((f"C2 clip{ci}: 後半を 300 倍にしても前半の log-mel（top_db=None）が同じ", np.allclose(mel_none[:half], mel_none2[:half], atol=1e-6)))
        mag = np.abs(sp[:, :, 0]) ** 2; mag2 = np.abs(sp2[:, :, 0]) ** 2
        off = librosa.power_to_db(np.dot(mag, fc._mel_wts)); off2 = librosa.power_to_db(np.dot(mag2, fc._mel_wts))
        ok.append((f"C2b clip{ci}: 公式の top_db=80 だと前半が変わる（最大差 {float(np.abs(off[:half] - off2[:half]).max()):.1f} dB）", not np.allclose(off[:half], off2[:half], atol=1e-6)))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    n = sum(c for _, c in ok); print(f"{n}/{len(ok)} {'ALL PASS' if n == len(ok) else 'FAILED'}")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
