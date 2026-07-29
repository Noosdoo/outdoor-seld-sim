# -*- coding: utf-8 -*-
"""FOAクリップ(4ch)を試聴用モノラルWAV(Wチャンネル)に変換する。

Wチャンネル=無指向成分=「マイク位置で聞こえた音」に相当。
音量は試聴用に正規化(元データの物理較正レベルは保持しない)。
使い方:
    python scripts/_make_listen_wavs.py            # 全カテゴリのデモを生成
    python scripts/_make_listen_wavs.py mix0123    # v11 coreの指定クリップのみ
出力: out/listen_v11_samples/<カテゴリ別フォルダ>/ (ダブルクリックで再生可)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "listen_v11_samples"

DS = {
    "core": ROOT / "out" / "dataset_outdoor_siren_v11" / "foa",
    "v10": ROOT / "out" / "dataset_outdoor_siren_v10" / "foa",
    "v10add": ROOT / "out" / "dataset_outdoor_siren_v10_2_add" / "foa",
    "eval": ROOT / "out" / "dataset_outdoor_siren_v11_eval" / "foa",
}

# (dataset, stem, サブフォルダ, 表示名)
PICKS = [
    # --- 学習コア(v11、割当表から選定) ---
    ("core", "fold1_room1_mix0047", "00_学習コア", "01_サイレン＋車重大接近_静止"),
    ("core", "fold1_room1_mix0010", "00_学習コア", "02_クラクション＋車safe_歩行"),
    ("core", "fold1_room1_mix0017", "00_学習コア", "03_バック警告音＋車2台_重大"),
    ("core", "fold1_room1_mix0008", "00_学習コア", "04_自転車ベル＋車3台_幹線_歩行"),
    ("core", "fold1_room1_mix0003", "00_学習コア", "05_踏切＋車1台"),
    ("core", "fold1_room1_mix0004", "00_学習コア", "06_純静穏"),
    ("core", "fold1_room1_mix0050", "00_学習コア", "07_警告音なしの重大接近車_歩行"),
    # --- 評価シナリオ6種＋α(v10共用、部屋とシナリオの対応はラベルで確認済み) ---
    ("v10", "fold2_room9_mix001", "10_評価シナリオ_v10", "交差点サイレン"),
    ("v10", "fold2_room8_mix001", "10_評価シナリオ_v10", "S1_踏切通過＋背後の車"),
    ("v10", "fold2_room7_mix001", "10_評価シナリオ_v10", "S2_背後からの自転車ベル"),
    ("v10", "fold2_room6_mix001", "10_評価シナリオ_v10", "S3_駐車場のバック車"),
    ("v10", "fold2_room5_mix001", "10_評価シナリオ_v10", "S4_完全静穏"),
    ("v10", "fold2_room4_mix001", "10_評価シナリオ_v10", "S5_悪条件サイレン"),
    ("v10", "fold8_room1_mix001", "10_評価シナリオ_v10", "交通量_車2-3台＋警告音"),
    ("v10", "fold9_room1_mix001", "10_評価シナリオ_v10", "プローブ_レベル正規化単独音"),
    ("v10add", "fold2_room3_mix001", "10_評価シナリオ_v10", "幻覚検定_車なし×サイレン"),
    # --- 弱点マップ N1-N7ほか(v11評価拡張) ---
    ("eval", "fold6_room1_mix0001", "20_弱点マップ_v11評価拡張", "N1_突然出現"),
    ("eval", "fold6_room2_mix0001", "20_弱点マップ_v11評価拡張", "N2_静音EV"),
    ("eval", "fold6_room3_mix0001", "20_弱点マップ_v11評価拡張", "N3_駐車場ブザー複数"),
    ("eval", "fold6_room4_mix0001", "20_弱点マップ_v11評価拡張", "N4_高速サイレン"),
    ("eval", "fold6_room5_mix0001", "20_弱点マップ_v11評価拡張", "N5_繁華街騒音"),
    ("eval", "fold6_room6_mix0001", "20_弱点マップ_v11評価拡張", "N6_至近追い越し"),
    ("eval", "fold6_room7_mix0001", "20_弱点マップ_v11評価拡張", "N7_停車から発進"),
    ("eval", "fold4_room2_mix0001", "20_弱点マップ_v11評価拡張", "safe車_距離カーブ用"),
    ("eval", "fold4_room1_mix0001", "20_弱点マップ_v11評価拡張", "幻覚600_車なし×サイレン"),
]


def convert(ds: str, stem: str, subdir: str, label: str) -> None:
    src = DS[ds] / f"{stem}.flac"
    audio, fs = sf.read(src)          # (T, 4ch) W,Y,Z,X
    w = np.asarray(audio, np.float64)[:, 0]
    peak = float(np.max(np.abs(w)))
    dest = OUT / subdir
    dest.mkdir(parents=True, exist_ok=True)
    if peak < 1e-9:                   # 完全無音(N2静音EVの不可聴半数など)
        sf.write(dest / f"{label}_{stem}.wav", w, fs, subtype="PCM_16")
        print(f"wrote {subdir}/{label}_{stem}.wav  (完全無音クリップ)")
        return
    sf.write(dest / f"{label}_{stem}.wav", w / peak * 0.9, fs, subtype="PCM_16")
    print(f"wrote {subdir}/{label}_{stem}.wav  ({len(w)/fs:.0f}s, 増幅x{0.9/peak:.0f})")


def main() -> None:
    args = sys.argv[1:]
    if args:
        for a in args:
            stem = a if a.startswith("fold") else f"fold1_room1_{a}"
            convert("core", stem, "99_個別指定", stem)
    else:
        for ds, stem, subdir, label in PICKS:
            convert(ds, stem, subdir, label)
    print(f"\n-> エクスプローラーで {OUT} を開く")


if __name__ == "__main__":
    main()
