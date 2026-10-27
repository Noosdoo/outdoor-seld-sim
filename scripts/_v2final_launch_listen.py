# -*- coding: utf-8 -*-
"""発進（D20・v19launch）の試聴セット（2026-09-22・本人「新しく作った音は一回聞いて確かめる」）。

入力: out/v2final_launch_listen/src/<clip>.flac（サーバの束からの FOA 4ch・24 kHz）と <clip>.scene.json
出力: out/v2final_launch_listen/listen/<番号>_<clip>_<種別>.wav（ステレオ・べき圧縮・ピーク 0.9）と README_試聴.md（聞きどころの表）
ステレオ化: FOA（ACN: W, Y, Z, X）から L = W + 0.7·Y、R = W − 0.7·Y（y = 左が正）。べき圧縮 |x|^0.6（小さい発進の始まりを聞こえやすくする。
学習データそのものは無加工）。使い方: python scripts/_v2final_launch_listen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = ROOT / "out/v2final_launch_listen"
SRC, OUT = BASE / "src", BASE / "listen"
GAIN_Y, POW, PEAK = 0.7, 0.6, 0.9
VAR_JA = {"normal": "通常車", "ev": "EV（静音）", "heavy": "大型車"}
TIER_JA = {"critical": "至近", "caution": "注意", "safe": "安全"}


def to_listen(foa: np.ndarray) -> np.ndarray:
    """ステレオ化＋べき圧縮。音量の正規化は 8 本まとめて 1 つのゲイン（本どうしの大小関係を保つ）。"""
    w, y = foa[:, 0], foa[:, 1]
    st = np.stack([w + GAIN_Y * y, w - GAIN_Y * y], axis=1)
    return np.sign(st) * np.abs(st) ** POW


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, items = [], []
    for p in sorted(SRC.glob("*.flac")):
        foa, fs = sf.read(p)
        items.append((p, fs, to_listen(foa)))
    gain = PEAK / max(float(np.max(np.abs(st))) for _, _, st in items)
    for i, (p, fs, st) in enumerate(items, 1):
        s = json.loads((SRC / f"{p.stem}.scene.json").read_text(encoding="utf-8"))
        src = s["sources"][0]
        L = src["launch"]
        mic = s.get("mic", {})
        tag = f"{VAR_JA[src.get('car_variant', 'normal')]}_{TIER_JA[src['danger_tier']]}_{'歩行' if mic.get('motion') == 'walk' else '静止'}"
        # 表示名は clip 番号だけ（本人 9/22 16:42。旧名 fold1_room1_mix28028 でも新名 train_syn_clip28028 でも同じ番号）
        num = int(p.stem.rsplit("mix", 1)[-1]) if "_mix" in p.stem else int(p.stem.rsplit("clip", 1)[-1])
        name = f"{i:02d}_clip{num:05d}_{tag}.wav"
        sf.write(OUT / name, (st * gain).astype(np.float32), fs, subtype="PCM_16")
        rel = "—"
        if mic.get("motion") == "walk":
            rel = "後ろから追い越す" if L.get("same_dir_as_walk") else "正面から来る"
        else:
            rel = "前方から" if L.get("dir_x", 1.0) < 0 else "後方から"
        rows.append((name, VAR_JA[src.get("car_variant", "normal")], TIER_JA[src["danger_tier"]],
                     "歩行" if mic.get("motion") == "walk" else "静止", rel, src["t_on"], L["v_end_kmh"], round(L["accel_mps2"], 2),
                     L["start_dist_m"], src.get("min_dist_m"), src.get("t_min_dist_s"), round(s["noise"]["dba"], 1), src.get("recv_meas_db"),
                     "あり" if s.get("rain") else "なし"))
        print(name)
    md = ["# 発進（D20）試聴セット — 2026-09-22（ジョブ 4834 の束から 8 本・PROVISIONAL）", "",
          "ステレオ化 = FOA の W ± 0.7·Y、べき圧縮 |x|^0.6、音量は 8 本まとめて 1 つのゲイン（いちばん大きい本のピークが 0.9。本どうしの大小関係はそのまま）。**学習データは無加工**（この加工は聞くためだけ）。",
          "元 = `src/<clip>.flac`（FOA 4ch・24 kHz）と `src/<clip>.scene.json`。ファイル名の mix 番号 = clip_id `fold1_room1_mix<番号>` の番号（fold/room は DCASE 形式の名残で屋外では意味が無いので表示から外した）。", "",
          "| # | ファイル | 車 | 距離帯 | マイク | 車の来方 | 発音開始 t_on [s] | 終端 km/h | 加速 m/s² | 発進位置の距離 [m] | 最接近 [m] @ [s] | 暗騒音 dB(A) | 受音 dB(A) | 雨 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r[0][:2]} | `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | {r[9]} @ {r[10]} | {r[11]} | {r[12]} | {r[13]} |")
    md += ["", "## 聞きどころ", "",
           "1. **最初の 1〜3 秒は暗騒音だけ**（車は静止・無音）。t_on で車の音が現れる。ここが不自然に「パッ」と出ないか（20 ms の立ち上げあり）。",
           "2. その後 1〜4 秒で徐行の速さまで加速し、最接近（表の @ 秒）で真横を通り、遠ざかる。**発進の始まりから最接近までが自然な間隔か**。",
           "3. 通常車・大型車はエンジン音の高さが変わらない（速度非依存の近似）。**EV はモータ音の高さが最初から終端速度の値で一定**（本来は上がる。本人決定 A の近似）。",
           "4. 至近（0.6〜1.5 m）は大きく速く横切る。安全帯（3.2〜15 m）は小さく、暗騒音が大きい本（#03・#07 は 60 dB(A)）では埋もれ気味 = 想定どおり。",
           "5. 歩行の本は足音なし（合成の規約どおり）。「後ろから追い越す」は音が後ろから前へ、「正面から来る」は前から後ろへ動く。",
           "", "OK なら「発進 試聴 OK」と言ってもらえれば、PROGRESS と変更票 #36 に記録します。気になる本は番号で。"]
    (BASE / "README_試聴.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("->", OUT, "/", BASE / "README_試聴.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
