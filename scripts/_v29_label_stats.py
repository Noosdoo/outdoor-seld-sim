# -*- coding: utf-8 -*-
"""_v29_label_stats.py — 描き直した束（v29）の条件別の「ラベルが残った本」の集計（Astra RR70-04・Fable 敵対的レビュー 重大 1／提案・2026-10-19）。

描いた束の work/<clip>/scene.json（row・sources の gate_*・n_label_frames）と metadata/<clip>.csv を全部読み、
  - 対象のクラス（場面の種類で決まる: 車の場面は car_drive・キックは kickboard・…・v19mix は mix の各札）ごとに、
    ラベルが 1 フレーム以上残った本の数／本数（＝可聴の本）を、場面の種類 × 圏（danger_tier）× 徐行の帯（D10・D19）× 雨の種類 × マイクの高さの帯で出す。
  - D19 は圏 × 帯（8 組）と「徐行車より近い別の車あり／なし」で分ける（敵対的レビュー 中 1）。
  - 音源ごとのゲート量（gate_snr_max_db）の分布も条件ごとに出す。
出力 = <ds>/LABEL_STATS_v29.md と LABEL_STATS_v29.csv。読むだけ（束は変えない）。eval v3 の束にも使える（層ごとの可聴の本数 = #70 r21 ②）。

使い方: python scripts/_v29_label_stats.py --ds out/dataset_outdoor_siren_v2final_v29 [--min-per-layer 5]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLS_IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3, "car_drive": 4, "crossing": 5, "kickboard": 6, "motorcycle": 7, "bicycle": 8}
MIX_TOKEN_CLS = {"car": "car_drive", "bicycle": "bicycle", "kick": "kickboard", "moto": "motorcycle", "siren": "siren", "horn": "horn",
                 "backup_beep": "backup_beep", "crossing": "crossing", "bike_bell": "bike_bell"}
SCEN_CLS = {"v11core": "car_drive", "v19fast": "car_drive", "v19turn": "car_drive", "v19platoon": "car_drive", "v19launch": "car_drive",
            "v12kick": "kickboard", "v19kickfar": "kickboard", "v12bike": "motorcycle", "v19motofar": "motorcycle",
            "v19bicycle": "bicycle", "v19bellbike": "bike_bell", "v19bellpin": "bike_bell"}   # v12train の crossing は D14 でラベルを落とす設計なので対象にしない（10/20・★の誤報を消す）


def label_frames_by_class(meta: Path) -> dict:
    out = defaultdict(set)
    if not meta.is_file():
        return out
    for line in meta.read_text(encoding="utf-8").splitlines():
        g = line.split(",")
        if len(g) >= 3 and g[0].strip().isdigit():
            out[int(g[1])].add(int(g[0]))
    return out


def targets_of(row: dict) -> list:
    """この本で『見つけてほしい』クラスの一覧（電車の車両は背景なので除く）。"""
    scen = row.get("scenario", "")
    if scen == "v19mix":
        toks = [t for t in (row.get("mix") or "").split("+") if t and t != "train"]
        return sorted({MIX_TOKEN_CLS[t] for t in toks})
    cls = SCEN_CLS.get(scen)
    out = set()
    if scen == "v11core":
        if row.get("car_side") in ("L", "R") and int(row.get("n_car") or 0) > 0:
            out.add("car_drive")
        for k in ("w1_class", "w2_class", "w3_class"):
            if row.get(k):
                out.add(row[k])
    elif cls:
        out.add(cls)
    return sorted(out)


def band_label(row: dict) -> str:
    if str(row.get("d19", "")) == "1":
        return f"D19 {row.get('d19_tier')} 帯{row.get('d19_band')}"
    if str(row.get("close_slow", "")) == "1":
        v = float(row.get("cs_speed_kmh") or 0)
        b = 1 if v < 7.5 else 2 if v < 10 else 3 if v < 12.5 else 4
        return f"D10 critical 帯{b}"
    return ""


def mic_bin(row: dict) -> str:
    z = float(row.get("mic_z") or 1.5)
    return "mic 1.4〜1.75" if z < 1.75 else "mic 1.75〜2.1"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ds = Path(argv[argv.index("--ds") + 1]) if "--ds" in argv else ROOT / "out/dataset_outdoor_siren_v2final_v29"
    min_per = int(argv[argv.index("--min-per-layer") + 1]) if "--min-per-layer" in argv else 5
    scenes = sorted((ds / "work").glob("*/scene.json"))
    if not scenes:
        print(f"scene.json が無い: {ds}"); return 1
    agg = defaultdict(lambda: [0, 0, []])          # key -> [本数, 可聴の本数, SNR 最大の一覧]
    rows_out = []
    for p in scenes:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print("読めない:", p, e); continue
        row = s.get("row") or {}; clip = p.parent.name
        lab = label_frames_by_class(ds / "metadata" / f"{clip}.csv")
        tier = row.get("danger_tier") or "-"; rain = row.get("rain") or "なし"; mb = mic_bin(row); band = band_label(row); scen = row.get("scenario", "")
        # 徐行車より近い別の車（D10／D19）・徐行車自身の可聴（track 0 の gate_snr_max_db ≥ 0・ラベルはクラス単位なので別の車と区別するため・10/20）
        closer = ""; src_audible = None
        if band:
            cars = [x for x in s.get("sources", []) if x.get("class") == "car_drive" and x.get("kind") == "vehicle"]
            t0 = next((x for x in cars if x.get("track", 0) == 0), None)
            if t0 is not None and t0.get("gate_snr_max_db") is not None:
                src_audible = float(t0["gate_snr_max_db"]) >= 0.0
            if t0 is not None and len(cars) > 1:
                lat0 = abs(float(np.array(t0["wp"], float)[0, 2]))
                others = [abs(float(np.array(x["wp"], float)[0, 2])) for x in cars if x is not t0 and len(x["wp"]) == 2]
                closer = "別の車が近い" if any(o < lat0 for o in others) else "別の車は遠い"
            else:
                closer = "車 1 台"
        for cls in targets_of(row):
            ci = CLS_IDX[cls]; n_lab = len(lab.get(ci, set()))
            srcs = [x for x in s.get("sources", []) if x.get("class") == cls and not x.get("no_label")]
            snr = max((float(x["gate_snr_max_db"]) for x in srcs if x.get("gate_snr_max_db") is not None), default=float("nan"))
            keys = [("場面×クラス", scen, cls, "", "", ""),
                    ("場面×クラス×圏", scen, cls, tier if cls == "car_drive" else "-", "", ""),
                    ("場面×クラス×雨", scen, cls, "", rain, ""),
                    ("場面×クラス×マイク高さ", scen, cls, "", "", mb)]
            if band and cls == "car_drive":
                keys.append(("徐行の帯", band, cls, "", "", ""))
                keys.append(("徐行の帯×別の車", band, cls, closer, "", ""))
                if src_audible is not None:
                    k2 = ("徐行車自身", band, cls, "", "", ""); a2 = agg[k2]; a2[0] += 1; a2[1] += int(src_audible)
                    a2[2].append(float(t0["gate_snr_max_db"]))
            for k in keys:
                a = agg[k]; a[0] += 1; a[1] += int(n_lab > 0)
                if np.isfinite(snr):
                    a[2].append(snr)
            rows_out.append({"clip_id": clip, "split": row.get("split", ""), "scenario": scen, "mix": row.get("mix", ""), "target": cls, "danger_tier": tier, "band": band, "closer_car": closer,
                             "slow_car_audible": ("" if src_audible is None or cls != "car_drive" else int(src_audible)),
                             "rain": rain, "mic_bin": mb, "label_frames": n_lab, "gate_snr_max_db": f"{snr:.2f}" if np.isfinite(snr) else ""})
    # 出力
    with open(ds / "LABEL_STATS_v29.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows_out)
    L = [f"# 条件別の「ラベルが残った本」（{ds.name}・scene {len(scenes):,} 本・2026-10-19 の道具 _v29_label_stats.py）", "",
         "- 可聴 = 対象のクラスのラベルが 1 フレーム以上残った本（可聴ゲート: 音源の 1 フレームの A 特性の受音量が背景〔暗騒音＋雨〕以上）。SNR = 音源の最大 SNR（scene.json の gate_snr_max_db）の中央値",
         f"- #70 r21 ②: 可聴の本が {min_per} 本未満の層は値を出さず本数だけ書く（★）。D19 の層は「別の車が近い／遠い」でも分ける（敵対的レビュー 中 1）", ""]
    def sec(title, pred):
        L.append(f"## {title}"); L.append("")
        for k in sorted(k for k in agg if pred(k)):
            n, a, snrs = agg[k]; med = f"{float(np.median(snrs)):+.1f}" if snrs else "-"
            star = "★" if a < min_per else ""
            label = "・".join(x for x in k[1:] if x and x != "-")
            L.append(f"- {label}: 可聴 {a:,}/{n:,}（{100 * a / max(n, 1):.0f}%）・SNR 中央 {med} dB {star}")
        L.append("")
    sec("1. 場面 × クラス", lambda k: k[0] == "場面×クラス")
    sec("2. 車の場面 × 圏", lambda k: k[0] == "場面×クラス×圏" and k[2] == "car_drive")
    sec("3. 徐行の帯（D10 至近・D19 注意／安全 × 帯）", lambda k: k[0] == "徐行の帯")
    sec("4. 徐行の帯 × 徐行車より近い別の車", lambda k: k[0] == "徐行の帯×別の車")
    sec("4b. 徐行車自身の可聴（track 0 の gate SNR ≥ 0 dB・別の車のラベルを数えない・eval v3 の D19 の層はこれで報告する〔#70 r21 ②〕）", lambda k: k[0] == "徐行車自身")
    sec("5. 場面 × クラス × 雨", lambda k: k[0] == "場面×クラス×雨")
    sec("6. 場面 × クラス × マイクの高さ", lambda k: k[0] == "場面×クラス×マイク高さ")
    (ds / "LABEL_STATS_v29.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40])); print(f"-> {ds / 'LABEL_STATS_v29.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
