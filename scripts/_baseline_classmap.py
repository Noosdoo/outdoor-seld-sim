# -*- coding: utf-8 -*-
"""比べる相手（変更票 #73）のクラスの対応表を作り・確かめる（2026-10-20・Fable）。

- 相手 A（PANNs・AudioSet 527 クラス）と 相手 D（事前学習だけの PSELDNets・170 クラス〔tsv の行順 = 出力の index〕）の出力を、
  この研究の 9 クラス（0 Siren・1 Horn・2 BackupBeep・3 BikeBell・4 CarDrive・5 Crossing・6 Kickboard・7 Motorcycle・8 Bicycle）へ写す表。
- 対応はラベル名で書き（変更票 #73 の文と同じ）、ここで index に解決して JSON（md/design/baselines_class_map_v1.json）に残す。表の元ファイルの sha256 も記録。
- 名前が表に無ければ止まる（黙って落とさない）。対象外のクラスは "absent" に理由つきで書く。
使い方:
  python scripts/_baseline_classmap.py build   → JSON を書く（元の表: out/_baselines/audioset_class_labels_indices.csv・out/_baselines/cls_indices_train_pseldnets170.tsv）
  python scripts/_baseline_classmap.py --selftest
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OURS = {0: "Siren", 1: "Horn", 2: "BackupBeep", 3: "BikeBell", 4: "CarDrive", 5: "Crossing", 6: "Kickboard", 7: "Motorcycle", 8: "Bicycle"}
AUDIOSET_CSV = ROOT / "out/_baselines/audioset_class_labels_indices.csv"
PSELD_TSV = ROOT / "out/_baselines/cls_indices_train_pseldnets170.tsv"
OUT_JSON = ROOT / "md/design/baselines_class_map_v1.json"

# 相手 A: AudioSet の display_name → この研究のクラス（複数は最大値で畳む）。変更票 #73 の文と同じ
MAP_A = {
    "Siren": ["Siren", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"],
    "Horn": ["Vehicle horn, car horn, honking", "Air horn, truck horn"],
    "BackupBeep": ["Reversing beeps"],
    "BikeBell": ["Bicycle bell"],
    "CarDrive": ["Car", "Car passing by", "Motor vehicle (road)", "Vehicle"],
    "Motorcycle": ["Motorcycle"],
    "Bicycle": ["Bicycle"],
}
ABSENT_A = {"Crossing": "AudioSet に踏切の警報音のクラスが無い（事実 §1）", "Kickboard": "AudioSet に電動キックボードのクラスが無い（事実 §1）"}
NOTE_A = {"Bicycle": "AudioSet の \"Bicycle\" は自転車の音の総称（子は Bicycle bell だけ）で走行音そのものではない"}

# 相手 D: PSELDNets 事前学習の 170 クラス（tsv の 3 列目のラベル・行順 = index）→ この研究のクラス
MAP_D = {
    "Siren": ["Siren"],
    "Horn": ["Vehicle horn, car horn, honking, toot"],
    "BikeBell": ["Bicycle bell"],
    "CarDrive": ["Car", "Car passing by", "Motor vehicle (road)", "Vehicle"],
    "Motorcycle": ["Motorcycle"],
    "Bicycle": ["Bicycle, tricycle"],
}
ABSENT_D = {"BackupBeep": "事前学習に Reversing beeps が無い（事実 §4）", "Crossing": "事前学習に踏切の警報音が無い（事実 §4）", "Kickboard": "事前学習に電動キックボードが無い（事実 §4）"}


def sha256(path: Path) -> str:
    """元の表の sha256。改行を LF にそろえてから取る（git の autocrlf で CRLF に変わっても同じ値になる）。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_audioset(path: Path) -> dict[str, int]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 527:
        raise SystemExit(f"AudioSet の表が 527 行でない: {len(rows)}")
    return {r["display_name"]: int(r["index"]) for r in rows}


def read_pseld(path: Path) -> dict[str, int]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) != 170:
        raise SystemExit(f"PSELDNets の表が 170 行でない: {len(lines)}")
    out = {}
    for idx, line in enumerate(lines):                     # Synthesis クラスと同じ: 行順 = index（id 列は飛びがある）
        out[line.split("\t")[2]] = idx
    return out


def resolve(mapping: dict, absent: dict, table: dict[str, int], who: str) -> dict:
    res = {}
    for ours, names in mapping.items():
        idxs = []
        for n in names:
            if n not in table:
                raise SystemExit(f"{who}: ラベル {n!r} が表に無い")
            idxs.append(table[n])
        res[ours] = {"ours_index": [k for k, v in OURS.items() if v == ours][0], "source_labels": names, "source_index": idxs}
    covered = set(mapping) | set(absent)
    if covered != set(OURS.values()):
        raise SystemExit(f"{who}: 9 クラスの対応か対象外が揃っていない: {sorted(set(OURS.values()) - covered)}")
    return res


# v2（Astra R73-04・本人 2026-10-20 23:34「5 おけ」）: 車は "Car" と "Car passing by" だけ。"Vehicle"・"Motor vehicle (road)" は船・鉄道・バイクも含む広い親クラスなので使わない
MAP_A_V2 = dict(MAP_A, CarDrive=["Car", "Car passing by"])
MAP_D_V2 = dict(MAP_D, CarDrive=["Car", "Car passing by"])
OUT_JSON_V2 = ROOT / "md/design/baselines_class_map_v2.json"
NOTE_V2 = {"CarDrive": "v2: 広い親クラス（Vehicle・Motor vehicle (road)）を外した（Astra R73-04）。Car の子クラス（クラクション等）は別の対象に写しているので Car 自体は走行音の近似対応",
           "Bicycle": "AudioSet の \"Bicycle\" は自転車の音の総称（子は Bicycle bell だけ）で走行音そのものではない"}


def build(out_json: Path = OUT_JSON, v2: bool = False) -> dict:
    A = read_audioset(AUDIOSET_CSV); D = read_pseld(PSELD_TSV)
    ma, md = (MAP_A_V2, MAP_D_V2) if v2 else (MAP_A, MAP_D)
    rec = {"version": "baselines_class_map/2" if v2 else "baselines_class_map/1",
           "made": ("2026-10-21・変更票 #73・Astra R73-04（車は Car と Car passing by だけ）・確定評価の前に凍結" if v2 else "2026-10-20・変更票 #73（下書き）・確定評価の前に凍結"),
           "ours": {str(k): v for k, v in OURS.items()},
           "A_panns_audioset": {"table": str(AUDIOSET_CSV.relative_to(ROOT)), "table_sha256": sha256(AUDIOSET_CSV), "n_source_classes": 527, "reduce": "max",
                                "map": resolve(ma, ABSENT_A, A, "A"), "absent": ABSENT_A, "notes": (NOTE_V2 if v2 else NOTE_A)},
           "D_pseldnets_pretrained": {"table": str(PSELD_TSV.relative_to(ROOT)), "table_sha256": sha256(PSELD_TSV), "n_source_classes": 170, "index_rule": "tsv の行順（Synthesis クラスの enumerate と同じ）",
                                      "reduce": "同じ対象クラスへ複数が写るときは向き 20° 以内の複製をまとめ、向きの違う音を長さ順に 3 トラックまで（_baseline_merge.py）",
                                      "map": resolve(md, ABSENT_D, D, "D"), "absent": ABSENT_D, "notes": (NOTE_V2 if v2 else {})}}
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rec


def selftest() -> int:
    ok = []
    rec = build(Path(__file__).resolve().parent.parent / "out/_baselines/_classmap_selftest.json")
    a, d = rec["A_panns_audioset"]["map"], rec["D_pseldnets_pretrained"]["map"]
    ok.append(("A: サイレン 5 クラスの index（396・322・323・324・325）", a["Siren"]["source_index"] == [396, 322, 323, 324, 325]))
    ok.append(("A: クラクション（308・318）・バックブザー（319）・ベル（203）・車（307・314・306・300）・オートバイ（326）・自転車（341）",
               a["Horn"]["source_index"] == [308, 318] and a["BackupBeep"]["source_index"] == [319] and a["BikeBell"]["source_index"] == [203]
               and a["CarDrive"]["source_index"] == [307, 314, 306, 300] and a["Motorcycle"]["source_index"] == [326] and a["Bicycle"]["source_index"] == [341]))
    ok.append(("A: 踏切とキックボードは対象外", set(rec["A_panns_audioset"]["absent"]) == {"Crossing", "Kickboard"}))
    # 行順の index（id 列ではない）: Siren は 56 行目 = 55・Horn 131 行目 = 130・Bicycle bell 153 行目 = 152・Car 155 行目 = 154・Car passing by 170 行目 = 169・Motor vehicle 1 行目 = 0・Vehicle 127 行目 = 126・Motorcycle 71 行目 = 70・Bicycle, tricycle 9 行目 = 8
    ok.append(("D: Siren=55・Horn=130・BikeBell=152・car=154/169/0/126・Motorcycle=70・Bicycle=8（行順の index・id 列〔58・137・160…〕ではない）",
               d["Siren"]["source_index"] == [55] and d["Horn"]["source_index"] == [130] and d["BikeBell"]["source_index"] == [152]
               and d["CarDrive"]["source_index"] == [154, 169, 0, 126] and d["Motorcycle"]["source_index"] == [70] and d["Bicycle"]["source_index"] == [8]))
    ok.append(("D: バックブザー・踏切・キックボードは対象外", set(rec["D_pseldnets_pretrained"]["absent"]) == {"BackupBeep", "Crossing", "Kickboard"}))
    ok.append(("9 クラスの番号と名前", rec["ours"] == {str(k): v for k, v in OURS.items()}))
    # v2（R73-04）: 車は Car と Car passing by だけ・ベル単独／バイク単独／列車単独を車に写さない
    r2 = build(Path(__file__).resolve().parent.parent / "out/_baselines/_classmap_v2_selftest.json", v2=True)
    a2, d2 = r2["A_panns_audioset"]["map"], r2["D_pseldnets_pretrained"]["map"]
    ok.append(("v2 A: 車は Car（307）と Car passing by（314）だけ", a2["CarDrive"]["source_index"] == [307, 314]))
    ok.append(("v2 D: 車は Car（154）と Car passing by（169）だけ", d2["CarDrive"]["source_index"] == [154, 169]))
    A_tab = read_audioset(AUDIOSET_CSV); D_tab = read_pseld(PSELD_TSV)
    car_a = set(a2["CarDrive"]["source_index"]); car_d = set(d2["CarDrive"]["source_index"])
    ok.append(("v2 A: ベル単独（Bicycle bell）・バイク単独（Motorcycle）・列車単独（Train）・Vehicle・Motor vehicle は車に写らない",
               not ({A_tab["Bicycle bell"], A_tab["Motorcycle"], A_tab["Train"], A_tab["Vehicle"], A_tab["Motor vehicle (road)"]} & car_a)))
    ok.append(("v2 D: Bicycle bell・Motorcycle・Train・Vehicle・Motor vehicle (road) は車に写らない",
               not ({D_tab["Bicycle bell"], D_tab["Motorcycle"], D_tab["Train"], D_tab["Vehicle"], D_tab["Motor vehicle (road)"]} & car_d)))
    ok.append(("v2 の版の印", r2["version"] == "baselines_class_map/2"))
    # 無いラベルは止まる
    try:
        resolve({"Siren": ["No such label"]}, {k: "" for k in OURS.values() if k != "Siren"}, read_audioset(AUDIOSET_CSV), "A"); ok.append(("無いラベルで止まる", False))
    except SystemExit:
        ok.append(("無いラベルで止まる", True))
    try:
        resolve({"Siren": ["Siren"]}, {}, read_audioset(AUDIOSET_CSV), "A"); ok.append(("9 クラスが揃わないと止まる", False))
    except SystemExit:
        ok.append(("9 クラスが揃わないと止まる", True))
    for name, c in ok:
        print(("PASS " if c else "FAIL ") + name)
    k = sum(c for _, c in ok); print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    return 0 if k == len(ok) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        v2 = "--v2" in sys.argv
        r = build(OUT_JSON_V2 if v2 else OUT_JSON, v2=v2); print("->", OUT_JSON_V2 if v2 else OUT_JSON, "A:", {k: v["source_index"] for k, v in r["A_panns_audioset"]["map"].items()}, "D:", {k: v["source_index"] for k, v in r["D_pseldnets_pretrained"]["map"].items()}); sys.exit(0)
    print(__doc__); sys.exit(2)
