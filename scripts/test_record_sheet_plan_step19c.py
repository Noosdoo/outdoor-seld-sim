"""記録シートが出す本数の検査のコマンドを、実際に step19c に通す結合試験（Astra 監査 2026-10-10 AS10-01）。

なぜ: step19c の --plan は既定の辞書（PLAN_DEFAULT）のうち、書いたキーだけを上書きする。変更票 #56 で D をやめたあと、
記録シートのコマンドから D を消しただけでは既定の D:8 が残り、正しい 187 本でも「区分 D: 0 テイク（計画 8）」の警告・
--strict で不合格になった。記録シートの文字列だけを見る jsdom の試験ではこれを見つけられないので、ここで実際に通す。

使い方: DynamicSound/.venv/Scripts/python.exe scripts/test_record_sheet_plan_step19c.py
（outdoor_seld_e2e で実行。偽の注釈 CSV は一時フォルダに作り、終わったら消す）
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "md/design/実録_記入用CSV_2026-09-05/収録記録シート_v19.html"
STEP19C = ROOT / "scripts/step19c_ann_validate.py"
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")   # step19c の出力を utf-8 で受け取る（Windows の既定は cp932）


def sheet_plans() -> list[str]:
    """記録シートの cmdText にある step19c の --plan と --opportunity の値（本番・家デモ）。"""
    html = SHEET.read_text(encoding="utf-8")
    found = re.findall(r"step19c_ann_validate\.py --ann [^ ]+ --cut (--plan [^ ]+ --opportunity [^ ']+)", html)
    assert len(found) == 2, f"step19c の行が 2 つ（本番・家デモ）見つからない: {found}"
    return found


def fake_ann(path: Path) -> None:
    """学内 176 本（A 32・B 4・C 16・E 12〔E2 8 ＋ 走りながらのクラクション E3 4・10/27〕・H 60・G 20・歩く比較 32・#69 r14）と負例露出 100 分の偽の注釈（Astra の再現と同じ形）。"""
    plan = {"A": 32, "B": 4, "C": 16, "E": 12, "H": 60, "G": 20, "歩行": 32}
    cls_of = {"A": "car_drive", "B": "car_drive", "C": "bike_bell", "E": "horn", "H": "bicycle", "G": "none", "歩行": "car_drive"}
    rows = []
    for kind, n in plan.items():
        for i in range(n):
            rows.append({"clip_id": f"fake_{kind}_{i}", "event_id": "1", "trial": "1", "class": cls_of[kind],
                         "quadrant": "B", "t_start": "2", "t_cpa": "8",   # 2026-10-12: 時刻 0 の正例は Fable の安全装置（S4・#59 r6）が止めるので、ありうる時刻にした
                         "take_id": f"{kind}_{i}", "区分": kind,
                         "状態": ("歩行" if kind == "歩行" and i % 2 else "静止"),
                         "pair_id": (str(i // 2) if kind == "歩行" else ""),
                         "横距離m": "2"})
    rows.append({"clip_id": "fake_exposure", "event_id": "1", "trial": "1", "class": "none", "quadrant": "", "t_start": "0",
                 "t_cpa": "6000", "take_id": "exposure", "区分": "負例露出", "状態": "静止", "pair_id": "", "横距離m": ""})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ok = True
    plans = sheet_plans()
    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / "fake_187_ann.csv"
        fake_ann(ann)
        for arg in plans:
            plan = re.search(r"--plan ([^ ]+)", arg).group(1)
            opp = re.search(r"--opportunity ([^ ]+)", arg).group(1)
            cmd = [sys.executable, str(STEP19C), "--ann", str(ann), "--plan", plan, "--opportunity", opp, "--strict"]
            p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=ENV)
            out = p.stdout + p.stderr
            # --opportunity X=8 は計画の合計に足されて「計画 195」と出る（X は機会枠で 0 本でも不合格にしない）。判定は警告 0 件で通ること
            good = p.returncode == 0 and "エラーなし（警告0件）" in out and "D=0" in out and "区分D" not in out
            print(("ok  " if good else "FAIL"), f"step19c --plan {plan} --opportunity {opp} --strict → 終了コード {p.returncode}")
            if not good:
                ok = False
                print(out[-1500:])
        # 書かないと既定の D:8 が残ることも確かめる（AS10-01 の再現。これが通るようになったら step19c の既定が変わった＝この試験を見直す）
        no_d = "A=32,B=4,C=16,E=8,H=60,G=20,歩行=32"
        p = subprocess.run([sys.executable, str(STEP19C), "--ann", str(ann), "--plan", no_d, "--opportunity", "X=8", "--strict"],
                           capture_output=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=ENV)
        reproduced = p.returncode != 0 and "区分D" in (p.stdout + p.stderr)
        print(("ok  " if reproduced else "NOTE"), "D を書かないと既定の D:8 が残る（AS10-01 の再現）" if reproduced else "D を書かなくても通った: step19c の既定が変わった。この試験の前提を見直す")
    print("ALL OK" if ok else "NG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
