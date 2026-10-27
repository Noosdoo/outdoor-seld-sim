# -*- coding: utf-8 -*-
"""v2final の D14（列車の通過音を背景に）: scenario=v12train のクリップの crossing（cls=5）行を metadata/ と metadata_dist/ から落とす
（2026-09-14・設計凍結 §2 の規則 = step10_v18_relabel.py と同じ。v18 は v16 からの複写だったが、v2final は自分の置き場の中で行う）。

- 元のラベルは metadata_orig/ ・ metadata_dist_orig/ に残す（初回だけ複写。2 回目以降は _orig を正として作り直す = 何度通しても同じ結果）
- 追加生成（2026-09-28 D21 の 1,500 本など）で _orig に無い clip が増えたときは、その clip の今のラベルを未変換の原本として _orig に足してから規則を当てる
  （既存の _orig は触らない。--check では足さず、_orig に無いファイルがあれば NG のまま）
- 記録 relabel_D14.json: 対象 clip 集合・落とした行数・空になった clip 数・変換前後の全ファイルの sha256（対象外の行が 1 行も変わらないことをここで検査）
- 音声（foa/）・場面（work/）・masks/ は触らない。警報機の crossing（v11core）はそのまま

使い方: python scripts/step10_v2final_relabel.py [--ds out/dataset_outdoor_siren_v2final] [--check]
  --check = 変換せず、いまの metadata/ が「_orig から規則を当てた結果」と一致するかだけ見る（終了コード 0/1）
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CROSSING = 5
TRAIN_SCENARIO = "v12train"


def _arg(name, default):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply_rule(lines: list[str]) -> list[str]:
    return [l for l in lines if l.strip() and int(l.split(",")[1]) != CROSSING]


def main() -> int:
    ds = Path(_arg("--ds", str(ROOT / "out/dataset_outdoor_siren_v2final")))
    check_only = "--check" in sys.argv
    plan = ds / "plan/assignment_v2final.csv"
    rows = list(csv.DictReader(open(plan, encoding="utf-8")))
    train_clips = sorted(r["clip_id"] for r in rows if r.get("scenario") == TRAIN_SCENARIO)
    rec = {"rule": f"scenario={TRAIN_SCENARIO} の clip から class={CROSSING}（crossing）の行を落とす。警報機の crossing は保持。音声・場面・masks は不変",
           "applied_at": datetime.now().isoformat(timespec="seconds"), "plan_sha256": sha(plan), "n_train_clips": len(train_clips),
           "train_clips_sha256": hashlib.sha256("\n".join(train_clips).encode()).hexdigest(), "subsets": {}}
    rc = 0
    for sub in ("metadata", "metadata_dist"):
        cur, orig = ds / sub, ds / f"{sub}_orig"
        n_added = 0
        if not cur.exists():
            print(f"skip（無い）: {cur}"); continue
        if not orig.exists():
            if check_only:
                print(f"NG: {orig} が無い（未変換）"); rc = 1; continue
            shutil.copytree(cur, orig)      # 初回: 元を残す
        elif not check_only:
            # 追加生成（2026-09-28 D21 の kickfar 1,500 本など）: _orig に無い clip は未変換の原本なので _orig に足す。既存の _orig は触らない
            # （9/28 job 5074 は _orig が 24,000 本のまま 1,500 本が「_orig に無いファイル」になって NG で止まった）
            new = sorted(p for p in cur.glob("*.csv") if not (orig / p.name).exists())
            for p in new:
                shutil.copy2(p, orig / p.name)
            if new:
                print(f"{sub}: _orig に無い {len(new)} 本を原本として追加（追加生成分・既存の _orig は不変）")
            n_added = len(new)
        files = sorted(orig.glob("*.csv"))
        n_dropped = n_emptied = n_changed = n_same = 0
        expected = {}
        for f in files:
            lines = f.read_text(encoding="utf-8").splitlines()
            keep = apply_rule(lines) if f.stem in set(train_clips) else lines
            text = "".join(l + "\n" for l in keep)
            expected[f.name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if f.stem in set(train_clips):
                n_dropped += len(lines) - len(keep)
                n_emptied += int(not keep)
            if not check_only:
                (cur / f.name).write_bytes(text.encode("utf-8"))     # write_text は Windows で改行を \r\n にして hash が合わなくなる
        # 検査: いまの metadata が期待どおりか、対象外の clip は _orig と同一か
        bad = []
        for f in files:
            now = cur / f.name
            if not now.exists() or sha(now) != expected[f.name]:
                bad.append(f.name); continue
            if f.stem not in set(train_clips):
                (n_same if sha(now) == sha(f) else n_changed)
                if sha(now) != sha(f):
                    bad.append(f.name + "（対象外なのに変わった）")
        extra = sorted(p.name for p in cur.glob("*.csv") if not (orig / p.name).exists())
        rec["subsets"][sub] = dict(n_files=len(files), n_added_to_orig=n_added, n_dropped_rows=n_dropped, n_emptied_clips=n_emptied, n_bad=len(bad), extra_files=len(extra),
                                   sha256_orig_all=hashlib.sha256("".join(f"{f.name}:{sha(f)}\n" for f in files).encode()).hexdigest(),
                                   sha256_after_all=hashlib.sha256("".join(f"{n}:{h}\n" for n, h in sorted(expected.items())).encode()).hexdigest())
        print(f"{sub}: {len(files)} files / 落とした crossing 行 {n_dropped} / 空になった clip {n_emptied} / 不一致 {len(bad)} / _orig に無いファイル {len(extra)}")
        if bad or extra:
            rc = 1
            print("  NG 例:", (bad + extra)[:5])
    if not check_only:
        (ds / "relabel_D14.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print("->", ds / "relabel_D14.json")
    print("D14 relabel", "OK" if rc == 0 else "NG")
    return rc


if __name__ == "__main__":
    sys.exit(main())
