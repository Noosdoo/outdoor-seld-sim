# -*- coding: utf-8 -*-
"""migrate_clip_ids.py — dataset 置き場の識別子を旧名（fold1_room1_mix0001）から新名（train_syn_clip00001）へ移す（2026-09-22）。

内容は 1 バイトも変えないファイル名の変更。名前を中に持つファイル（scene.json の row・plan の csv・manifest・relabel_D14.json）だけ書き換え、
旧版を <name>.pre_rename として同じ場所に残す。改名前後の全ファイルの sha256 を台帳 rename_ledger_<日付>.json に残し、--verify で照合する。
対応の無い名前（fold3・fold10〜13 など）は --archive-unmapped <dir> に同じ構造で移す（改名しない）。指定が無ければ止まる。

使い方:
  python scripts/migrate_clip_ids.py <dataset_dir> --dry-run                 # 何をどう変えるかを数えるだけ
  python scripts/migrate_clip_ids.py <dataset_dir> [--archive-unmapped <dir>] [--no-audio-sha]
  python scripts/migrate_clip_ids.py <dataset_dir> --verify                  # 台帳と照合（改名後）
規則 = src/outdoor_seld/clip_ids.py（to_new / split_to_new）。手順書 = md/design/識別子の改名_手順書_2026-09-22.md §5。
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from outdoor_seld.clip_ids import RE_NEW, is_new, split_to_new, to_new  # noqa: E402

FILE_SUBDIRS = ("foa", "metadata", "masks", "metadata_dist", "metadata_dist_h", "metadata_orig", "metadata_dist_orig")
DIR_SUBDIRS = ("work",)
MANIFEST_RENAME = {"manifest_fold2_val.csv": "manifest_val_syn.csv"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def map_name(stem: str):
    """旧名 → 新名。対応が無ければ None。新名はそのまま。"""
    try:
        return to_new(stem)
    except ValueError:
        return None


def plan_items(ds: Path) -> list:
    """置き場の中で「名前を中に持つ」ファイル（書き換え対象）。"""
    items = []
    plan = ds / "plan"
    if plan.exists():
        items += sorted(plan.glob("*.csv"))
    items += sorted(ds.glob("manifest_*.csv"))                 # 置き場直下の manifest（out/_eval_aux_v12 など）
    if (ds / "relabel_D14.json").exists():
        items.append(ds / "relabel_D14.json")
    return items


def collect(ds: Path, audio_sha: bool) -> dict:
    """置き場の全対象を列挙し、(kind, old_path, new_path, sha) を返す。"""
    entries, unmapped = [], []
    for sub in FILE_SUBDIRS:
        d = ds / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.name.endswith(".pre_rename"):
                continue
            new = map_name(p.stem)
            if new is None:
                unmapped.append(p); continue
            h = None if (not audio_sha and p.suffix == ".flac") else sha256(p)
            entries.append(("file", p, d / f"{new}{p.suffix}", h))
    for sub in DIR_SUBDIRS:
        d = ds / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_dir():
                continue
            new = map_name(p.name)
            if new is None:
                unmapped.append(p); continue
            for q in sorted(p.rglob("*")):
                if q.is_file() and not q.name.endswith(".pre_rename"):
                    rel = q.relative_to(p)
                    h = None if (not audio_sha and q.suffix == ".flac") else sha256(q)
                    entries.append(("workfile", q, d / new / rel, h))
            entries.append(("dir", p, d / new, None))
    return {"entries": entries, "unmapped": unmapped}


def rewrite_scene_json(p: Path) -> bool:
    """work/<clip>/scene.json の row.clip_id・row.split を新名に。書き換えたら True。旧版は .pre_rename に残す。"""
    s = json.loads(p.read_text(encoding="utf-8"))
    row = s.get("row") or {}
    old_id, old_split = row.get("clip_id"), row.get("split")
    if old_id is None:
        return False
    new_id = map_name(old_id)
    if new_id is None or (new_id == old_id and split_to_new(old_split) == old_split):
        return False
    shutil.copyfile(p, p.with_name(p.name + ".pre_rename"))
    row["clip_id"] = new_id
    if old_split is not None:
        row["split"] = split_to_new(old_split)
    p.write_text(json.dumps(s, indent=2, default=str))        # 生成器（step11_v13_render）と同じ書式
    return True


def rewrite_csv(p: Path) -> tuple[int, Path]:
    """plan / manifest の csv: clip_id と split の列を新名に（他の列・行順・列順は不変）。戻り = (書き換えた行数, 新しいパス)。"""
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames or [], list(rd)
    n = 0
    for r in rows:
        if "clip_id" in r:
            new = map_name(r["clip_id"])
            if new is not None and new != r["clip_id"]:
                r["clip_id"] = new; n += 1
        if "split" in r and r["split"] in ("fold1", "fold2", "fold20"):
            r["split"] = split_to_new(r["split"]); n += 1
    dst = p.with_name(MANIFEST_RENAME.get(p.name, p.name))
    if n == 0 and dst == p:
        return 0, p
    shutil.copyfile(p, p.with_name(p.name + ".pre_rename"))
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    if dst != p:
        p.unlink()
    return n, dst


def rewrite_relabel(p: Path) -> int:
    """relabel_D14.json: clip の一覧と sha 表の鍵（ファイル名）を新名に。sha の値は不変（中身が変わらないため）。"""
    def conv(x):
        if isinstance(x, str):
            stem, ext = (x.rsplit(".", 1) + [""])[:2] if "." in x else (x, "")
            new = map_name(stem)
            return (f"{new}.{ext}" if ext else new) if new else x
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, dict):
            return {conv(k): conv(v) for k, v in x.items()}
        return x
    before = p.read_text(encoding="utf-8")
    after = json.dumps(conv(json.loads(before)), ensure_ascii=False, indent=1)
    if json.loads(before) == json.loads(after):
        return 0
    shutil.copyfile(p, p.with_name(p.name + ".pre_rename"))
    p.write_text(after, encoding="utf-8")
    return 1


def main() -> int:
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 2
    ds = Path(a[0]).resolve()
    dry, verify = "--dry-run" in a, "--verify" in a
    audio_sha = "--no-audio-sha" not in a
    archive = Path(a[a.index("--archive-unmapped") + 1]).resolve() if "--archive-unmapped" in a else None
    assert ds.is_dir(), ds
    ledger_path = ds / f"rename_ledger_{dt.date.today().isoformat()}.json"

    if verify:
        ledgers = sorted(ds.glob("rename_ledger_*.json"))
        assert ledgers, "台帳が無い"
        L = json.loads(ledgers[-1].read_text(encoding="utf-8"))
        rewritten = set(L["rewritten"])
        bad = n_checked = 0
        for e in L["entries"]:
            if e["kind"] == "dir":
                if not Path(e["new"]).is_dir():
                    bad += 1; print("NG dir 無し:", e["new"])
                continue
            q = Path(e["new"])
            if not q.is_file():
                bad += 1; print("NG 無し:", q); continue
            if str(q) in rewritten:                       # 名前を中に持つファイル（scene.json 等）は書き換え済み = 旧版 .pre_rename と比べる
                pre = q.with_name(q.name + ".pre_rename")
                if e["sha"] is not None and (not pre.is_file() or sha256(pre) != e["sha"]):
                    bad += 1; print("NG 書き換え前の写し（.pre_rename）が台帳と合わない:", q)
                continue
            if e["sha"] is not None:
                n_checked += 1
                if sha256(q) != e["sha"]:
                    bad += 1; print("NG sha 不一致:", q)
        for r in L["rewritten"]:
            if not Path(r).is_file():
                bad += 1; print("NG 書き換えファイル無し:", r)
        print(f"verify {ds.name}: entries {len(L['entries']):,}（sha 照合 {n_checked:,}）/ rewritten {len(L['rewritten'])} / NG {bad}")
        return 1 if bad else 0

    col = collect(ds, audio_sha)
    entries, unmapped = col["entries"], col["unmapped"]
    n_files = sum(1 for e in entries if e[0] != "dir")
    n_dirs = sum(1 for e in entries if e[0] == "dir")
    n_same = sum(1 for e in entries if e[1] == e[2])
    print(f"{ds.name}: 対象 {n_files:,} ファイル・{n_dirs:,} フォルダ（既に新名 {n_same:,}）・対応の無い名前 {len(unmapped):,}（{', '.join(sorted({p.name.split('_')[0] for p in unmapped})) or '-'}）")
    plans = plan_items(ds)
    print(f"  書き換え候補: {[p.name for p in plans]}")
    if unmapped and archive is None and not dry:
        print("対応の無い名前がある。--archive-unmapped <dir> で退避先を指定する。中止"); return 3
    if dry:
        ex = [e for e in entries if e[1] != e[2]][:3]
        for e in ex:
            print(f"  例: {e[1].relative_to(ds)} -> {e[2].relative_to(ds)}")
        return 0

    # 1) 対応の無い名前を退避（改名しない・同じ構造）
    for p in unmapped:
        dst = archive / p.relative_to(ds)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
    if unmapped:
        print(f"  退避: {len(unmapped):,} 件 -> {archive}")
    # 2) 改名（ファイル → フォルダの順。フォルダの中のファイルは親ごと動く）
    n_mv = 0
    for kind, old, new, _ in entries:
        if kind == "file" and old != new:
            assert not new.exists(), f"移動先が既にある: {new}"
            os.rename(old, new); n_mv += 1
    for kind, old, new, _ in entries:
        if kind == "dir" and old != new:
            assert not new.exists(), f"移動先が既にある: {new}"
            os.rename(old, new); n_mv += 1
    print(f"  改名: {n_mv:,} 件")
    # 3) 中身に名前を持つファイルの書き換え
    rewritten = []
    for kind, old, new, _ in entries:
        if kind == "dir" and (new / "scene.json").is_file() and rewrite_scene_json(new / "scene.json"):
            rewritten.append(str(new / "scene.json"))
    for p in plans:
        if p.suffix == ".csv":
            n, dst = rewrite_csv(p)
            if n:
                rewritten.append(str(dst))
        elif p.name == "relabel_D14.json" and rewrite_relabel(p):
            rewritten.append(str(p))
    print(f"  書き換え: scene.json {sum(1 for r in rewritten if r.endswith('scene.json')):,} / plan・manifest・relabel {sum(1 for r in rewritten if not r.endswith('scene.json'))}")
    # 4) 台帳
    ledger = {"dataset": str(ds), "date": dt.datetime.now().isoformat(timespec="seconds"), "audio_sha": audio_sha,
              "archive": str(archive) if archive else None, "n_unmapped": len(unmapped),
              "entries": [{"kind": k, "old": str(o), "new": str(n), "sha": h} for k, o, n, h in entries],
              "rewritten": rewritten}
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    print(f"  台帳: {ledger_path.name}（{len(entries):,} 件）。次に --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
