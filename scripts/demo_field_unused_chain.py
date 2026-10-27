# -*- coding: utf-8 -*-
"""開発用の学内の本（台帳で使わないの本・デモの組）の、読みから正解までの段（2026-10-27 Opus・一覧 md §10）。評価の本と同じ道具で正解を作る。

入口は固定記録（md/design/demo_field_unused_fixed_record.json）の本だけ（Astra R5-30-03・置き場のフォルダの中を拾わない）。始める前に固定記録の確かめ（demo_field_unused_fix の --check と同じ）が一致であること。
日付ごとに（置き場 out/_demo_field_unused_149/<日付>/chain/）:
  1. events_scored.csv: events_demo.csv の行に、本人の読みを写す（拍手_録音側秒 = audio_reads の clap_audio_s・拍手_動画側秒 = sync の video_s・
     ベルの本（class bike_bell）の ラップ秒 = audio_reads の bell_onset_s〔鳴り始め〕）。拍手の読みが無い本と、ベルの本で鳴り始めの読みが無い本は「読み待ち」で外す（一覧に出す）。
  2. step19l_when_where（reads_A.csv・sessions_demo.csv・events_scored.csv）→ when_where_A.csv（いつ）・reads_mid_A.csv（どこ）
  3. step19d_field_csv_to_ann（--when-csv）→ 用途 = 調整用なので ann_orig_tuning.csv（評価の注釈 ann_orig には入れない）
  4. 切り出し: 最接近 t_cpa を 8 秒目に置く 10 秒（step19b の plan_event_cut・録音の端では詰める）を、準備で変換した全長の音（conv/・149 の約束・向きの直しずみ）から切る。
     名前は固定記録の clip_id（tune_field_clip8NNNN・評価の eval_field_ と別）。→ <置き場>/infer/clips/<clip_id>.flac と ann_all.csv（clip_id・cut_offset_s つき）
  5. step19k_field_perception（既定 = 評価と同じ・範囲 ±10 m・向きの窓 2.5〜1.5 s・--sync）→ perception/gt/<clip_id>.csv
     ベルの正解は --warn-interval（chain/warn_interval.csv = 波形で読んだ鳴り始め b〜鳴り終わり e）で鳴っている間だけに絞る（2026-10-27・本人 12:35 (c)）。
     ベルの本で鳴り終わりの読みが無い本は「読み待ち」で外す。
     ベルの本には自転車の走行音の正解も付ける（本人 10/27 13:09「d つける」）: 前輪の読みを cls bicycle としても step19l に通し、ふつうの自転車の本と同じ作り方
     （車軸の中点・±10 m・向きの窓）。注釈と切り出しはベルの行のまま（切り出しの位置は変わらない）。
  6. 除外区間（除外区間メモの [入り, 出)・全長の録音の秒）を cut_offset で切り出しの秒へ直す（field_exclusion.to_clip）→ exclusions.csv
最後に <置き場>/infer/ に、推論の入力（clips/・height_table.csv〔clip_id, mic_z〕）と、正解（gt/）・ann_all.csv・exclusions.csv・waiting.csv（読み待ち）をまとめる。
推論はサーバの demo_bike_check_infer.sbatch（DEMO_DIR=<infer>・DEMO_PREFIX=tune_field_・N_CLIPS=本数）。採点は別の段（scripts/demo_field_unused_score.py・作るのはこのあと）。
使い方: python scripts/demo_field_unused_chain.py [--base out/_demo_field_unused_149] [--record md/design/demo_field_unused_fixed_record.json] [--raw-dir raw]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import demo_field_unused_fix as FX  # noqa: E402
import field_exclusion as XC  # noqa: E402

DUR = 10.0


def read_csv(p) -> list:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list, cols=None) -> None:
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def warn_intervals(ev: list, aud: dict) -> list:
    """ベルの本の正解の区間（2026-10-27・本人 12:35 (c)）: 波形で読んだ鳴り始め（b）〜鳴り終わり（e）・全長の録音の秒 → step19k --warn-interval の行。"""
    rows, seen = [], set()
    for e in ev:
        o = e["原本"].strip()
        if (e.get("class") or "").strip() != "bike_bell" or o in seen:
            continue
        seen.add(o)
        a = aud[o]
        rows.append({"take": o, "class": "bike_bell", "on_s": f"{float(a['bell_onset_s']):.4f}", "end_s": f"{float(a['bell_end_s']):.4f}"})
    return rows


def bell_bicycle_reads(reads: list, ev: list) -> list:
    """ベルの本の自転車の正解（2026-10-27・本人 13:09「d つける」）: ベルを鳴らす自転車は実際に走っているので、ベルの本の前輪の読み（cls bike_bell）を
    cls bicycle の行としても写す。step19l がふつうの自転車の本と同じに車軸の中点へずらし（session の備考の軸の間）、step19k が ±10 m の範囲の正解を作る。
    events には自転車の行を足さない（注釈・切り出しはベルの行のまま＝切り出しの位置は変わらない・step19d は使わない「いつ」の行を読まない）。"""
    bell = {e["原本"].strip() for e in ev if (e.get("class") or "").strip() == "bike_bell"}
    return [dict(r, cls="bicycle") for r in reads if (r.get("take") or "").strip() in bell and (r.get("cls") or "").strip().lower() == "bike_bell"]


def fill_events(ev: list, takes: dict, aud: dict, sync: dict) -> tuple[list, list]:
    """events_demo の行に本人の読みを写す。戻り = (写した行, 読み待ち [(原本, 理由)])。
    ベルの本は鳴り始め（b）と鳴り終わり（e）の両方が要る（2026-10-27・正解の区間 = 鳴り始め〜鳴り終わり）。"""
    out, waiting = [], []
    bad = set()
    for e in ev:
        o = (e.get("原本") or "").strip()
        if o not in takes:
            continue
        a = aud.get(o, {})
        if a.get("clap_audio_s") is None or o not in sync:
            bad.add((o, "拍手の読み（波形 a と動画 c）が無い"))
            continue
        x = dict(e)
        x["拍手_録音側秒"] = f"{float(a['clap_audio_s']):.4f}"
        x["拍手_動画側秒"] = f"{float(sync[o]['video_s']):.3f}"
        if (e.get("class") or "").strip() == "bike_bell":
            if a.get("bell_onset_s") is None:
                bad.add((o, "ベルの鳴り始めの読み（波形 b）が無い"))
                continue
            if a.get("bell_end_s") is None:
                bad.add((o, "ベルの鳴り終わりの読み（波形 e）が無い"))
                continue
            if float(a["bell_end_s"]) <= float(a["bell_onset_s"]):
                bad.add((o, f"ベルの鳴り終わり {float(a['bell_end_s']):.3f} s が鳴り始め {float(a['bell_onset_s']):.3f} s 以前"))
                continue
            x["ラップ秒"] = f"{float(a['bell_onset_s']):.4f}"
        out.append(x)
    bad_o = {o for o, _ in bad}
    out = [x for x in out if x["原本"].strip() not in bad_o]
    return out, sorted(bad)


def run(cmd: list, cwd=E2E) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(cwd))
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    print(f"  {Path(cmd[1]).name}: rc {r.returncode}・{tail[:200]}")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=E2E / "out" / "_demo_field_unused_149")
    ap.add_argument("--record", type=Path, default=FX.DEF_RECORD)
    ap.add_argument("--list", type=Path, default=FX.DEF_LIST)
    ap.add_argument("--raw-dir", type=Path, default=E2E / "raw")
    ap.add_argument("--skip-record-check", action="store_true", help="試験用")
    a = ap.parse_args(argv)
    py = sys.executable
    base = a.base.resolve()
    if not a.skip_record_check:
        old = json.loads(a.record.read_text(encoding="utf-8"))
        per, whole = FX.compare(FX.content_of(old), FX.build(a.list.resolve(), base, (a.raw_dir / "ledger").resolve()))
        if per or whole:
            print(f"❌ 固定記録と今のファイルが違う（本ごと {len(per)}・全体 {len(whole)}）→ demo_field_unused_fix.py で確かめてから")
            return 3
    takes_all = FX.fixed_takes(a.record)
    infer = base / "infer"
    if infer.exists():
        shutil.rmtree(infer)
    (infer / "clips").mkdir(parents=True)
    (infer / "gt").mkdir()
    import soundfile as sf
    from step19b_realsmoke_cut import plan_event_cut
    all_ann, all_ex, waiting, heights = [], [], [], []
    for d in sorted({t["日付"] for t in takes_all}):
        dd = base / d
        ch = dd / "chain"
        if ch.exists():
            shutil.rmtree(ch)
        ch.mkdir()
        takes = {t["原本"]: t for t in takes_all if t["日付"] == d}
        aud = json.loads((dd / "reads" / "audio_reads_A.json").read_text(encoding="utf-8"))
        sync = {r["take"]: r for r in read_csv(dd / "reads" / "sync_A.csv")}
        ev, wt = fill_events(read_csv(dd / "events_demo.csv"), takes, aud, sync)
        waiting += [{"日付": d, "原本": o, "clip_id": takes[o]["clip_id"], "理由": why} for o, why in wt]
        print(f"== {d}: 本 {len(takes)}・読みがそろった {len({e['原本'] for e in ev})}・読み待ち {len(wt)} {[o for o, _ in wt]}")
        if not ev:
            continue
        write_csv(ch / "events_scored.csv", ev, list(read_csv(dd / "events_demo.csv")[0].keys()))
        ok_o = {e["原本"].strip() for e in ev}
        rd_all = read_csv(dd / "reads" / "reads_A.csv")
        rd_ok = [r for r in rd_all if (r.get("take") or "").strip() in ok_o]                 # 読み待ちの本は reads からも外す
        write_csv(ch / "reads_A.csv", rd_ok + bell_bicycle_reads(rd_ok, ev), list(rd_all[0].keys()))   # ベルの本は自転車の行も（本人 13:09）
        r = run([py, str(E2E / "scripts" / "step19l_when_where.py"), "--reads", str(ch / "reads_A.csv"), "--session", str(dd / "sessions_demo.csv"),
                 "--events", str(ch / "events_scored.csv"), "--out", str(ch)])
        if r.returncode != 0:
            return 4
        r = run([py, str(E2E / "scripts" / "step19d_field_csv_to_ann.py"), "--session", str(dd / "sessions_demo.csv"), "--events", str(ch / "events_scored.csv"),
                 "--audio-dir", str(a.raw_dir), "--out", str(ch / "ann_orig.csv"), "--when-csv", str(ch / "when_where_A.csv")])
        if r.returncode != 0 or not (ch / "ann_orig_tuning.csv").is_file():
            print("❌ 注釈（ann_orig_tuning.csv）ができなかった"); return 5
        ann = read_csv(ch / "ann_orig_tuning.csv")
        idx = {r["原本"]: r for r in read_csv(dd / "index.csv")}
        memo = {r["原本"]: r for r in read_csv(dd / "除外区間メモ.csv")} if (dd / "除外区間メモ.csv").is_file() else {}
        ann_d = []
        by_orig = {}
        for r in ann:
            by_orig.setdefault(Path(r["orig_file"]).name, []).append(r)
        for o, rows in sorted(by_orig.items()):
            t = takes[o]
            conv = dd / idx[o]["conv"]
            info = sf.info(str(conv))
            cpas = [float(r["t_cpa"]) for r in rows if (r.get("t_cpa") or "").strip() and r.get("class") != "none"]
            if not cpas:
                waiting.append({"日付": d, "原本": o, "clip_id": t["clip_id"], "理由": "注釈に最接近の秒（t_cpa）が無い"}); continue
            off = plan_event_cut(min(cpas), info.duration, DUR)
            x, fs = sf.read(str(conv), dtype="float64", always_2d=True, start=int(round(off * info.samplerate)), frames=int(round(DUR * info.samplerate)))
            if len(x) < int(round(DUR * info.samplerate)):
                waiting.append({"日付": d, "原本": o, "clip_id": t["clip_id"], "理由": f"切り出しが 10 秒に足りない（{len(x) / fs:.2f} s）"}); continue
            sf.write(str(infer / "clips" / f"{t['clip_id']}.flac"), x, fs, subtype="PCM_24")
            for r in rows:
                ann_d.append(dict(r, clip_id=t["clip_id"], cut_offset_s=f"{off:.3f}", 区分=t["区分"], 日付=d))
            heights.append({"clip_id": t["clip_id"], "mic_z": rows[0].get("mic_z") or "1.5"})
            m = memo.get(o)
            if m and (m.get("除外の入り_音の秒") or "").strip():
                iv = XC.to_clip([(m["除外の入り_音の秒"], (m.get("除外の出_音の秒") or "").strip() or None)], off, info.duration)
                for s_, e_ in iv:
                    all_ex.append({"clip_id": t["clip_id"], "原本": o, "入り_録音の秒": m["除外の入り_音の秒"], "出_録音の秒": m.get("除外の出_音の秒", ""),
                                   "入り_クリップの秒": f"{s_:.3f}", "出_クリップの秒": f"{e_:.3f}", "①": m.get("自転車が最後まで通った（○／×）", "")})
        write_csv(ch / "ann_all.csv", ann_d)
        write_csv(ch / "warn_interval.csv", warn_intervals(ev, aud), ["take", "class", "on_s", "end_s"])
        r = run([py, str(E2E / "scripts" / "step19k_field_perception.py"), "--reads", str(ch / "reads_mid_A.csv"), "--ann", str(ch / "ann_all.csv"),
                 "--out", str(ch / "perception"), "--sync", str(dd / "reads" / "sync_A.csv"), "--warn-interval", str(ch / "warn_interval.csv")])
        if r.returncode != 0:
            return 6
        for g in (ch / "perception" / "gt").glob("*.csv"):
            shutil.copy2(g, infer / "gt" / g.name)
        all_ann += ann_d
    write_csv(infer / "ann_all.csv", all_ann) if all_ann else None
    write_csv(infer / "height_table.csv", heights, ["clip_id", "mic_z"]) if heights else None
    write_csv(infer / "exclusions.csv", all_ex, ["clip_id", "原本", "入り_録音の秒", "出_録音の秒", "入り_クリップの秒", "出_クリップの秒", "①"])
    write_csv(infer / "waiting.csv", waiting, ["日付", "原本", "clip_id", "理由"])
    n = len(list((infer / "clips").glob("*.flac")))
    print(f"→ {infer}: クリップ {n} 本・正解 {len(list((infer / 'gt').glob('*.csv')))}・除外区間 {len(all_ex)}・読み待ち {len(waiting)}")
    for w in waiting:
        print(f"  読み待ち: {w['日付']} {w['原本']}（{w['clip_id']}）: {w['理由']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
