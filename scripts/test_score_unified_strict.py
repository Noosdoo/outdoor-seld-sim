# -*- coding: utf-8 -*-
"""統一採点器の --strict の中身の検査（依頼 2-5・Astra R1・2026-09-14）。

受け入れ: (A) 空 csv（完了記録なし）(B) 評価 manifest に 1 本も合わない csv（run_meta 無し）(C) 6 列の行が混ざった csv (F) 推論 manifest に無い ID
→ 終了コード ≠ 0。(D) 4,500 → 300 型（予測 ⊃ 評価 manifest・run_meta あり）(E) 完了記録つきの正当な無検出（空 csv）→ 終了コード 0。
使い方: python scripts/test_score_unified_strict.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)) if detail and not cond else ''}")


def run(out_md, plan, meta, items, strict=True, env=None):
    cmd = [PY, str(ROOT / "scripts/_score_unified.py"), str(out_md)] + (["--strict"] if strict else []) + ["--plan", str(plan), "--meta", str(meta)] + items
    base = {k: v for k, v in __import__("os").environ.items() if not k.startswith("NOTIFY_ATTRIB_")}   # 検査は必ず既定の環境から始める
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env={**base, "PYTHONIOENCODING": "utf-8", **(env or {})})
    return r.returncode, (r.stderr or "")[-400:]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="su_strict_"))
    try:
        plan = tmp / "plan.csv"
        plan.write_text("clip_id,split\nval_a,val\nval_b,val\ntrain_z,train\n", encoding="utf-8")   # 2026-09-22 改名（旧 fold2/fold1）
        meta = tmp / "gt"; meta.mkdir()
        for c in ("val_a", "val_b"):
            (meta / f"{c}.csv").write_text("".join(f"{k},4,0,10,0,{1.0 + k * 0.01:.2f}\n" for k in range(100)), encoding="utf-8")
        row = lambda clip, k=50: f"{clip},{k},4,0,12,0,1.05\n"
        # (A) 空 csv・完了記録なし
        a = tmp / "A.csv"; a.write_text("", encoding="utf-8")
        rc, err = run(tmp / "A.md", plan, meta, [f"A={a}"])
        check("A 空 csv（完了記録なし）は止まる", rc != 0 and "空の予測" in err, (rc, err))
        # (B) manifest に 1 本も合わない（run_meta 無し）
        b = tmp / "B.csv"; b.write_text(row("other_1") + row("other_2"), encoding="utf-8")
        rc, err = run(tmp / "B.md", plan, meta, [f"B={b}"])
        check("B 評価 manifest に 1 本も合わない csv は止まる", rc != 0 and "1 本も合わない" in err, (rc, err))
        # (C) 6 列の行が混ざる
        c = tmp / "C.csv"; c.write_text(row("val_a") + "val_b,50,4,0,12,0\n", encoding="utf-8")
        rc, err = run(tmp / "C.md", plan, meta, [f"C={c}"])
        check("C 6 列の行が混ざった csv は止まる", rc != 0 and "7 列でない" in err, (rc, err))
        check("C' 同じ csv でも --strict 無しなら通る（既定の挙動は変えない）", run(tmp / "C2.md", plan, meta, [f"C={c}"], strict=False)[0] == 0)
        # (D) 4,500 → 300 型: 予測は評価 manifest より広い（run_meta の推論 manifest に含まれる）
        d = tmp / "D.csv"; d.write_text(row("val_a") + row("val_b") + row("val_extra"), encoding="utf-8")
        d.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_a.flac", "val_b.flac", "val_extra.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "D.md", plan, meta, [f"D={d}"])
        check("D 評価 manifest ⊂ 推論 manifest（4,500→300 型）は通る", rc == 0, (rc, err))
        # (E) 完了記録つきの正当な無検出
        e = tmp / "E.csv"; e.write_text("", encoding="utf-8")
        e.with_suffix(".done.json").write_text(json.dumps({"n_rows": 0, "n_clips": 2, "n_files": 2}), encoding="utf-8")
        e.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_a.flac", "val_b.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "E.md", plan, meta, [f"E={e}"])
        check("E 完了記録つきの空 csv は無検出として通る", rc == 0, (rc, err))
        # (F) run_meta があるのに未知 ID
        f = tmp / "F.csv"; f.write_text(row("val_a") + row("ghost"), encoding="utf-8")
        f.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_a.flac", "val_b.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "F.md", plan, meta, [f"F={f}"])
        check("F 推論 manifest に無い ID の予測は止まる", rc != 0 and "推論 manifest に無い" in err, (rc, err))
        # (G) 評価 manifest の clip が推論 manifest に無い
        g = tmp / "G.csv"; g.write_text(row("val_a"), encoding="utf-8")
        g.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_a.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "G.md", plan, meta, [f"G={g}"])
        check("G 評価 manifest の clip が推論 manifest に無ければ止まる", rc != 0 and "推論 manifest に無い" in err, (rc, err))
        # (H)(I) 2026-09-22 改名（監査 9/22 R01）: 旧名の来歴（run_meta・予測）と新名の評価 manifest を同じ規約で比べる
        plan2 = tmp / "plan2.csv"
        plan2.write_text("clip_id,split\nval_syn_clip00001,val\nval_syn_clip00002,val\n", encoding="utf-8")
        meta2 = tmp / "gt2"; meta2.mkdir()
        for c in ("val_syn_clip00001", "val_syn_clip00002"):
            (meta2 / f"{c}.csv").write_text("".join(f"{k},4,0,10,0,{1.0 + k * 0.01:.2f}\n" for k in range(100)), encoding="utf-8")
        h = tmp / "H.csv"; h.write_text(row("fold2_room1_mix0001") + row("fold2_room1_mix0002"), encoding="utf-8")   # 旧名の予測
        h.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["fold2_room1_mix0001.flac", "fold2_room1_mix0002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "H.md", plan2, meta2, [f"H={h}"])
        check("H 旧名の予測＋旧名の推論 manifest を新名の評価 manifest で --strict 採点できる（改名後の再現経路）", rc == 0, (rc, err))
        i = tmp / "I.csv"; i.write_text(row("fold2_room1_mix0001") + row("val_syn_clip00002"), encoding="utf-8")
        i.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["fold2_room1_mix0001.flac", "val_syn_clip00001.flac", "val_syn_clip00002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "I.md", plan2, meta2, [f"I={i}"])
        check("I 推論 manifest に同じ本の旧名と新名が混在（衝突）すれば止まる", rc != 0 and "衝突" in err, (rc, err))
        j = tmp / "J.csv"; j.write_text(row("fold2_room1_mix0001") + row("fold2_room1_mix0009"), encoding="utf-8")   # 0009 は推論 manifest に無い
        j.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["fold2_room1_mix0001.flac", "fold2_room1_mix0002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "J.md", plan2, meta2, [f"J={j}"])
        check("J 旧名でも推論 manifest に無い ID の予測は止まる", rc != 0 and "推論 manifest に無い" in err, (rc, err))
        # (K)(L) 帰属の許容の環境変数の混入防止（監査 9/21 M05・9/22 R03）
        rc, err = run(tmp / "K.md", plan2, meta2, [f"H={h}"], env={"NOTIFY_ATTRIB_AZ_TOL": "20"})
        check("K 帰属の許容を環境変数で変えたのに NOTIFY_ATTRIB_SWEEP=1 が無ければ止まる（置き忘れの混入防止）", rc != 0 and "NOTIFY_ATTRIB_SWEEP" in err, (rc, err))
        rc, err = run(tmp / "L.md", plan2, meta2, [f"H={h}"], env={"NOTIFY_ATTRIB_AZ_TOL": "20", "NOTIFY_ATTRIB_SWEEP": "1"})
        head_l = (tmp / "L.md").read_text(encoding="utf-8") if (tmp / "L.md").exists() else ""
        check("L 掃引の明示（SWEEP=1）なら通り、採点 md の見出しに実効値（≤20°・掃引）が出る", rc == 0 and "≤20°" in head_l and "掃引の設定" in head_l, (rc, err, head_l[:200]))
        rc, err = run(tmp / "M.md", plan2, meta2, [f"H={h}"])
        head_m = (tmp / "M.md").read_text(encoding="utf-8") if (tmp / "M.md").exists() else ""
        check("M 既定（環境変数なし）の見出しは ±0.5 s・≤30° で掃引の印が無い", rc == 0 and "±0.5 s・≤30°" in head_m and "掃引の設定" not in head_m, (rc, err, head_m[:200]))
        # (N)(O)(P) 再監査 9/22 R01: 予測 csv・評価 manifest の中の旧名と新名の衝突も拒む。正当な複数フレーム・複数 track は通す
        n = tmp / "N.csv"; n.write_text(row("fold2_room1_mix0001") + row("val_syn_clip00001", k=60) + row("fold2_room1_mix0002"), encoding="utf-8")
        n.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "N.md", plan2, meta2, [f"N={n}"])
        check("N 予測 csv に同じ本の旧名と新名が混在すれば止まる", rc != 0 and "混在" in err, (rc, err))
        plan3 = tmp / "plan3.csv"
        plan3.write_text("clip_id,split\nval_syn_clip00001,val\nfold2_room1_mix0001,val\nval_syn_clip00002,val\n", encoding="utf-8")
        o = tmp / "O.csv"; o.write_text(row("val_syn_clip00001") + row("val_syn_clip00002"), encoding="utf-8")
        o.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "O.md", plan3, meta2, [f"O={o}"])
        check("O 評価 manifest に同じ本の旧名と新名が混在すれば止まる（同じ本を 2 本として採点しない）", rc != 0 and "重複" in err, (rc, err))
        q = tmp / "P.csv"; q.write_text(row("fold2_room1_mix0001", k=50) + row("fold2_room1_mix0001", k=51) + "fold2_room1_mix0001,52,4,1,12,0,1.05\n" + row("fold2_room1_mix0002"), encoding="utf-8")
        q.with_suffix(".run_meta.json").write_text(json.dumps({"clips": ["fold2_room1_mix0001.flac", "fold2_room1_mix0002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "P.md", plan2, meta2, [f"P={q}"])
        check("P 同じ本の複数フレーム・複数 track の正当な予測は通る", rc == 0, (rc, err))
        # (Q)(R)(S) Opus 5.5 監査 2026-09-23 F01: 知覚層・通知掃引の道具も評価 manifest を正規化し、--strict の中身の検査を統一採点器と同じにする
        def run_tool(script, out_md, plan_, meta_, items, extra=(), strict=True, env=None):
            cmd = [PY, str(ROOT / "scripts" / script), str(out_md)] + (["--strict"] if strict else []) + ["--plan", str(plan_), "--meta", str(meta_)] + list(extra) + items
            base = {k: v for k, v in __import__("os").environ.items() if not k.startswith("NOTIFY_")}
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env={**base, "PYTHONIOENCODING": "utf-8", **(env or {})})
            return r.returncode, (r.stderr or "")[-400:]
        plan_old = tmp / "plan_old.csv"      # 旧名だけの plan ＋ 旧名だけの正解（= 手元の v16 の組）
        plan_old.write_text("clip_id,split\nfold2_room1_mix0001,fold2\nfold2_room1_mix0002,fold2\n", encoding="utf-8")
        meta_old = tmp / "gt_old"; meta_old.mkdir()
        for c in ("fold2_room1_mix0001", "fold2_room1_mix0002"):
            (meta_old / f"{c}.csv").write_text("".join(f"{k},4,0,10,0,{1.0 + k * 0.01:.2f}\n" for k in range(100)), encoding="utf-8")
        rc, err = run_tool("_perception_eval.py", tmp / "Q.md", plan_old, meta_old, [f"Q={h}"])
        check("Q 知覚層: 旧名だけの plan と正解は黙って 0 にならず止まる（新名の正解が無い）", rc != 0 and "missing GT" in err, (rc, err))
        rc, err = run_tool("_perception_eval.py", tmp / "Q2.md", plan2, meta2, [f"Q2={h}"])
        check("Q' 知覚層: 旧名の予測＋旧名の推論 manifest を新名の plan・正解で --strict 採点できる", rc == 0, (rc, err))
        rc, err = run_tool("_perception_eval.py", tmp / "R.md", plan3, meta2, [f"R={o}"])
        check("R 知覚層: 評価 manifest に同じ本の旧名と新名が混在すれば止まる", rc != 0 and "重複" in err, (rc, err))
        rc, err = run_tool("_perception_eval.py", tmp / "R2.md", plan2, meta2, [f"R2={f}"])
        check("R' 知覚層: 推論 manifest に無い ID の予測は止まる（strict_check_pred を通る）", rc != 0 and "推論 manifest に無い" in err, (rc, err))
        rc, err = run_tool("_notify_threshold_sweep.py", tmp / "S.md", plan3, meta2, [f"S={o}"], extra=["--scales", "1.0"])
        check("S 通知掃引: 評価 manifest に同じ本の旧名と新名が混在すれば止まる", rc != 0 and "重複" in err, (rc, err))
        rc, err = run_tool("_notify_threshold_sweep.py", tmp / "S2.md", plan2, meta2, [f"S2={h}"], extra=["--scales", "0.8,1.0,1.2"])
        body_s = (tmp / "S2.md").read_text(encoding="utf-8") if (tmp / "S2.md").exists() else ""
        check("S' 通知掃引: 旧名の予測を新名の plan で --strict 採点でき、撤回した「控えめ／厳格」「宣言値」の札が出ない（F02）",
              rc == 0 and "宣言値" not in body_s and "厳格" not in body_s and "控えめ" not in body_s and "×1.2" in body_s, (rc, err, body_s[:300]))
        rc, err = run_tool("_notify_threshold_sweep.py", tmp / "S3.md", plan_old, meta_old, [f"S3={h}"], extra=["--scales", "1.0"])
        check("S'' 通知掃引: 旧名だけの plan と正解は黙って 0 にならず止まる", rc != 0 and "missing GT" in err, (rc, err))
        # (T) Opus 5.5 監査 F03: 撤回した --mode 厳格 は NOTIFY_MODE_SWEEP=1 が無ければ拒み、あれば見出しに写す
        rc, err = run(tmp / "T.md", plan2, meta2, [f"T={h}"] , env={})
        rc_t, err_t = run_tool("_score_unified.py", tmp / "T1.md", plan2, meta2, [f"T={h}"], extra=["--mode", "厳格"])
        check("T --mode 厳格 は NOTIFY_MODE_SWEEP=1 が無ければ止まる（撤回した設定の混入防止）", rc_t != 0 and "NOTIFY_MODE_SWEEP" in err_t, (rc_t, err_t))
        rc_t2, err_t2 = run_tool("_score_unified.py", tmp / "T2.md", plan2, meta2, [f"T={h}"], extra=["--mode", "厳格"], env={"NOTIFY_MODE_SWEEP": "1"})
        head_t2 = (tmp / "T2.md").read_text(encoding="utf-8") if (tmp / "T2.md").exists() else ""
        check("T' 掃引の明示（NOTIFY_MODE_SWEEP=1）なら通り、見出しに --mode 厳格 と撤回の印が出る", rc_t2 == 0 and "--mode 厳格" in head_t2 and "#37" in head_t2, (rc_t2, err_t2, head_t2[:300]))
        head_t = (tmp / "T.md").read_text(encoding="utf-8") if (tmp / "T.md").exists() else ""
        check("T'' --mode なし（本番）の見出しに mode の印は無い", rc == 0 and "--mode" not in head_t, (rc, err, head_t[:200]))
        # (U) Opus 5.5 監査 F15: --strict なしで旧名と新名が混在しても帯・層別の表で 2 回数えない（本数が 2 本になる）
        rc, err = run(tmp / "U.md", plan3, meta2, [f"U={o}"], strict=False)
        head_u = (tmp / "U.md").read_text(encoding="utf-8") if (tmp / "U.md").exists() else ""
        check("U --strict なしの混在は 1 本として数える（見出し = 2 本）", rc == 0 and "= 2 本" in head_u, (rc, err, head_u[:200]))
        # (V) Opus 5.5 監査 F14: プロセス内で帰属の許容を書き換えても settings() の default が追従する
        code = ("import importlib.util,sys;from pathlib import Path;p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('att',p);m=importlib.util.module_from_spec(s);"
                "sys.modules['att']=m;s.loader.exec_module(m);a=m.settings()['default'];m.AZ_TOL=20.0;b=m.settings()['default'];print(a,b)")
        r = subprocess.run([PY, "-c", code, str(ROOT / "scripts/_notify_v43_attrib.py")], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={k: v for k, v in __import__("os").environ.items() if not k.startswith("NOTIFY_")})
        check("V settings() の default は書き換えに追従する（True → False）", r.returncode == 0 and r.stdout.strip() == "True False", (r.returncode, r.stderr[-300:]))
        # (W) Opus 5.5 監査 B06（2026-09-24）: サーバの置き方（<run>/run_meta.json・<stem>.run_meta.json は無い）でも推論 manifest の検査が働く
        wdir = tmp / "infer_w"; wdir.mkdir()
        w = wdir / "val_all_causal.csv"; w.write_text(row("val_syn_clip00001") + row("ghost"), encoding="utf-8")
        (wdir / "run_meta.json").write_text(json.dumps({"clips": ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "W.md", plan2, meta2, [f"W={w}"])
        check("W 同じフォルダの run_meta.json（サーバの置き方）でも未知 ID の予測は止まる", rc != 0 and "推論 manifest に無い" in err, (rc, err))
        w2dir = tmp / "infer_w2"; w2dir.mkdir()
        w2 = w2dir / "val_all_causal.csv"; w2.write_text(row("val_syn_clip00001") + row("val_syn_clip00002"), encoding="utf-8")
        (w2dir / "run_meta.json").write_text(json.dumps({"clips": ["val_syn_clip00001.flac", "val_syn_clip00002.flac"]}), encoding="utf-8")
        rc, err = run(tmp / "W2.md", plan2, meta2, [f"W2={w2}"])
        check("W' 同じフォルダの run_meta.json で評価 ⊆ 推論なら通り、省略の注記が出ない", rc == 0 and "推論 manifest の検査は省略" not in err, (rc, err))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in results)
    print(f"{n_ok}/{len(results)} {'ALL PASS' if n_ok == len(results) else 'FAILED'}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
