# -*- coding: utf-8 -*-
"""step10_v2final_plan.py — 確定評価 v2 の最終学習データの割当表（2026-09-07・2026-09-14 に新場面 D15/D17/D18 を組み込み）。

土台: v13 → v14 → v15 → v16 の plan 生成器をそのまま鎖にして、つまみだけ md/design/v2final_params.json から差し替える
（D3 雨の比率・D2 雨のレベル・D4 歩行比率・D10 至近低速の割合・D13 高さの幅・高さ増強）。
seed の塩（SALT）は v13/v16 と同じなので、つまみが v16 と同じ値なら土台は **assignment_v16.csv と 1 バイト違わず一致**する
（--check-v16 で確認。これが「鎖を壊していない」証拠）。D1 SIREN_MIX は描画側（_run_v2final_gen.py）で効く。

新場面（設計凍結 2026-09-14 §2・本人決定「追加」）: params の D15_FAST / D17_TURN / D18_PLATOON（adopt=true）を step10_v19_plan.build で
作り（clip_id・seed・motion・mic_z は v19 と同じ規則 = v19/v20a の試作と同じ場面）、**雨だけ v13 の規則（params の D3/D2）を掛ける**
（同じ場面の 2 高さは同じ雨・seed×SALT×19 で決定論的）。土台 18,000 ＋ 新場面 4,500 = N_ROWS_TOTAL_EXPECTED（22,500）。
2026-09-21 に D20 発進（+1,500 = 24,000）、2026-09-28 に D21 キックの遠い横距離（+1,500 = 25,500・変更票 #42・描画は step11_v21_render.py）。

出力: out/dataset_outdoor_siren_v2final/plan/
  assignment_v2final.csv       全行（土台 ＋ 新場面）— 生成・距離ラベル・束はこれを読む
  assignment_v2final_base.csv  土台だけ（--check-v16 の比較対象）
  assignment_v2final_new.csv   新場面だけ（--check-plan の対象）
  README_plan_v2final.md       params の SHA256・件数・検査結果
  中間の v13/v14/v15/v16 形式は plan/chain/ に残す。v13〜v16 の既存出力には書かない。

使い方: python scripts/step10_v2final_plan.py [--check-v16] [--check] [--params <json>]
  --check = 構造の検査（総行数・場面ごとの本数と fold・同 seed の 2 高さが同 fold・clip_id の一意性・雨と歩行の比率）＋ turn の軌道検査
  （step11_v19_render.check_plan）。1 つでも落ちれば終了コード 1。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from outdoor_seld.clip_ids import SPLIT_TO_OLD, number, split_to_new, to_new, to_old  # noqa: E402  （2026-09-22 改名。旧名の基準ファイルとは to_old で戻してから比べる）

PARAMS_DEFAULT = ROOT / "md/design/v2final_params.json"
OUT = ROOT / "out/dataset_outdoor_siren_v2final/plan"
CHAIN = OUT / "chain"
V16_PLAN = ROOT / "out/dataset_outdoor_siren_v16/plan/assignment_v16.csv"
SCENE_KEYS = {"D15_FAST": "fast", "D16_STATIC": "static", "D17_TURN": "turn", "D18_PLATOON": "platoon",
              "D20_LAUNCH": "launch",     # D20 = 発進（2026-09-21・変更票 #36 の (b)・既定 adopt=false）
              "D21_KICK_FAR": "kickfar"}  # D21 = キックの遠い横距離（2026-09-28・変更票 #42・描画は step11_v21_render.py）
D19_KEY = "D19_SLOW_TIERS"                # D19 = 徐行を注意帯・安全帯にも（#36 の (a)・既定 adopt=false・step10_v14_plan の SLOW_TIERS/SLOW_FRAC）
RAIN_SALT_EXTRA = 19     # 新場面の雨の乱数: rng([seed, SALT_v13, 19])。motion/mic_z は v19 の規則のまま


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list, columns: list) -> None:
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def sha256_as_old(p: Path) -> str:
    """plan を旧名（fold/room/mix・split=fold1/fold2）に戻して書いたときの sha256。2026-09-22 の改名前に固定した基準 sha
    （assignment_v16.csv・9/14 設計の 2b1a4302…）と比べるため。名前以外が 1 バイトでも違えば一致しない。"""
    rows = read_rows(p)
    old = []
    for r in rows:
        o = dict(r)
        o["clip_id"] = to_old(r["clip_id"])
        o["split"] = SPLIT_TO_OLD.get(r["split"], r["split"])
        old.append(o)
    tmp = p.with_suffix(".as_old.tmp.csv")
    write_rows(tmp, old, list(rows[0].keys()))
    h = sha256(tmp)
    tmp.unlink(missing_ok=True)
    return h


def build_base(P, p13, p14, p15, p16) -> Path:
    """v13 → v16 の鎖（つまみは params）。戻り = chain/assignment_v16.csv"""
    CHAIN.mkdir(parents=True, exist_ok=True)
    p13.WALK_FRAC = float(P["D4_WALK_FRAC"])
    p13.RAIN_FRAC = float(P["D3_RAIN_FRAC"])
    p13.RAIN_KIND_P = {k: float(v) for k, v in P["RAIN_KIND_P"].items()}
    p13.RAIN_DBA = {k: (float(v[0]), float(v[1])) for k, v in P["D2_RAIN_DBA"].items()}
    p13.SALT = int(P["SALTS"]["v13"])
    p13.OUT_DIR = CHAIN
    p13.main()
    p14.SRC_V13 = CHAIN / "assignment_v13.csv"
    p14.OUT_DIR = CHAIN
    p14.CLOSE_SLOW_FRAC = float(P["D10_CLOSE_SLOW_FRAC"])
    d19 = P.get(D19_KEY) or {}
    if d19.get("adopt"):
        tiers = tuple(str(t) for t in d19["tiers"])
        assert set(tiers) <= {"caution", "safe"}, f"{D19_KEY}.tiers は caution/safe だけ（critical は D10）: {tiers}"
        p14.SLOW_TIERS = ("critical",) + tiers
        p14.SLOW_FRAC = {t: float(d19["frac"][t]) for t in tiers}
    else:
        p14.SLOW_TIERS = ("critical",)                       # 既定 OFF = v16 と同一出力
    p14.main()
    p15.SRC_V14 = CHAIN / "assignment_v14.csv"
    p15.OUT_DIR = CHAIN
    p15.MIC_Z_RANGE = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    p15.main()
    assert int(P["HEIGHT_AUG_COPIES"]) == 2, "高さ増強は ×2 だけ実装（v16 と同じ）"
    p16.SRC_V15 = CHAIN / "assignment_v15.csv"
    p16.OUT_DIR = CHAIN
    p16.MIC_Z_RANGE = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    p16.SALT16 = int(P["SALTS"]["v16"])
    p16.main()
    return CHAIN / "assignment_v16.csv"


def build_new_scenes(P, p13) -> tuple[list, dict]:
    """params で adopt=true の新場面を v19 の規則で作り、雨だけ v13 の規則（params 由来）を掛ける。戻り = (rows, 場面ごとの要約)"""
    import step10_v19_plan as p19
    p19.WALK_FRAC = float(P["D4_WALK_FRAC"])
    p19.MIC_Z_RANGE = tuple(float(x) for x in P["D13_MIC_Z_RANGE"])
    kinds, ps = zip(*p13.RAIN_KIND_P.items())
    rows, summary = [], {}
    for key, scene in SCENE_KEYS.items():
        d = P.get(key) or {}
        if not d.get("adopt"):
            continue
        n = int(d["n_scenes"])
        scene_rows = p19.build(scene, n)
        assert len(scene_rows) == int(d["n_clips"]) == n * int(P["HEIGHT_AUG_COPIES"]), (scene, len(scene_rows), d)
        for r in scene_rows:
            rng = np.random.default_rng([int(r["seed"]), p13.SALT, RAIN_SALT_EXTRA])
            if rng.random() < p13.RAIN_FRAC:
                kind = str(rng.choice(kinds, p=ps))
                r["rain"], r["rain_rate"], r["rain_dba"] = kind, f"{p13.RAIN_RATE[kind]:.0f}", f"{rng.uniform(*p13.RAIN_DBA[kind]):.1f}"
            else:
                r["rain"], r["rain_rate"], r["rain_dba"] = "", "", ""
        c = Counter(r["split"] for r in scene_rows)
        summary[scene] = dict(n=len(scene_rows), train=c["train"], val=c["val"],
                              walk=sum(r["motion"] == "walk" for r in scene_rows) / len(scene_rows),
                              rain=sum(bool(r["rain"]) for r in scene_rows) / len(scene_rows),
                              mix=(scene_rows[0]["clip_id"][-5:], scene_rows[-1]["clip_id"][-5:]))
        rows += scene_rows
    return rows, summary


def check_structure(P, base: list, new: list, summary: dict) -> list:
    """構造の検査。戻り = 問題の一覧（空なら合格）"""
    bad = []
    n_total = int(P.get("N_ROWS_TOTAL_EXPECTED", P["N_ROWS_EXPECTED"]))
    if len(base) != int(P["N_ROWS_EXPECTED"]):
        bad.append(f"土台の行数 {len(base)} ≠ N_ROWS_EXPECTED {P['N_ROWS_EXPECTED']}")
    if len(base) + len(new) != n_total:
        bad.append(f"総行数 {len(base) + len(new)} ≠ N_ROWS_TOTAL_EXPECTED {n_total}")
    ids = [r["clip_id"] for r in base + new]
    if len(set(ids)) != len(ids):
        bad.append("clip_id が重複")
    copies = int(P["HEIGHT_AUG_COPIES"])
    for key, scene in SCENE_KEYS.items():
        d = P.get(key) or {}
        s = summary.get(scene)
        if not d.get("adopt"):
            if s:
                bad.append(f"{scene}: adopt=false なのに行がある")
            continue
        if s is None:
            bad.append(f"{scene}: adopt=true なのに行が無い"); continue
        if (s["n"], s["train"], s["val"]) != (int(d["n_clips"]), int(d["train"]), int(d["val"])):
            bad.append(f"{scene}: 本数 {s['n']} (train {s['train']} / val {s['val']}) ≠ params {d['n_clips']} ({d['train']} / {d['val']})")
        if abs(s["n"] / n_total - float(d["mix_frac"])) > 1e-3:
            bad.append(f"{scene}: mix_frac {s['n'] / n_total:.4f} ≠ params {d['mix_frac']}")
        # 同 seed の 2 高さは同 fold・同じ雨
        by_seed = defaultdict(list)
        for r in new:
            if r["scenario"] == f"v19{scene}":
                by_seed[r["seed"]].append(r)
        if any(len(v) != copies for v in by_seed.values()):
            bad.append(f"{scene}: seed ごとの高さの写しが {copies} 本でない")
        if any(len({r["split"] for r in v}) != 1 or len({r["rain"] for r in v}) != 1 for v in by_seed.values()):
            bad.append(f"{scene}: 同じ seed の 2 高さで fold か雨が違う")
        # 比率（二項分布の 3σ）
        n_sc = len(by_seed)
        for name, frac, p in (("歩行", s["walk"], float(P["D4_WALK_FRAC"])), ("雨", s["rain"], float(P["D3_RAIN_FRAC"]))):
            sd = math.sqrt(p * (1 - p) / max(n_sc, 1))
            if abs(frac - p) > 3 * sd + 1e-9:
                bad.append(f"{scene}: {name}の比率 {frac:.3f} が params {p} から 3σ（{3 * sd:.3f}）超えて離れている")
    # 土台の mix 番号と新場面の mix 番号が重ならない
    mix = number                                          # clip 番号（旧名でも新名でも）
    if base and new and max(mix(r["clip_id"]) for r in base) >= min(mix(r["clip_id"]) for r in new):
        bad.append("新場面の mix 番号が土台と重なる")
    return bad


def check_d19(P, base: list, p14) -> list:
    """D19（徐行を注意帯・安全帯にも）ON のときの土台の検査: v16 と比べて、対象行（v11core × caution/safe × 住宅街/日常 × L/R）の
    4 列（close_slow, cs_cpa_m, cs_speed_kmh, cs_level_adj_db）以外は 1 セルも変わらない・critical の行は完全一致・置換の割合が params と 3σ 内・
    最接近が層の範囲内。戻り = 問題の一覧（空なら合格）。"""
    bad = []
    d19 = P.get(D19_KEY) or {}
    if not d19.get("adopt"):
        return bad
    if not V16_PLAN.exists():
        return [f"{D19_KEY}: 比較対象 {V16_PLAN} が無い"]
    v16 = {}
    for r in read_rows(V16_PLAN):                                   # 基準は改名前の旧名 → 新名に直してから突き合わせる
        r = dict(r); r["clip_id"] = to_new(r["clip_id"]); r["split"] = split_to_new(r["split"]); v16[r["clip_id"]] = r
    cols4 = ("close_slow", "cs_cpa_m", "cs_speed_kmh", "cs_level_adj_db")
    n_el, n_cs, n_other_diff, n_crit_diff = Counter(), Counter(), 0, 0
    for r in base:
        o = v16.get(r["clip_id"])
        if o is None:
            bad.append(f"{D19_KEY}: v16 に無い clip_id {r['clip_id']}"); continue
        tier = r["danger_tier"]
        eligible = (r["scenario"] == "v11core" and tier in p14.SLOW_TIERS and tier != "critical"
                    and r["scene_type"] in p14.ELIGIBLE_SCENE and r["car_side"] in ("L", "R"))
        diff = [k for k in r if r[k] != o.get(k)]
        if eligible:
            if any(k not in cols4 for k in diff):
                n_other_diff += 1
            n_el[tier] += 1
            if r["close_slow"] == "1":
                n_cs[tier] += 1
                lo, hi = p14.SLOW_CPA[tier]
                if not (lo - 1e-9 <= float(r["cs_cpa_m"]) <= hi + 1e-9):
                    bad.append(f"{D19_KEY}: {r['clip_id']} の cs_cpa_m {r['cs_cpa_m']} が {tier} の範囲 {lo}〜{hi} の外")
                if not (p14.CS_SPEED_KMH[0] - 1e-9 <= float(r["cs_speed_kmh"]) <= p14.CS_SPEED_KMH[1] + 1e-9):
                    bad.append(f"{D19_KEY}: {r['clip_id']} の cs_speed_kmh {r['cs_speed_kmh']} が範囲外")
        elif diff:
            n_crit_diff += 1 if tier == "critical" else 0
            n_other_diff += 0 if tier == "critical" else 1
    if n_other_diff:
        bad.append(f"{D19_KEY}: 対象外の行、または対象行の 4 列以外が v16 と違う（{n_other_diff} 行）")
    if n_crit_diff:
        bad.append(f"{D19_KEY}: critical の行が v16 と違う（{n_crit_diff} 行）— critical は D10 のまま不変のはず")
    for tier in p14.SLOW_TIERS:
        if tier == "critical":
            continue
        p, n = float(p14.SLOW_FRAC[tier]), n_el[tier]
        sd = math.sqrt(p * (1 - p) / max(n, 1))
        frac = n_cs[tier] / max(n, 1)
        if abs(frac - p) > 3 * sd + 1e-9:
            bad.append(f"{D19_KEY}: {tier} の置換割合 {frac:.3f}（{n_cs[tier]}/{n}）が params {p} から 3σ（{3 * sd:.3f}）超えて離れている")
    return bad


def check_kickfar_rows(new: list, v21r) -> int:
    """D21（キックの遠い横距離）の全行を salt=0 でサンプルし、横距離 2.5〜8.0 m・最接近が t_cpa・速さ/音量/時刻の範囲を検査する。違反数を返す。"""
    bad = n = 0
    for r in new:
        if r["scenario"] != v21r.SCENARIO:
            continue
        rr = dict(r); rr["seed"] = int(rr["seed"]); rr["n_warnings"] = int(rr["n_warnings"] or 0)
        try:
            s = v21r.sample_scene_v19(rr)
            assert len(s["sources"]) == 1 and s["sources"][0]["class"] == "kickboard", "音源がキックボード 1 台でない"
            v21r.check_kickfar(s["sources"][0], s["_mic_arr"])
            n += 1
        except AssertionError as e:
            bad += 1
            print(f"NG {r['clip_id']}: {e}")
    print(f"check_kickfar {v21r.SCENARIO}: {n} 本 OK / {bad} 本 NG")
    return bad


def main() -> int:
    params_path = Path(_arg("--params", str(PARAMS_DEFAULT)))
    P = json.loads(params_path.read_text(encoding="utf-8"))
    import step10_v13_plan as p13
    import step10_v14_plan as p14
    import step10_v15_plan as p15
    import step10_v16_plan as p16

    base_src = build_base(P, p13, p14, p15, p16)
    OUT.mkdir(parents=True, exist_ok=True)
    base_dst = OUT / "assignment_v2final_base.csv"
    shutil.copyfile(base_src, base_dst)
    base = read_rows(base_dst)
    columns = list(base[0].keys())
    new, summary = build_new_scenes(P, p13)
    if new:
        assert set(new[0].keys()) == set(columns), f"新場面の列が土台と違う: {sorted(set(new[0].keys()) ^ set(columns))}"   # 並びは土台に合わせて書く
    new_dst = OUT / "assignment_v2final_new.csv"
    write_rows(new_dst, new, columns)
    dst = OUT / "assignment_v2final.csv"
    write_rows(dst, base + new, columns)
    n_total = int(P.get("N_ROWS_TOTAL_EXPECTED", P["N_ROWS_EXPECTED"]))
    folds = Counter(r["split"] for r in base + new)
    L = [f"# v2final plan（{len(base) + len(new):,} 行 = 土台 {len(base):,} ＋ 新場面 {len(new):,}）— {params_path.name} sha256 {sha256(params_path)[:16]}… / status={P.get('status')}", "",
         f"- つまみ: 歩行 {p13.WALK_FRAC:.0%} / 雨 {p13.RAIN_FRAC:.0%} dB(A) {p13.RAIN_DBA} / 至近低速 {p14.CLOSE_SLOW_FRAC:.0%} / 高さ {p15.MIC_Z_RANGE} m ×{P['HEIGHT_AUG_COPIES']} / SALT {P['SALTS']}",
         f"- {D19_KEY}（徐行を注意帯・安全帯にも）: {'ON ' + str(p14.SLOW_FRAC) + ' 塩 ' + str(p14.SALT14_SLOW) if len(p14.SLOW_TIERS) > 1 else 'OFF（critical だけ = v16 と同一）'}",
         f"- D1 SIREN_MIX {P['D1_SIREN_MIX']} は描画側（_run_v2final_gen.py）", f"- 内訳 split: {dict(folds)}（期待 総行数 {n_total:,}）"]
    for scene, s in summary.items():
        L.append(f"- 新場面 v19{scene}: {s['n']:,} 本（train {s['train']:,} / val {s['val']:,}）・歩行 {s['walk']:.1%}・雨 {s['rain']:.1%}・clip {s['mix'][0]}〜{s['mix'][1]}・雨の乱数 rng([seed, SALT_v13, {RAIN_SALT_EXTRA}])")
    L += [f"- assignment_v2final.csv sha256 {sha256(dst)}", f"- assignment_v2final_base.csv sha256 {sha256(base_dst)}", f"- assignment_v2final_new.csv sha256 {sha256(new_dst)}", ""]
    rc = 0
    if "--check-v16" in sys.argv:
        # 2026-09-22 改名後: 土台を旧名に戻して書いた sha256 が、改名前に固定した assignment_v16.csv の sha256 と一致すれば「内容不変」
        same = V16_PLAN.exists() and sha256(V16_PLAN) == sha256_as_old(base_dst)
        L.append(f"- 土台と v16 の一致（旧名に戻して比較。つまみが v16 と同じなら一致するはず）: {'一致' if same else '不一致'}")
        print("土台と v16 の一致:", "一致" if same else "不一致")
        if not same and "--allow-v16-diff" not in sys.argv:
            rc = 1      # 不一致は終了コードにも出す（つまみを v16 と変えた意図的な不一致は --allow-v16-diff で明示。Opus 5.5 監査 9/23 F11）
    if "--check" in sys.argv:
        bad = check_structure(P, base, new, summary)
        L.append(f"- 構造の検査: {'合格' if not bad else '不合格 ' + ' / '.join(bad)}")
        if (P.get(D19_KEY) or {}).get("adopt"):
            bad19 = check_d19(P, base, p14)
            L.append(f"- {D19_KEY} の検査（v16 との差が対象行の 4 列だけ・critical 不変・割合・範囲）: {'合格' if not bad19 else '不合格 ' + ' / '.join(bad19)}")
            bad += bad19
        if new:
            import step11_v19_render as v19r
            import step11_v21_render as v21r  # noqa: F401  v19kickfar の登録（変更票 #42・2026-09-28）。無いと kickfar 行が「未対応の scenario」で NG になる
            n_bad = v19r.check_plan(new_dst)
            if (P.get("D21_KICK_FAR") or {}).get("adopt"):
                n_bad += check_kickfar_rows(new, v21r)
            L.append(f"- 新場面の軌道検査（check_plan・turn の折れ線・launch の発進条件・kickfar の横距離と最接近）: {'違反 0' if n_bad == 0 else f'違反 {n_bad}'}")
            bad += [f"軌道検査の違反 {n_bad}"] if n_bad else []
        print("構造の検査:", "合格" if not bad else "不合格")
        for b in bad:
            print("  NG", b)
        rc = 1 if (bad or rc) else 0
    (OUT / "README_plan_v2final.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6 + len(summary)]))
    print("->", dst)
    return rc


if __name__ == "__main__":
    sys.exit(main())
