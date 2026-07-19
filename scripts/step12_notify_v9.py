# -*- coding: utf-8 -*-
"""Step 12: 通知層 v1 — ルール発火＋リードタイム採点＋オラクル上限（v9 run1用）。

設計: out/v9_design_v2_2026-07-16.md 6節 / out/notification_design_research_2026-07-16.md
2層構造の下段。入力は知覚層の予測（infer CSV: clip,frame,class,az,el）と
混合音声のレベル軌跡・scene.json（採点の真値）のみ。学習には一切触れない。

ルール v1（事前決定済みの定数。調整するならtrain/valのみ、testは最終1回）:
  - 警告音5クラス: 同一クラスが 0.3秒（3フレーム）連続で検出されたら発火。
    1イベント1回、同一クラスの再通知は不応期 5秒。
  - 車: 1.0秒（10フレーム中9以上）持続検出 ＋ 方位変化小（窓内ドリフト≤15°、
    CBDR原則=「方位が変わらず近づく物は衝突コース」）＋ 音量上昇（混合W chの
    A特性フレームレベルの回帰傾き>0）が揃ったら「接近車」発火。
  - 通知対象: 警告音=全イベント / 車=危険層（マイク相対CPA≤3m）のみ。
    CPA>3mの車への発火は「過剰通知」として別計上（設計①の非対称マトリクス）。

採点:
  - 警告音: イベント通知率・発火遅れ（発音開始→発火）・通知方向誤差
  - 車(危険層): リードタイム = cpa_rel_time − 発火時刻。合格線 2.5s(理想)/2.0s(最低)。
    **オラクル上限** = cpa_rel_time − 可聴開始時刻（ルール遅れゼロの物理上限）を必ず並記
  - 誤通知: 該当イベントのない発火（回/時）＋safe層の車への過剰通知
  - scenario(交差点サイレン20本): 通過時刻 t_pass に対するリードタイム

使い方:
  python scripts/step12_notify_v9.py            # val240 + scenario20 を採点
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from outdoor_seld.calibration import frame_spl_a  # noqa: E402

def _argval(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


V91 = "--v91" in sys.argv
_SUF = "_1" if V91 else ""
DS = ROOT / "out" / f"dataset_outdoor_siren_v9{_SUF}"
PRED = ROOT / "out" / f"predictions_v9{_SUF}"
OUT = ROOT / "out" / f"step12_notify_v9{_SUF}"
# fold3（最終テスト）採点用（監査 o3-r6 指摘1）: --split fold3 --pred <out相対CSV> [--out <名>]
SPLIT = _argval("--split")           # None=既定(val+scenario) / "fold3"
PRED_OVERRIDE = _argval("--pred")    # out/ からの相対CSVパス
if _argval("--out"):
    OUT = ROOT / "out" / _argval("--out")

CLASSES = ["Siren", "Horn", "BackupBeep", "BikeBell", "CarDrive", "Crossing"]
CAR = 4
FPS = 10  # 0.1s/frame


def emit_time(k):
    """出力フレーム k の検出が因果的に利用可能になる時刻[s]。

    フレーム k は音声区間 [k/FPS, (k+1)/FPS) を表すため、未来を見ずに通知を出せる
    最速時刻は区間終端の (k+1)/FPS。従来は k/FPS を使いリード/遅延を一律0.1s
    楽観方向にずらしていた（監査 o3-r6 指摘3、2026-07-19 修正）。可聴開始オラクルにも
    同一規約を適用するため、モデル-オラクル差は不変で全リードが一律 -0.1s になる。
    方向誤差・距離の GT サンプル時刻は監査対象外のため従来どおり（変更しない）。
    """
    return (k + 1) / FPS


def poisson_upper95(count, exposure_hours):
    """観測 count 件/exposure_hours 時間 のPoisson率[件/時]の片側95%上限（正確法）。

    count=0 のとき rule-of-3（≈3/T）に一致。誤通知フロアを「ゼロ回/時」と断定せず
    「観測ゼロ→95%上限 X 回/時まで」と述べるために使う（監査 o3-r6 指摘6）。
    """
    from scipy.stats import chi2
    if exposure_hours <= 0:
        return float("inf")
    return float(chi2.ppf(0.95, 2 * count + 2) / 2.0 / exposure_hours)


def wilson_ci(k, n, z=1.96):
    """二項比率 k/n の95% Wilson score 信頼区間 (lo, hi)。n=0 は (0,1)。

    単一シード・n=20 中心の率（例 20/20, 0/20）に区間を添えるため（監査 o3-r6）。
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fmt_rate(k, n):
    """'k/n (P% [95%CI lo–hi])' 形式の文字列。Wilson区間。"""
    if not n:
        return f"{k}/{n}"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}% [95%CI {100 * lo:.0f}–{100 * hi:.0f}])"

# ルールv1定数（事前決定）
WARN_CONFIRM = 3          # 0.3s 連続
CAR_WIN = 10              # 1.0s 窓
CAR_MIN_HITS = 9          # 窓内 9/10 フレーム以上
CAR_AZ_DRIFT_MAX = 15.0   # [deg] CBDR: 方位変化小
REFRACTORY = 50           # 5s
DIR_REFRACT_DEG = 45.0    # v2(2026-07-18): 不応期は「クラス×方向」単位。前回発火と方位が
                          # これ以上離れていれば別イベントとして再発火を許す（監査H5-4:
                          # 同一クラス2イベント=siren×2等の2本目が構造的に通知不能だった修正。
                          # 単一音源クリップでは5s内の方位移動<45°のため従来結果と同一）
LEAD_PASS, LEAD_MIN = 2.5, 2.0


def load_pred(path):
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        p = line.strip().split(",")
        if len(p) >= 5:
            out[p[0]][int(p[1])].append((int(p[2]), float(p[3]), float(p[4])))
    return out


def circ_drift(az_list):
    """方位列の円周ドリフト幅[deg]（最大-最小を循環で）。"""
    a = np.radians(np.asarray(az_list))
    ref = np.angle(np.mean(np.exp(1j * a)))
    d = np.degrees(np.angle(np.exp(1j * (a - ref))))
    return float(d.max() - d.min())


def fire_events(pred_clip, mix_level):
    """1クリップの予測から通知発火列 [(frame, class, az)] を返す（ルールv1）。"""
    fires = []
    last_fires = defaultdict(list)   # class -> [(frame, az)] 方向別不応期用
    # フレームごとのクラス→(az,el)辞書に整形
    byframe = {k: {c: (a, e) for c, a, e in evs} for k, evs in pred_clip.items()}

    def blocked(c, k, az):
        return any(k - kp < REFRACTORY
                   and abs((az - ap + 180) % 360 - 180) <= DIR_REFRACT_DEG
                   for kp, ap in last_fires[c])

    for k in range(100):
        for c in range(6):
            if c == CAR:
                ks = list(range(k - CAR_WIN + 1, k + 1))
                if ks[0] < 0:
                    continue
                hits = [kk for kk in ks if c in byframe.get(kk, {})]
                if len(hits) < CAR_MIN_HITS:
                    continue
                azs = [byframe[kk][c][0] for kk in hits]
                if circ_drift(azs) > CAR_AZ_DRIFT_MAX:
                    continue
                lv = mix_level[ks[0]:k + 1]
                slope = np.polyfit(np.arange(len(lv)), lv, 1)[0]
                if slope <= 0.0:
                    continue
                if blocked(c, k, azs[-1]):
                    continue
                fires.append((k, c, azs[-1]))
                last_fires[c].append((k, azs[-1]))
            else:
                ks = list(range(k - WARN_CONFIRM + 1, k + 1))
                if ks[0] < 0:
                    continue
                if all(c in byframe.get(kk, {}) for kk in ks):
                    az = byframe[k][c][0]
                    if blocked(c, k, az):
                        continue
                    fires.append((k, c, az))
                    last_fires[c].append((k, az))
    return fires


def gt_az_at(scene, cls_name, t):
    """scene.jsonの音源真値から時刻tの方位[deg]を出す（通知方向誤差の採点用）。"""
    from outdoor_seld.geometry import apparent_azel_deg, sound_speed
    for i, src in enumerate(scene["sources"]):
        if src["class"] == cls_name:
            mic = (np.array([0.0, 0.0, 1.5]) if scene["mic"]["motion"] == "static"
                   else _mic_wp(scene))
            az, _, _, _ = apparent_azel_deg(np.array([t]), np.array(src["wp"], float),
                                            mic, sound_speed(20.0))
            return float(az[0])
    return None


def _mic_wp(scene):
    m = scene["mic"]
    if "waypoints" in m:                     # scenario2以降はscene.jsonに軌道を保存済み
        return np.array(m["waypoints"], dtype=np.float64)
    if m["motion"] == "walk":
        v, d = m["walk_speed_mps"], m["walk_dir_x"]
        x0 = -d * v * 5.0
        return np.array([[0.0, x0, 0.0, 1.5], [10.0, x0 + d * v * 10.0, 0.0, 1.5]])
    v, ts = m["walk_speed_mps"], m["t_stop_s"]
    y0 = -0.5 - v * ts
    return np.array([[0.0, 0.0, y0, 1.5], [ts, 0.0, -0.5, 1.5], [10.0, 0.0, -0.5, 1.5]])


CLSNAME2IDX = {"siren": 0, "horn": 1, "backup_beep": 2, "bike_bell": 3,
               "car_drive": 4, "crossing": 5}


def score_clip(clip, pred_clip, rows_out):
    scene = json.loads((DS / "work" / clip / "scene.json").read_text())
    mix = np.asarray(sf.read(DS / "foa" / f"{clip}.flac")[0], np.float64).T
    mix_level = frame_spl_a(mix[0], 24000)
    fires = fire_events(pred_clip, mix_level)
    res = {"clip": clip, "tier": scene["row"]["danger_tier"],
           "scenario": scene["row"]["scenario"], "motion": scene["row"]["motion"]}

    fired_by_cls = defaultdict(list)
    for k, c, az in fires:
        fired_by_cls[c].append((k, az))

    # --- 警告音イベントの採点 ---
    warn_results = []
    for src in scene["sources"]:
        ci = CLSNAME2IDX[src["class"]]
        if ci == CAR:
            continue
        t_on, t_off = src["t_on"], src["t_off"]
        cand = [(k, az) for k, az in fired_by_cls[ci]
                if t_on * FPS - 1 <= k <= (t_off + 1.0) * FPS]
        if cand:
            k, az = cand[0]
            latency = emit_time(k) - t_on
            gaz = gt_az_at(scene, src["class"], k / FPS)
            daz = abs((az - gaz + 180) % 360 - 180) if gaz is not None else np.nan
            warn_results.append((src["class"], True, latency, daz))
        else:
            warn_results.append((src["class"], False, np.nan, np.nan))

    # --- 車の採点 ---
    car_src = next((s for s in scene["sources"] if s["class"] == "car_drive"), None)
    car = {}
    if car_src is not None:
        t_cpa = scene.get("cpa_rel_time_s")
        fires_car = fired_by_cls[CAR]
        t_fire = emit_time(fires_car[0][0]) if fires_car else None
        t_aud = car_src.get("t_first_audible_s")
        car = {"tier": scene["row"]["danger_tier"], "t_cpa": t_cpa,
               "t_fire": t_fire,
               "lead": (t_cpa - t_fire) if t_fire is not None else None,
               "oracle_lead": (t_cpa - t_aud) if t_aud is not None else None,
               "n_fires": len(fires_car)}

    # --- 誤通知（イベントのないクラスの発火）---
    present = {CLSNAME2IDX[s["class"]] for s in scene["sources"]}
    false_fires = sum(len(v) for c, v in fired_by_cls.items() if c not in present)

    res.update(n_fires=len(fires), false_fires=false_fires)
    rows_out.append((res, warn_results, car))
    return res, warn_results, car


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# step12 通知層v1 採点（v9 run1、2026-07-17）", ""]
    percl = []

    # 評価名簿は「予測に現れたクリップ」ではなく plan の割当から取る（監査 o3-r6）。
    # 完全検出失敗（予測ゼロ）のクリップも分母に残すため。val=fold2 / scenario=割当表。
    core = list(csv.DictReader(open(DS / "plan" / "assignment_core.csv")))
    scen = list(csv.DictReader(open(DS / "plan" / "assignment_scenario.csv")))
    roster = {"val": [r["clip_id"] for r in core if r["split"] == "fold2"],
              "scenario": [r["clip_id"] for r in scen],
              "fold3": [r["clip_id"] for r in core if r["split"] == "fold3"]}

    # SPLIT指定時はその1集合のみ（fold3=最終テスト、監査指摘1）。予測は --pred で差し替え可
    tags = (SPLIT,) if SPLIT else ("val", "scenario")
    for tag in tags:
        pred_path = (ROOT / "out" / PRED_OVERRIDE) if PRED_OVERRIDE \
            else PRED / f"{tag}_all.csv"
        pred = load_pred(pred_path)
        rows = []
        clips = roster[tag]
        n_missing = sum(1 for c in clips if c not in pred)
        for clip in clips:
            score_clip(clip, pred.get(clip, {}), rows)

        # ---- 集計 ----
        warn_all = [w for _, ws, _ in rows for w in ws]
        n_warn = len(warn_all)
        notified = [w for w in warn_all if w[1]]
        lat = np.array([w[2] for w in notified])
        daz = np.array([w[3] for w in notified if np.isfinite(w[3])])
        report.append(f"## {tag}（{len(rows)}クリップ）")
        if n_missing:
            report.append(f"- ⚠️ 予測ゼロ（完全検出失敗）で分母に残したクリップ: {n_missing}本")
        if n_warn:
            _wlo, _whi = wilson_ci(len(notified), n_warn)
            report.append(f"- 警告音イベント {n_warn}件: 通知 {len(notified)}件 "
                          f"({len(notified)/n_warn:.1%} [95%CI {_wlo:.1%}–{_whi:.1%}])、発火遅れ中央値 "
                          f"{np.median(lat):.2f}s / p90 {np.percentile(lat, 90):.2f}s、"
                          f"通知方向誤差 中央値 {np.median(daz):.1f}°")
            byc = defaultdict(lambda: [0, 0, []])
            for cname, ok, l, _ in warn_all:
                byc[cname][1] += 1
                if ok:
                    byc[cname][0] += 1
                    byc[cname][2].append(l)
            for cname in sorted(byc):
                g = byc[cname]
                report.append(f"    {cname}: {g[0]}/{g[1]} "
                              f"(遅れ中央値 {np.median(g[2]):.2f}s)" if g[2] else
                              f"    {cname}: {g[0]}/{g[1]}")
        cars = [c for _, _, c in rows if c]
        danger = [c for c in cars if c["tier"] in ("critical", "caution")]
        safe = [c for c in cars if c["tier"] == "safe"]
        if danger:
            fired = [c for c in danger if c["lead"] is not None]
            leads = np.array([c["lead"] for c in fired])
            oleads = np.array([c["oracle_lead"] for c in danger
                               if c["oracle_lead"] is not None])
            _clo, _chi = wilson_ci(len(fired), len(danger))
            report.append(
                f"- 危険層の車 {len(danger)}台: 通知 {len(fired)}台 "
                f"({len(fired)/len(danger):.1%} [95%CI {_clo:.1%}–{_chi:.1%}])")
            if len(leads):
                report.append(
                    f"    リードタイム: 中央値 {np.median(leads):.2f}s / "
                    f">= {LEAD_PASS}s: {(leads >= LEAD_PASS).mean():.1%} / "
                    f">= {LEAD_MIN}s: {(leads >= LEAD_MIN).mean():.1%}")
            report.append(
                f"    オラクル上限: 中央値 {np.median(oleads):.2f}s / "
                f">= {LEAD_PASS}s: {(oleads >= LEAD_PASS).mean():.1%} / "
                f">= {LEAD_MIN}s: {(oleads >= LEAD_MIN).mean():.1%} "
                f"（物理とデータの上限。モデルの遅れ=この差）")
        if safe:
            over = sum(1 for c in safe if c["n_fires"] > 0)
            report.append(f"- safe層(CPA>3m)の車 {len(safe)}台: 過剰通知 {over}台 "
                          f"({over/len(safe):.1%})（設計: 通知しないのが正解）")
        ff = sum(r[0]["false_fires"] for r in rows)
        hours = len(rows) * 10.0 / 3600.0
        report.append(f"- 該当イベントなしの誤通知: {ff}件/{hours:.2f}h = {ff/hours:.1f}回/時 "
                      f"(Poisson 95%上限 {poisson_upper95(ff, hours):.1f}回/時)")
        # scenario専用: t_pass に対するリードタイム
        if tag == "scenario":
            leads_s = []
            for res, ws, _ in rows:
                scene = json.loads((DS / "work" / res["clip"] / "scene.json").read_text())
                t_pass = next(s["t_pass_s"] for s in scene["sources"]
                              if s["class"] == "siren")
                w = ws[0]
                if w[1]:
                    leads_s.append(t_pass - (w[2] + 0.0))  # w[2]=latency, t_on=0
            leads_s = np.array(leads_s)
            report.append(
                f"- 交差点サイレン: 通過の {np.median(leads_s):.1f}s 前に通知（中央値）、"
                f"最小 {leads_s.min():.1f}s / 全{len(leads_s)}本が2.5s以上前: "
                f"{(leads_s >= 2.5).mean():.1%}")
        report.append("")
        for res, ws, car in rows:
            percl.append({"set": tag, **res,
                          "car_lead": car.get("lead") if car else "",
                          "car_oracle": car.get("oracle_lead") if car else ""})

    with open(OUT / "per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(percl[0].keys()))
        w.writeheader()
        w.writerows(percl)
    (OUT / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("->", OUT)


if __name__ == "__main__":
    main()
