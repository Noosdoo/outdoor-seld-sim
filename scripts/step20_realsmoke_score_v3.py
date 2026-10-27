# -*- coding: utf-8 -*-
"""Step 20 v3: 実録採点 — 現行の通知規則 **v4.3＋警告音 hold** で採点し、方向は「注釈と同じ窓の推定方位」で比べる（2026-09-06）。

敵対的レビュー（`md/audit/敵対的レビュー_実録プロトコル_2026-09-05_Astra.md`）R02・R03・R10・W7 への対応。
変更票 #57（2026-10-10・実装 2026-10-12）: 報告に「来る方向（象限）の内訳」の補助の表を足した（quadrant_breakdown・主要評価は変えない・記述だけ）。旧 v2（v3.4 規則）は比較用に残す。

規則（合成データの統一採点器 `scripts/_score_unified.py` と同じ正規実装を呼ぶ・独自の規則は発明しない）:
  - 距離クラス（car/kick/bike）: `step12_notify_v43.run_rule3`（`out/notify_v43_sweep/winner.json` の採用構成）→ 発火 (frame, az, tier=強/中, d)
  - 警告音クラス: `step12_notify_v9c_multitrack.warn_fires_v9c`（名札方式・2026-09-28 変更票 #41 で hold から置換・既定）。旧 `step12_notify_v9b_hold.warn_fires(hold=True)` は `--warn-rule hold` で併記
  - 発火時刻は因果時刻 (k+1)/FPS。クリップ長は予測の最終フレームから（可変長）

3 つの時点・窓を分けて定義する（R03）:
  ① 型分類の窓（記述用のみ・CPA の 2.5〜1.5 秒前）: ここでは使わない（GT の連続方位が無い）
  ② **通知成功の窓** = [t_start − 1 s, t_cpa ＋ 1 s]（合成の採点器と同じ）。critical は強、caution は中か強、safe は強も中も無いこと
  ③ **方向の比較時点** = 注釈の象限を書いた窓（距離クラス: CPA の 2.5〜1.5 秒前 = 記入用 CSV の定義。警告音クラス: 通知が出た瞬間の推定方位を「鳴り始めの向き」と比べる・再監査2 Q06）。
     その窓の**推定方位の中央（単位ベクトル平均）**を 4 象限に丸めて注釈と比べる。発火時刻の方位ではない。窓に推定が無ければ「方向は評価不能（未検出）」

分母（W7）: 到達・抑制は「採点対象イベント」を分母に。方向は「全イベント（未検出は不一致扱い）」と「推定があった例だけ」の両方を出す。
警告音（再監査 N02）: 時間指標は **遅れ = 発火 − 鳴り始め（t_start）**。距離クラスのリード（CPA − 発火）とは別に集計する。方向は鳴り始め直後の窓
  [t_start, t_start + --warn-dir-span（既定 1.0 s）] の推定方位（3 秒全体を平均しない）。「鳴り始め＝気づいた瞬間」は観測上の限界として明記する。
歩行対比（再監査 N05）: pair_id のある行（区分=歩行）は ablation の主要評価に入れない（静止側も）。別集計で **検出フレーム率**（イベント窓内で
  そのクラスが出ているフレームの割合＝歩行対比の主指標）と到達を出し、pair_id ごとの静止−歩行の差を並べる。
前方 F（R02）: 変更票 #40（2026-09-28）で**主要評価に含める（既定）**。旧の除外は `--exclude-front`（前方は「前方（参考）」に別集計）。`--include-front` は旧の明示で、受け付ける。
履歴不足（R10）: クリップ内の t_cpa が `--min-history` 秒（既定 7.5）未満のイベントは `history_short` として主要評価から外し件数を出す。
v30 の暖機（本体 §2-6・§9-1 の 1・§9-4 の 1・2026-10-26・`--warmup-v30`・既定は旧のまま＝発効のあとで既定に）: 合成の統一採点器 `--events` と同じ決まり。
  - 通知の規則（v4.3・v9c）への入力は frame 40（4.0 s）以降の予測だけ（それより前の発火は数えず、名札の記憶にも入れない・出力を後で捨てる形ではない）
  - 通過（距離クラス）と発進（B2）: 物理の最接近 t_cpa < 4.0 s は主から外す（history_short・warmup_reason = warmup_cpa）。発進の主の列は「最接近 − 発火」のまま、
    「動き出しからの遅れ」delay_move = 窓の中の最初の知らせ − max(動き出し, 4.0)。B2 で動き出しの読み（t_move_s）が無い本があれば止める
  - 警告音: 鳴り終わり（t_warn_end = 波形で読んだ鳴り終わり〔Astra R142-05・暖機の判定だけ〕・無ければ t_end・それも無ければ t_cpa = 鳴り始め + 3 s）≤ 4.0 s だけ外す（warmup_end）。採点上の鳴り始め t_score_on = max(鳴り始め, 4.0)・
    成功の窓 = [t_score_on − 1, t_score_on + 4]・遅れ = 発火 − t_score_on。物理の鳴り始めは t_start の列のまま（上書きしない）
  - --min-history は 4.0（ほかの値と一緒には使えない）。events に t_score_on・warmup_reason・delay_move の列。報告の頭に暖機の除外の数
多重車（R06 の暫定）: n_car ≥ 2 の行は「群のいずれかへの通知」として別集計（車両単位の到達率と混ぜない）。
負例（class=none）: 半開区間 [t_start, t_cpa) の担当区間で v4.3＋警告音規則（既定 v9c）の発火を数える（注釈イベント窓はマスク）。誤警告率は両側 95% CI と片側 95% 上限。
  2026-09-29（Opus 説明係の指摘）: 区分 G（仕込んだ 60 秒の負例テイク）と 負例露出（連続録音）を**分けて**数える。事前登録の合格線（片側上限 < 1.8 回/h）は
  負例露出の群に掛かる。負例の本（区分 G・負例露出）に偶然入った相手（class≠none）は「計画外」として主要評価に入れず別掲（窓のマスクだけ効かせる）。

入力列: clip_id,event_id,take_id,pair_id,trial,class,quadrant,t_start,t_cpa,区分,状態[,横距離m,n_car,scored,orig_file,cut_offset_s,...]
予測 CSV: 7 列 [clip,frame,class,track,az,el,dist]（因果推論の val_all_causal.csv 形式）
使い方: python scripts/step20_realsmoke_score_v3.py --pred <csv> --ann <csv> [--out md] [--events-out csv] [--pred-b csv]
        [--dir-window 2.5-1.5] [--min-history 7.5] [--exclude-front] [--gt-dirwin <dir> [--gt-dir <dir>]]
--gt-dirwin（2026-10-26 Fable 仕様（先行分）§5・決定 ⑲・Astra R30-03／09）: 方向の正解を記録シートの象限ではなく**正解の軌道**から取る。
  距離クラス = step19k の gt_dirwin/<clip>.csv（最接近の 2.5〜1.5 s 前の窓の正解の方位）・警告音クラス = step19k の gt/<clip>.csv の
  [t_start, t_start + --warn-dir-span] の正解の方位（--gt-dir・既定は gt_dirwin の隣の gt）。窓の中の方位は円周平均（outdoor_seld.geometry.circular_mean_deg・
  打ち消し合うと決まらない）→ quadrant_of で象限。軌道の無い本・窓に正解の無い本・決まらない本は**主の方向の集計で欠測**（分母に入れない・本数を報告）。
  記録シートの象限での代用は参考の別表（主と混ぜない）。食い違い（軌道の象限 ≠ 記録シートの象限）の本数も 1 行。付けないときの出力は今までと 1 バイトも同じ。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v4 = _load("nv4", "step12_notify_v4_ttc.py")
V43 = _load("nv43", "step12_notify_v43.py")
H = _load("nhold", "step12_notify_v9b_hold.py")
V9C = _load("nv9c", "step12_notify_v9c_multitrack.py")   # 警告音の規則 v9c（名札で 1 台ずつ追跡・変更票 #41・2026-09-28 16:13 本人採用）
V2 = _load("s20v2", "step20_realsmoke_score_v2.py")     # 統計関数・負例の重なり控除・列定義を再利用
# 警告音の規則（変更票 #41・凍結文書 改訂 r5）: 既定 = v9c。旧 hold は --warn-rule hold で並べて採点する（統一採点器と同じ切り替え）
WARN_RULE = "v9c"
WARN_RULES = ("v9c", "hold")
WARN_GATE_DEG, WARN_DEDUP_DEG = 45.0, 30.0


def warn_fires(fw, nframes: int):
    """警告音の発火 [(frame, cls, az)]。WARN_RULE = v9c（既定）か hold（旧・併記用）"""
    if WARN_RULE == "hold":
        return H.warn_fires(fw, hold=True, nframes=nframes)
    if WARN_RULE == "v9c":
        return V9C.warn_fires_v9c(fw, WARN_GATE_DEG, nframes=nframes, dedup_deg=WARN_DEDUP_DEG)
    raise ValueError(f"未知の警告音の規則: {WARN_RULE}（{WARN_RULES}）")

FPS = 10.0
T3, SUPP = 1.5, 3.2                     # GT 区分（横距離 = 水平距離）: critical ≤1.5 / caution ≤3.2 / safe >3.2
WIN_PRE, WIN_POST = 1.0, 1.0
WARMUP_V30 = False                  # 本体 §2-6・§9-1 の 1・§9-4 の 1: --warmup-v30（既定は旧のまま）
WARMUP_S, RULE_START = 4.0, 40      # 暖機 4.0 s ＝ 規則への入力を frame 40 から
V30_FIELDS = ["t_score_on", "warmup_reason", "delay_move"]
CLS_IDX = V2.CLS_IDX
DIST_CLS = V2.DIST_CLS
quadrant_of, gt_tier_of, is_scored, lateral_of = V2.quadrant_of, V2.gt_tier_of, V2.is_scored, V2.lateral_of
lateral_range_of = V2.lateral_range_of
GT_DIRWIN = None          # --gt-dirwin のときだけ {clip: [(frame, cls, track, az)]}（先行分 §5）。None = 旧の動き（記録シートの象限）
GT_FULL = None            # 同じく gt/<clip>.csv（警告音の鳴り始めの窓に使う）


def load_gt_rows(d: Path) -> dict:
    """step19k の gt／gt_dirwin（6 列 frame,cls,track,az,el,dist・見出しなし）→ {clip: [(frame, cls, track, az)]}。ファイルがある本だけ。"""
    out = {}
    for p in sorted(Path(d).glob("*.csv")):
        rows = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                v = ln.split(",")
                rows.append((int(v[0]), int(v[1]), int(v[2]), float(v[3])))
        out[p.stem] = rows
    return out


def quadrant_gt(clip: str, ci: int, event_id, t_start: float, warn_dir_span: float) -> tuple[str, str]:
    """正解の軌道の象限（先行分 §5）。返り値 (象限, 欠測の理由)。象限が決まれば理由は空。
    距離クラス = gt_dirwin の窓の行・警告音 = gt の [t_start, t_start + warn_dir_span] の行。同じクラスに track = event_id の行があればそれだけを使う。"""
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from outdoor_seld.geometry import circular_mean_deg
    src = GT_DIRWIN if ci in DIST_CLS else GT_FULL
    if src is None or clip not in src:
        return "", "軌道なし（gt_dirwin／gt に本が無い）"
    rows = [r for r in src[clip] if r[1] == ci]
    if ci not in DIST_CLS:
        rows = [r for r in rows if t_start - 1e-9 <= (r[0] + 0.5) / FPS <= t_start + warn_dir_span + 1e-9]
    try:
        ev = int(str(event_id).strip())
    except ValueError:
        ev = None
    if ev is not None and any(r[2] == ev for r in rows):
        rows = [r for r in rows if r[2] == ev]
    if not rows:
        return "", "窓に正解が無い"
    m = circular_mean_deg([r[3] for r in rows])
    if m is None:
        return "", "窓の方位が打ち消し合って決まらない"
    return quadrant_of(m), ""


def tier_of_range(rng):
    """横距離の幅 (lo, hi) → GT 区分。lo と hi が同じ区分ならその区分、区分の境界（1.5 / 3.2 m）をまたぐ幅は "boundary:<lo側>/<hi側>"（R11）。"""
    a, b = gt_tier_of(rng[0]), gt_tier_of(rng[1])
    return a if a == b else f"boundary:{a}/{b}"


def _arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def load_pred7(path: Path):
    """clip -> frame -> [(cls, az, el, d)]。7 列（因果推論）が前提。6 列（el なし）も読む。"""
    out = defaultdict(lambda: defaultdict(list))
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) >= 7:
            clip, k, c, az, el = p[0], int(p[1]), int(p[2]), float(p[4]), float(p[5])
            d = float(p[6]) if p[6] not in ("", "nan") else float("nan")
        elif len(p) == 6:
            clip, k, c, az, el, d = p[0], int(p[1]), int(p[2]), float(p[3]), float(p[4]), float(p[5])
        else:
            continue
        out[clip][k].append((c, az, el, d))
    return dict(out)


NOTIFY_RULE = "v43"                 # 変更票 #73（Q2）: 既定 = v4.3。--notify-rule detect = 「検出したら知らせる（距離を見ない）」（scripts/_notify_detect.py・比較用・統一採点器と同じ）
NOTIFY_RULES = ("v43", "detect")
DET = _load("ndetect20", "_notify_detect.py")


def fires_of(frames, cfg43, nframes):
    """v4.3＋hold の発火 [(t, az, tier, cls)]（tier: 強/中/警告）。frames: frame -> [(cls, az, el, d)]。NOTIFY_RULE = detect なら距離の規則だけ _notify_detect に替える"""
    if WARMUP_V30:                                   # §9-1 の 1: 規則への入力を frame 40 から（状態・名札の記憶もそこから）
        frames = {k: v for k, v in frames.items() if k >= RULE_START}
    fd = {k: [(c, az, d) for c, az, el, d in v if c in DIST_CLS and math.isfinite(d)] for k, v in frames.items()}
    fw = {k: [(c, az, el) for c, az, el, d in v] for k, v in frames.items()}
    fires = []
    if NOTIFY_RULE == "detect":
        res = DET.run_rule_detect({"x": fd}, cfg43, nframes=nframes).get("x", {})
    else:
        res = V43.run_rule3({"x": fd}, cfg43, nframes=nframes).get("x", {})
    for cls, lst in res.items():
        for j, az, tier, d in lst:
            fires.append(((j + 1) / FPS, az, tier, cls))
    for k, cls, az in warn_fires(fw, nframes):
        fires.append(((k + 1) / FPS, az, "警告", cls))
    return sorted(fires)


def frame_recall(frames, cls_idx, t0, t1):
    """イベント窓 [t0, t1] のうち、そのクラスの推定があるフレームの割合（歩行対比の主指標。v2 と同じ定義）。"""
    k0, k1 = max(int(math.floor(t0 * FPS)), 0), int(math.ceil(t1 * FPS))
    if k1 <= k0:
        return None
    hit = sum(1 for k in range(k0, k1) if any(c == cls_idx for c, az, el, d in frames.get(k, [])))
    return round(hit / (k1 - k0), 3)


def median_az(frames, cls_idx, t0, t1):
    """窓 [t0, t1] のそのクラスの推定方位の中央（単位ベクトル平均）。無ければ None。"""
    k0, k1 = int(math.floor(t0 * FPS)), int(math.ceil(t1 * FPS))
    xs, ys = [], []
    for k in range(max(k0, 0), k1 + 1):
        for c, az, el, d in frames.get(k, []):
            if c == cls_idx:
                xs.append(math.cos(math.radians(az))); ys.append(math.sin(math.radians(az)))
    if not xs:
        return None
    return math.degrees(math.atan2(sum(ys), sum(xs)))


NEG_GROUPS = ("G", "負例露出")          # 負例の区分（記録シートの区分）。G = 仕込んだ 60 秒 × 20 本、負例露出 = 連続録音 100 分
NEG_PASS_PER_H = 1.8                    # 旧の合格線（#70 r2・2026-10-19 本人: 合否は出さない。回数・正味の時間・1 時間あたり・片側 95% 上限を出すだけ。値は記録として残す）。旧の説明:（変更票: 片側 95% 上限 < 1.8 回/h・負例露出に掛かる。本人 9/28 決定 (a): 線はそのまま・実質 100 分で 0 回）


def neg_group(r: dict) -> str:
    k = (r.get("区分") or "").strip()
    return k if k in NEG_GROUPS else "その他"


def warn_end_of(r) -> float:
    """警告音の鳴り終わり（v30 の暖機）: t_warn_end（波形で読んだ鳴り終わり・field_score_members が区間の表から入れる・Astra R142-05）があればそれ・
    無ければ t_end・それも無ければ t_cpa（step19d の 鳴り始め + 3 s）。t_warn_end は暖機の判定だけに使う（通知の窓のマスクは t_end のまま）。"""
    t_end = (r.get("t_warn_end") or "").strip() or (r.get("t_end") or "").strip()
    return float(t_end) if t_end else float(r["t_cpa"])


def warmup_v30(r, ci) -> tuple[int, str]:
    """v30 の暖機の除外（§9-1 の 1）: 通過・発進 = 最接近 < 4.0 s・警告音 = 鳴り終わり ≤ 4.0 s。"""
    if ci in DIST_CLS:
        return (1, "warmup_cpa") if float(r["t_cpa"]) < WARMUP_S else (0, "")
    return (1, "warmup_end") if warn_end_of(r) <= WARMUP_S else (0, "")


def _evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span=1.0):
    if WARMUP_V30:                                   # §2-6: 発進（B2）の基準 = 動き出しの秒（台帳の列）。無ければ止める
        miss = [f"{r['clip_id']}/{r.get('event_id', '')}" for r in rows if (r.get("台本ID") or "").strip().upper().startswith("B2") and not (r.get("t_move_s") or "").strip()]
        if miss:
            raise SystemExit(f"--warmup-v30: 発進（B2）に動き出しの読み（t_move_s）が無い本がある（{len(miss)} 本・例 {miss[:3]}）→ 止める")
    need = defaultdict(float)
    for r in rows:
        need[r["clip_id"]] = max(need[r["clip_id"]], float(r["t_cpa"]) + WIN_POST + 1.0)
        if WARMUP_V30 and r["class"].strip() != "none" and CLS_IDX.get(r["class"].strip()) is not None and CLS_IDX.get(r["class"].strip()) not in DIST_CLS:
            need[r["clip_id"]] = max(need[r["clip_id"]], max(float(r["t_start"]), WARMUP_S) + 4.0 + 1.0)
    fires_by_clip = {}
    for clip in {r["clip_id"] for r in rows}:
        fr = pred.get(clip, {})
        nf = int(max(need[clip] * FPS, (max(fr.keys()) + 1) if fr else 0, 100)) + 1
        fires_by_clip[clip] = fires_of(fr, cfg43, nf) if fr else []

    pos_windows = defaultdict(list)
    unknown_windows = defaultdict(list)
    for r in rows:
        if r["class"].strip() != "none":
            t_end = (r.get("t_end") or "").strip()                           # #69 r18（B8・2026-10-20）: 長回しの「② の終わり」= 相手がいた区間の終わり（step19d の 終わり秒）
            if (r.get("t_end_unknown") or "").strip() == "1":                 # R72-03: 終わり不明の長い相手 = 始まりから録音末尾まで（この clip の最後まで）除外・不明として別計上
                pos_windows[r["clip_id"]].append((float(r["t_start"]) - WIN_PRE, float("inf")))
                unknown_windows[r["clip_id"]].append((float(r["t_start"]) - WIN_PRE, float("inf")))
            else:
                pos_windows[r["clip_id"]].append((float(r["t_start"]) - WIN_PRE, (float(t_end) if t_end else float(r["t_cpa"])) + WIN_POST))

    def masked(clip, t):
        return any(a <= t <= b for a, b in pos_windows[clip])

    events, n_false, exposure_s = [], 0, 0.0
    neg_by = defaultdict(lambda: {"n_false": 0, "exposure_s": 0.0, "unknown_s": 0.0, "n_unknown": 0})     # 区分ごと（G／負例露出／その他）・unknown = 終わり不明の長い相手で除外した秒と本数（R72-03）
    extras = defaultdict(int)
    used_by_clip = {clip: [False] * len(f) for clip, f in fires_by_clip.items()}

    def meta(r):
        return {"take_id": r.get("take_id", ""), "pair_id": r.get("pair_id", ""), "state": r.get("状態", ""),
                "n_car": r.get("n_car", ""), "session": r.get("session_id", "")}

    for r in sorted(rows, key=lambda x: (x["clip_id"], float(x["t_cpa"]))):
        clip, cls = r["clip_id"], r["class"].strip()
        if cls == "none":
            t0, t1 = float(r["t_start"]), float(r["t_cpa"])
            # 重なり控除は v2 と同じ考え方: 担当区間からイベント窓を引く
            segs = sorted((max(t0, a), min(t1, b)) for a, b in pos_windows[clip] if b > t0 and a < t1)
            ov, cur = 0.0, None
            for a, b in segs:
                if cur is None or a > cur[1]:
                    if cur:
                        ov += cur[1] - cur[0]
                    cur = [a, b]
                else:
                    cur[1] = max(cur[1], b)
            if cur:
                ov += cur[1] - cur[0]
            exp_here = (t1 - t0) - ov
            nf_here = sum(1 for (t, az, tier, c) in fires_by_clip[clip] if t0 <= t < t1 and not masked(clip, t))
            exposure_s += exp_here; n_false += nf_here
            g = neg_by[neg_group(r)]; g["exposure_s"] += exp_here; g["n_false"] += nf_here
            uw = [(max(t0, a), min(t1, b)) for a, b in unknown_windows[clip] if b > t0 and a < t1]      # 不明の区間がこの担当区間から除いた秒（報告用・既知の窓との重なりは引かない）
            if uw:
                g["unknown_s"] += sum(max(0.0, bb - aa) for aa, bb in uw); g["n_unknown"] += len(uw)
            continue
        base = {"clip": clip, "trial": r.get("trial", ""), "event_id": r.get("event_id", "1"), "class": cls,
                "quadrant": r.get("quadrant", r.get("象限", "")).strip(), **meta(r),
                "history_short": int(float(r["t_cpa"]) < min_history), "front": int(r.get("quadrant", r.get("象限", "")).strip() == "F"),
                "multi": int(str(r.get("n_car", "1")).strip() not in ("", "1")),
                "walk": int(bool(r.get("pair_id", "").strip()) or r.get("区分", "").strip() == "歩行"),
                "unplanned": int(neg_group(r) in NEG_GROUPS),      # 負例の本に偶然入った相手 = 計画外（主要評価に入れない・窓のマスクだけ効く）
                "delay": None, "frame_recall": None, "boundary": 0,
                "t_move": (float(r["t_move_s"]) if (r.get("t_move_s") or "").strip() else None), "before_move": None,   # #69 r23: 発進の動き出し
                "script_id": (r.get("台本ID") or "").strip().upper()}
        if WARMUP_V30 and CLS_IDX.get(cls) is not None:      # §9-1 の 1: 暖機の除外と採点上の時刻（物理の t_start は上書きしない）
            _ci30 = CLS_IDX[cls]
            base["history_short"], base["warmup_reason"] = warmup_v30(r, _ci30)
            if _ci30 not in DIST_CLS:
                base["t_score_on"] = round(max(float(r["t_start"]), WARMUP_S), 3)
            elif base["t_move"] is not None:
                base["t_score_on"] = round(max(base["t_move"], WARMUP_S), 3)
        if GT_DIRWIN is not None:                            # 先行分 §5: 方向の正解 = 正解の軌道（記録シートの象限は quadrant_sheet に残す）
            _ci = CLS_IDX.get(cls)
            _q, _why = quadrant_gt(clip, _ci, base["event_id"], float(r["t_start"]), warn_dir_span) if _ci is not None else ("", "知らないクラス")
            base.update(quadrant_sheet=base["quadrant"], quadrant_gt=_q, quad_gt_missing=int(not _q), quad_gt_why=_why)
        qref = base["quadrant_gt"] if GT_DIRWIN is not None else base["quadrant"]
        if not is_scored(r):
            extras["n_maskonly"] += 1
            events.append({**base, "gt_tier": "mask", "notified": None, "fired_tier": None, "lead": None, "quad_ok": None, "az_est": None})
            continue
        ci = CLS_IDX.get(cls)
        if ci is None:
            extras["n_unknown_class"] += 1
            continue
        frames = pred.get(clip, {})
        t0, t1 = float(r["t_start"]) - WIN_PRE, float(r["t_cpa"]) + WIN_POST
        # ③ 方向: 距離クラスは注釈の窓 [t_cpa − dir_win[0], t_cpa − dir_win[1]]、警告音クラスは [t_start, t_cpa]（ラップ＝音の始まり）の推定方位の中央
        if ci in DIST_CLS:
            az_est = median_az(frames, ci, float(r["t_cpa"]) - dir_win[0], float(r["t_cpa"]) - dir_win[1])
        else:
            az_est = median_az(frames, ci, float(r["t_start"]), float(r["t_start"]) + warn_dir_span)   # 鳴り始め直後（N02）
        # 変更票 #69 r31 ①（2026-10-23）: 検出フレーム率の分母 = 相手の中点が道に沿って ±10 に入っている時刻（step19l → step19d の t_in10／t_out10）。
        #   無い本（読みが無い・欠測）は旧の窓 [t_start, t_cpa] で出し、fr_window に "start_cpa" と残して本数を報告する（黙って混ぜない）
        if (r.get("t_in10") or "").strip() and (r.get("t_out10") or "").strip():
            base["frame_recall"] = frame_recall(frames, ci, float(r["t_in10"]), float(r["t_out10"])); base["fr_window"] = "in10"
            extras["n_fr_in10"] += 1
        else:
            base["frame_recall"] = frame_recall(frames, ci, float(r["t_start"]), float(r["t_cpa"])); base["fr_window"] = "start_cpa"
            extras["n_fr_start_cpa"] += 1
        quad_ok = None if not qref else (None if az_est is None else quadrant_of(az_est) == qref)
        if GT_DIRWIN is not None:                            # 参考（主と混ぜない）: 記録シートの象限で比べた場合
            base["quad_ok_sheet"] = None if not base["quadrant_sheet"] else (None if az_est is None else quadrant_of(az_est) == base["quadrant_sheet"])
        fires = fires_by_clip[clip]; used = used_by_clip[clip]
        if ci in DIST_CLS:
            rng = lateral_range_of(r)
            if rng is None:
                extras["n_unscored"] += 1
                events.append({**base, "gt_tier": "-", "notified": None, "fired_tier": None, "lead": None, "quad_ok": quad_ok, "az_est": az_est})
                continue
            tier = tier_of_range(rng)
            in_w = [i for i, (t, az, ft, c) in enumerate(fires) if c == ci and ft in ("強", "中") and t0 <= t <= t1]
            _fa = [i for i in in_w if not used[i]]
            if base["t_move"] is not None:
                base["before_move"] = int(bool(_fa) and fires[_fa[0]][0] < base["t_move"])   # #69 r23: 最初の知らせが動き出しより前（かける音・記述だけ）
                if WARMUP_V30:                               # §9-1 の 1: 発進の「動き出しからの遅れ」（別の列・主の列は 最接近 − 発火 のまま）
                    base["delay_move"] = round(fires[_fa[0]][0] - max(base["t_move"], WARMUP_S), 2) if _fa else None
            lf = (round(float(r["t_cpa"]) - fires[_fa[0]][0], 2) if _fa else None)      # #70 r12／R109-04: 窓の中の最初の知らせ（強さを問わない・まだ他の事象に付いていないもの）までのリード
            if tier.startswith("boundary"):
                # 境界例（幅が区分の境界をまたぐ）: 区分別の主要評価には入れず、「強・中の通知が出たか」だけを別掲する（R11）
                extras["n_boundary"] += 1
                events.append({**base, "gt_tier": tier, "notified": bool(in_w), "fired_tier": (fires[in_w[0]][2] if in_w else None),
                               "lead": (round(float(r["t_cpa"]) - fires[in_w[0]][0], 2) if in_w else None), "quad_ok": quad_ok, "az_est": az_est, "boundary": 1})
                continue
            if tier == "safe":
                extras["n_mid_on_safe"] += sum(1 for i in in_w if fires[i][2] == "中")
                events.append({**base, "gt_tier": tier, "notified": not in_w, "fired_tier": (fires[in_w[0]][2] if in_w else None),
                               "lead": None, "quad_ok": quad_ok, "az_est": az_est})
                continue
            pick = None
            for i in in_w:
                if used[i]:
                    continue
                if tier == "critical" and fires[i][2] != "強":
                    continue
                pick = i; break
            if pick is not None:
                used[pick] = True
                if tier == "caution" and fires[pick][2] == "強":
                    extras["n_strong_on_caution"] += 1
                events.append({**base, "gt_tier": tier, "notified": True, "fired_tier": fires[pick][2],
                               "lead": round(float(r["t_cpa"]) - fires[pick][0], 2), "lead_first": lf, "quad_ok": quad_ok, "az_est": az_est})
            else:
                events.append({**base, "gt_tier": tier, "notified": False, "fired_tier": None, "lead": None, "lead_first": lf, "quad_ok": quad_ok, "az_est": az_est})
        else:
            t_on = float(r["t_start"])
            if WARMUP_V30:                                   # §9-4 の 1: 採点上の鳴り始め = max(鳴り始め, 4.0)・窓 [−1, ＋4]・遅れもそこから
                t_on = max(t_on, WARMUP_S)
                t0, t1 = t_on - WIN_PRE, t_on + 4.0
            in_w = [i for i, (t, az, ft, c) in enumerate(fires) if c == ci and ft == "警告" and t0 <= t <= t1 and not used[i]]
            if in_w:
                used[in_w[0]] = True
                # 方向 = 通知が出た瞬間の推定方位（記録紙の「鳴り始めの向き」と、その瞬間の推定を比べる・再監査2 Q06）。窓の平均は使わない
                az_fire = fires[in_w[0]][1]
                quad_ok_f = None if not qref else quadrant_of(az_fire) == qref
                if GT_DIRWIN is not None:
                    base["quad_ok_sheet"] = None if not base["quadrant_sheet"] else quadrant_of(az_fire) == base["quadrant_sheet"]
                events.append({**base, "gt_tier": "warn", "notified": True, "fired_tier": "警告",
                               "lead": None, "delay": round(fires[in_w[0]][0] - t_on, 2),   # 遅れ = 発火 − 鳴り始め（N02・v30 は採点上の鳴り始め）
                               "quad_ok": quad_ok_f, "az_est": az_fire})
            else:
                events.append({**base, "gt_tier": "warn", "notified": False, "fired_tier": None, "lead": None, "quad_ok": quad_ok, "az_est": az_est})
    return events, {"n_false": n_false, "exposure_s": exposure_s, "by": {k: dict(v) for k, v in neg_by.items()}}, extras


def evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span=1.0, t3=None):
    """変更票 #69 r20（本人 10/18 22:56・B11 2026-10-20）: t3 を渡すと至近の境目（既定 1.5 m）をその値にして GT 区分を引き直す（感度の確かめ用）。
    主の採点は t3=None（1.5 m）。境目は V2.gt_tier_of の T3 なので、呼ぶ間だけ差し替えて必ず戻す。"""
    if t3 is None:
        return _evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span)
    old = V2.T3
    try:
        V2.T3 = float(t3)
        return _evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span)
    finally:
        V2.T3 = old


def summarize(events, include_front):
    """主要評価（前方除外・履歴不足除外・単車・計画外除外）と参考集計を分ける。"""
    scored = [e for e in events if e["notified"] is not None]
    main = [e for e in scored if (include_front or not e["front"]) and not e["history_short"] and not e["multi"] and not e["walk"] and not e.get("boundary") and not e.get("unplanned")]
    side = {"boundary": [e for e in scored if e.get("boundary") and not e.get("unplanned")],
            "front": [e for e in scored if e["front"] and not include_front and not e["walk"] and not e.get("unplanned")],
            "history_short": [e for e in scored if e["history_short"] and not e["walk"] and not e.get("unplanned")],
            "multi": [e for e in scored if e["multi"] and not e["walk"] and not e.get("unplanned")],
            "walk": [e for e in scored if e["walk"] and not e.get("unplanned")],
            "unplanned": [e for e in scored if e.get("unplanned")]}

    def rates(evs):
        by = defaultdict(lambda: [0, 0])
        for e in evs:
            by[e["gt_tier"]][0] += int(bool(e["notified"])); by[e["gt_tier"]][1] += 1
        return by
    return main, side, rates


def quadrant_breakdown(main_ev: list[dict], walk_ev: list[dict] | None = None, title: str | None = None, note: str | None = None) -> list[str]:
    """変更票 #57（2026-10-10 本人「内訳として出すのはいいですね」・2026-10-12 実装）: 主要評価の各指標を象限（F／B／L／R・記録の決まりのまま）ごとに
    本数と値で並べる補助の表。主要評価は #40 のまま分けない。数が少ないので割合と本数を並べるだけ（検定しない・差があっても無くても出す・#39）。
    歩く比較は相手ごとに向きが一つに決まる（車は後ろから・ベルは向かってくる）ので、方向の差とクラスの差を分けられない → 参考と書く。"""
    import numpy as _np
    L = ["", title or "## 補助: 来る方向（象限）の内訳（変更票 #57・記述だけ・検定しない・結果に関係なく出す）",
         "- 象限 = 記録の決まり（距離クラスは最接近の 2.5〜1.5 秒前の向き・警告音は鳴り始めの向き）。F = 前→後・B = 後→前・L／R = 横。主要評価（上）は方向で分けていない。",
         "", "| 象限 | 本数 | 成功（到達） | 区分別 成功/本数 | 方向一致（全分母） | リード中央値[s] | 遅れ中央値[s] | 検出フレーム率中央値 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]

    def row(label, evs):
        if not evs:
            return f"| {label} | 0 | — | — | — | — | — | — |"
        n = len(evs); ok = sum(1 for e in evs if e["notified"])
        by = defaultdict(lambda: [0, 0])
        for e in evs:
            by[e["gt_tier"]][0] += int(bool(e["notified"])); by[e["gt_tier"]][1] += 1
        tiers = "、".join(f"{t} {by[t][0]}/{by[t][1]}" for t in ("critical", "caution", "safe", "warn") if by[t][1])
        dq = [e for e in evs if e["quadrant"]]; n_dir_ok = sum(1 for e in dq if e["quad_ok"])
        leads = [e["lead"] for e in evs if e.get("lead") is not None]; delays = [e["delay"] for e in evs if e.get("delay") is not None]
        frs = [e["frame_recall"] for e in evs if e.get("frame_recall") is not None]
        med = lambda v, f="%.1f": (f % _np.median(v)) if v else "—"
        return f"| {label} | {n} | {ok}/{n}（{100.0 * ok / n:.0f}%） | {tiers or '—'} | {n_dir_ok}/{len(dq)} | {med(leads)} | {med(delays, '%.2f')} | {med(frs, '%.3f')} |"
    quads = ("F", "B", "L", "R")
    for q in quads:
        L.append(row(f"{q}（{dict(F='前→後', B='後→前', L='左', R='右')[q]}）", [e for e in main_ev if (e.get("quadrant") or "") == q]))
    L.append(row("象限の記入なし", [e for e in main_ev if not e.get("quadrant")]))
    L.append(row("全部（主要評価と同じ）", list(main_ev)))
    # クラス × 象限（本数と成功だけ）
    cls_list = sorted({e["class"] for e in main_ev})
    if cls_list:
        L += ["", "| クラス | " + " | ".join(f"{q}: 成功/本数" for q in quads) + " |", "| --- | " + " | ".join("---" for _ in quads) + " |"]
        for c in cls_list:
            cells = []
            for q in quads:
                evs = [e for e in main_ev if e["class"] == c and (e.get("quadrant") or "") == q]
                cells.append(f"{sum(1 for e in evs if e['notified'])}/{len(evs)}" if evs else "—")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
    if walk_ev:
        L += ["", "- 歩く比較（参考・相手ごとに向きが一つに決まるので方向の差とクラスの差は分けられない）: "
              + "、".join(f"{q} {sum(1 for e in walk_ev if (e.get('quadrant') or '') == q and e['notified'])}/{sum(1 for e in walk_ev if (e.get('quadrant') or '') == q)}" for q in quads
                          if any((e.get("quadrant") or "") == q for e in walk_ev))]
    L.append("- 読み方: 本数が少ない象限（例 C は片側 4 本）は率がぶれる。「差がある」とは書かず、本数と値を並べて報告する（#57）。")
    return L


EVENT_FIELDS = ["clip", "session", "take_id", "pair_id", "state", "trial", "event_id", "class", "n_car", "quadrant",
                "gt_tier", "notified", "fired_tier", "lead", "lead_first", "delay", "frame_recall", "az_est", "quad_ok", "front", "history_short", "multi", "walk", "boundary", "unplanned", "t_move", "before_move"]
GT_DIRWIN_FIELDS = ["quadrant_sheet", "quadrant_gt", "quad_gt_missing", "quad_gt_why", "quad_ok_sheet"]   # --gt-dirwin のときだけ events の列に足す（先行分 §5）


def gt_dirwin_lines(main_ev: list[dict]) -> list[str]:
    """先行分 §5・R30-03／09: 方向の主の集計（正解 = 軌道・欠測は分母に入れない）・食い違い・記録シートで代用した参考（別表）。"""
    have = [e for e in main_ev if e.get("quadrant_gt")]
    miss = [e for e in main_ev if e.get("quad_gt_missing")]
    n_all = len(have); n_est = sum(1 for e in have if e["quad_ok"] is not None); n_ok = sum(1 for e in have if e["quad_ok"])
    why = defaultdict(int)
    for e in miss:
        why[e.get("quad_gt_why") or "?"] += 1
    both = [e for e in have if e.get("quadrant_sheet")]
    n_diff = sum(1 for e in both if e["quadrant_sheet"] != e["quadrant_gt"])
    sub = [e for e in miss if e.get("quadrant_sheet")]
    s_est = sum(1 for e in sub if e.get("quad_ok_sheet") is not None); s_ok = sum(1 for e in sub if e.get("quad_ok_sheet"))
    L = [f"- 方向 4 象限一致（正解 = 正解の軌道の窓の方位の円周平均・--gt-dirwin・先行分 §5）: 全イベント分母 **{n_ok}/{n_all}**（未検出 {n_all - n_est} 件は不一致扱い）／推定があった例だけ {n_ok}/{n_est}",
         f"- 方向の正解が欠測（軌道が無い・窓に正解が無い・決まらない）: **{len(miss)} 本**（主の分母に入れない）" + (f"＝" + "・".join(f"{k} {v}" for k, v in sorted(why.items())) if why else ""),
         f"- 軌道の象限と記録シートの象限の食い違い: {n_diff}/{len(both)} 本（両方ある本）",
         f"- 参考（主と混ぜない・欠測の本を記録シートの象限で代用した場合）: 一致 {s_ok}/{len(sub)}（推定があった例だけ {s_ok}/{s_est}）"]
    return L


def walk_pairs_summary(walk_events):
    """歩行対比: (pair_id, class) ごとに静止側・歩行側を対応づけ、検出フレーム率の差を出す。

    同じ (pair, 状態, class) に事象が 2 件以上ある対は「対象が 1 件に決まらない」として未集計にし、ambiguous に返す（黙って最後の 1 件に置き換えない・再監査2 Q04）。"""
    by = defaultdict(lambda: defaultdict(list))
    for e in walk_events:
        by[(e["pair_id"], e["class"])][e["state"]].append(e)
    rows, diffs, ambiguous = [], [], []
    for (pid, cls), d in sorted(by.items()):
        if any(len(v) > 1 for v in d.values()):
            ambiguous.append((pid, cls, {k: len(v) for k, v in d.items()}))
            continue
        a = d.get("静止", [None])[0]; b = d.get("歩行", [None])[0]
        fa = a["frame_recall"] if a else None; fb = b["frame_recall"] if b else None
        if fa is not None and fb is not None:
            diffs.append(fb - fa)
        rows.append((f"{pid}/{cls}", a, b, fa, fb))
    return rows, diffs, ambiguous


def main() -> int:
    global WARN_RULE
    WARN_RULE = _arg("--warn-rule", WARN_RULE)            # 変更票 #41: 既定 v9c。旧 hold は併記用
    global NOTIFY_RULE
    NOTIFY_RULE = _arg("--notify-rule", NOTIFY_RULE)      # 変更票 #73（Q2）: v43（既定）か detect（比較用）
    if NOTIFY_RULE not in NOTIFY_RULES:
        raise ValueError(f"--notify-rule は {NOTIFY_RULES} のどれか: {NOTIFY_RULE}")
    if WARN_RULE not in WARN_RULES:
        raise ValueError(f"--warn-rule は {WARN_RULES} のどれか: {WARN_RULE}")
    pred_path, ann_path = Path(_arg("--pred")), Path(_arg("--ann"))
    out_md = Path(_arg("--out", str(ROOT / "out" / "realsmoke" / "score_v3.md")))
    events_out = Path(_arg("--events-out", str(out_md.with_name(out_md.stem + "_events.csv"))))
    a, b = _arg("--dir-window", "2.5-1.5").split("-")
    dir_win = (float(a), float(b))
    global WARMUP_V30
    WARMUP_V30 = "--warmup-v30" in sys.argv              # 本体 §2-6・§9-1 の 1・§9-4 の 1（既定は旧のまま＝発効のあとで既定に）
    min_history = float(_arg("--min-history", "4.0" if WARMUP_V30 else "7.5"))
    if WARMUP_V30 and min_history != WARMUP_S:
        raise ValueError(f"--warmup-v30 の暖機は {WARMUP_S} s（--min-history {min_history} とは一緒に使えない）")
    warn_dir_span = float(_arg("--warn-dir-span", "1.0"))
    global GT_DIRWIN, GT_FULL
    gt_dirwin_dir = _arg("--gt-dirwin")
    if gt_dirwin_dir:                                    # 先行分 §5（2026-10-26）: 象限の正解を正解の軌道から
        GT_DIRWIN = load_gt_rows(Path(gt_dirwin_dir))
        GT_FULL = load_gt_rows(Path(_arg("--gt-dir", str(Path(gt_dirwin_dir).parent / "gt"))))
    include_front = "--exclude-front" not in sys.argv      # 変更票 #40・B13（2026-10-20）: 既定で前方を含む（--include-front は旧の明示・受け付ける）
    winner = Path(_arg("--winner", str(ROOT / "out/notify_v43_sweep/winner.json")))
    cfg43 = V43.Cfg43(**json.loads(winner.read_text(encoding="utf-8")))
    pred = load_pred7(pred_path)
    rows = list(csv.DictReader(open(ann_path, encoding="utf-8-sig")))
    events, neg, extras = evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span)
    main_ev, side, rates = summarize(events, include_front)
    by = rates(main_ev)
    hours = neg["exposure_s"] / 3600.0
    label = {"critical": "critical 到達（強）", "caution": "caution 到達（中以上）", "safe": "safe 抑制（強・中とも無し）", "warn": f"警告音 到達（{WARN_RULE}）"}
    rep = [f"# 実録採点 v3（v4.3＋警告音 {WARN_RULE}・{V43.label43(cfg43)}・pred={pred_path.name}）", "",
           f"- 通知規則: {'v4.3' if NOTIFY_RULE == 'v43' else '検出したら知らせる（距離を見ない・比較用・#73 Q2）'}（{winner.name}）＋警告音 {WARN_RULE}{'（名札方式・門 45°・同一方位 30° を 1 つに・名札ごとに 1 回・寿命 5 s・変更票 #41）' if WARN_RULE == 'v9c' else '（旧・併記用）'}。成功の窓 [開始−{WIN_PRE:.0f} s, CPA＋{WIN_POST:.0f} s]。方向の比較時点 = CPA の {dir_win[0]:.1f}〜{dir_win[1]:.1f} 秒前の推定方位の中央（発火時刻ではない）",
           f"- イベント {len(events)} 件（採点対象 {len([e for e in events if e['notified'] is not None])} 件）。主要評価（ablation・A〜F）{len(main_ev)} 件 = "
           f"{'前方を含む（#40・既定）' if include_front else '前方 F を除外（--exclude-front）'}・履歴不足（CPA<{min_history:.1f} s）除外・n_car≥2 除外・歩行対比（pair_id あり）除外・計画外（負例の本に偶然入った相手）除外",
           f"- 警告音の時間指標 = 遅れ（発火 − 鳴り始め）。方向 = 通知が出た瞬間の推定方位を記録紙の「鳴り始めの向き」と比べる（未発火なら鳴り始めから {warn_dir_span:.1f} s の中央・参考）", ""]
    if WARMUP_V30:                                       # §9-1 の 1: 暖機の除外の数を報告の頭に
        _wc = defaultdict(int)
        for e in events:
            if e.get("warmup_reason"):
                _wc[e["warmup_reason"]] += 1
        rep[-1:-1] = [f"- **暖機（v30・本体 §9-1 の 1・§9-4 の 1）**: 規則への入力は frame {RULE_START}（{WARMUP_S:.1f} s）以降の予測だけ（名札の記憶も）。"
                      f"除外 = 通過・発進の最接近 < {WARMUP_S:.1f} s **{_wc['warmup_cpa']} 件**・警告音の鳴り終わり ≤ {WARMUP_S:.1f} s **{_wc['warmup_end']} 件**。"
                      f"警告音の窓 = [採点上の鳴り始め − 1, ＋4]・遅れ = 発火 − 採点上の鳴り始め（= max(鳴り始め, {WARMUP_S:.1f})・物理の鳴り始めは t_start のまま）。"
                      f"発進の主の列は 最接近 − 発火・動き出しからの遅れは delay_move（events）"]
    rep.append("## 主要評価（イベント単位・分母 = 採点対象）")
    for t in ("critical", "caution", "safe", "warn"):
        if by[t][1]:
            rep.append(f"- {label[t]}: **{by[t][0]}/{by[t][1]}**（{100*by[t][0]/by[t][1]:.0f}%）")
    if extras["n_strong_on_caution"]:
        rep.append(f"- caution 車への強通知（過剰・別掲）: {extras['n_strong_on_caution']} 件")
    if extras["n_mid_on_safe"]:
        rep.append(f"- safe 車への中通知（失敗の内訳）: {extras['n_mid_on_safe']} 件")
    leads = [e["lead"] for e in main_ev if e["lead"] is not None]
    rep.append(f"- 距離クラスのリード中央値 {np.median(leads):.1f} s（範囲 {min(leads):.1f}〜{max(leads):.1f} s、正解時刻はラップ基準・知らせが出た事象の値）" if leads else "- リード: n/a")
    ev_E = [e for e in main_ev if e["gt_tier"] in ("critical", "caution")]                      # #70 r8／r12／r14: E = 評価できる至近・注意の事象全体
    n_in = sum(1 for e in ev_E if e.get("lead_first") is not None and float(e["lead_first"]) >= 2.0)
    rep.append(f"- 設定した 2 秒の基準を満たす早さ（#70）: 最接近の 2.0 秒以上前に最初の知らせ（強さを問わない）が出た事象 **{n_in}/{len(ev_E)}**"
               + (f"（{100 * n_in / len(ev_E):.0f}%・分母 = 評価できる至近・注意の事象全体・知らせなしは「間に合わなかった」・2.0 ちょうどは成功・「よけられる保証」ではない）" if ev_E else "（分母 0 → 未定義）"))
    delays = [e["delay"] for e in main_ev if e["delay"] is not None]
    rep.append(f"- 警告音の遅れ中央値 {np.median(delays):.2f} s（範囲 {min(delays):.2f}〜{max(delays):.2f} s、鳴り始めはラップ）" if delays else "- 警告音の遅れ: n/a")
    n_warn_miss = sum(1 for e in main_ev if e["gt_tier"] == "warn" and not e["notified"])
    if delays:
        rep.append(f"- 警告音の遅れの遅い順に 1 割目（90 パーセンタイル・numpy の既定）{np.percentile(delays, 90):.2f} s・負の遅れも残す・知らせが出なかった警告音 {n_warn_miss} 件（#70 r3／R108-02）")
    # 方向（2 つの分母）
    if GT_DIRWIN is None:
        dq = [e for e in main_ev if e["quadrant"]]
        n_all = len(dq); n_est = sum(1 for e in dq if e["quad_ok"] is not None); n_ok = sum(1 for e in dq if e["quad_ok"])
        rep.append(f"- 方向 4 象限一致: 全イベント分母 **{n_ok}/{n_all}**（未検出 {n_all - n_est} 件は不一致扱い）／推定があった例だけ {n_ok}/{n_est}")
    else:
        rep += gt_dirwin_lines(main_ev)
    def neg_line(name, n, sec, pass_line=False):
        h = sec / 3600.0
        if sec <= 1e-9:
            return f"- {name}: 露出 0 s（全部の区間が除かれた・率は未定義・R72-03）"
        if h <= 0:
            return f"- {name}: 露出なし"
        lo, hi = V2.poisson_rate_ci(n, h); up1 = V2.poisson_upper95_one_sided(n, h)
        s = f"- {name}: 誤警告 {n} 件 / {h:.2f} h（v4.3＋警告音 {WARN_RULE} の発火・注釈窓マスク済み）= {n/h:.2f} 回/h（両側 95% CI {lo:.2f}〜{hi:.2f}／片側 95% 上限 {up1:.2f}）"
        if pass_line:
            s += "。合格線は置かない（#70 r2・2026-10-19 本人）: 回数・正味の時間・1 時間あたり・片側 95% 上限を出すだけ。「実用になる」とは書かない"
        return s
    if hours > 0:
        rep.append(neg_line("誤警告（負例の全部）", neg["n_false"], neg["exposure_s"]))
        unk = {k: v for k, v in neg["by"].items() if v.get("n_unknown")}
        if unk:
            rep.append("- 終わり不明の長い相手（始まりから録音末尾まで除外・不明として別計上・R72-03）: " + "・".join(f"{k} {v['n_unknown']} 区間 {v['unknown_s']:.0f} s" for k, v in unk.items()))
        for grp, name, pl in (("G", "負例テイク G（仕込んだ 60 秒 × 本数・場面の確認用）", False),
                              ("負例露出", "負例露出（連続録音）", True),
                              ("その他", "負例（区分の記入なし）", False)):
            if grp in neg.get("by", {}):
                rep.append(neg_line(f"　{name}", neg["by"][grp]["n_false"], neg["by"][grp]["exposure_s"], pl))
    else:
        rep.append("- 負例露出なし")
    if GT_DIRWIN is None:
        rep += quadrant_breakdown(main_ev, side.get("walk"))          # 変更票 #57: 来る方向の内訳（補助の表）
    else:                                                            # 先行分 §5（Fable 15:2x）: 内訳も軌道の象限・記録シートの内訳は参考の別表（主と混ぜない）
        _gt = lambda evs: [dict(e, quadrant=e.get("quadrant_gt", "")) for e in (evs or [])]
        _sh = lambda evs: [dict(e, quadrant=e.get("quadrant_sheet", ""), quad_ok=e.get("quad_ok_sheet")) for e in (evs or [])]
        rep += quadrant_breakdown(_gt(main_ev), _gt(side.get("walk")),
                                  title="## 補助: 来る方向（象限）の内訳（変更票 #57・象限 = 正解の軌道・--gt-dirwin・記述だけ・検定しない）")
        rep += quadrant_breakdown(_sh(main_ev), _sh(side.get("walk")),
                                  title="## 参考: 来る方向の内訳（記録シートの象限で数えた場合・主と混ぜない・先行分 §5）")
    b2 = [e for e in events if (e.get("script_id") or "").startswith("B2") or e.get("t_move") is not None]     # #69 r23: 発進（B2）の補助の数（点数に入れない）
    lv = [e for e in b2 if e.get("t_move") is not None and e["notified"] is not None]
    sub = [e for e in b2 if e.get("t_move") is None]
    if b2:
        rep += ["", f"- 発進（B2・#69 r23）: {len(b2)} 本 = 動き出しから採点 {len(lv)} 本 ＋ 代替（動き出しの読みなし・最接近 − 6 s）{len(sub)} 本（R72-04・主表に併記・黙って混ぜない）。"
                f"動き出しから採点した本のうち、最初の知らせ（採点の窓の中）が動き出しより前だった本 {sum(1 for e in lv if e.get('before_move'))} 本（エンジンをかける音・合成に無い・記述だけ）。"
                "窓は動き出しの 1 秒前から（その 1 秒に入った始動音の知らせは成功に数える・合成と完全に同じ条件ではない限界）"]
    rep += ["", "## 参考集計（主要評価から外したもの）"]
    bd = side["boundary"]
    if bd:
        rep.append(f"- 境界例（横距離の幅が区分の境界をまたぐ・R11）: {len(bd)} 件、強・中の通知あり {sum(1 for e in bd if e['notified'])} 件（区分別の率には入れない）: "
                   + "、".join(f"{e['clip']}/{e['event_id']} {e['gt_tier'].split(':')[1]}" for e in bd[:8]))
    for key, name in (("front", "前方 F（視野内・参考）"), ("history_short", (f"暖機で除外（v30: 通過・発進の最接近 < {WARMUP_S:.1f} s／警告音の鳴り終わり ≤ {WARMUP_S:.1f} s）" if WARMUP_V30
                                                                              else f"履歴不足（CPA < {min_history:.1f} s）")), ("multi", "多重車 n_car≥2（群のいずれかへの通知）"),
                      ("unplanned", "計画外の事象（負例の本に偶然入った相手・主要評価に入れない）")):
        evs = side[key]
        if not evs:
            rep.append(f"- {name}: 0 件"); continue
        r_ = rates(evs); parts = [f"{t} {r_[t][0]}/{r_[t][1]}" for t in ("critical", "caution", "safe", "warn") if r_[t][1]]
        rep.append(f"- {name}: {len(evs)} 件（" + "、".join(parts) + "）")
    wrows, wdiffs, wamb = walk_pairs_summary(side["walk"])
    if wrows or wamb:
        r_w = rates(side["walk"])
        rep += ["", f"## 歩行対比（pair_id あり・{len(side['walk'])} 件・{len(wrows)} 対（pair×クラス）・対象が決まらず未集計 {len(wamb)} 対）— 主指標 = 検出フレーム率",
                "- 到達: " + "、".join(f"{t} {r_w[t][0]}/{r_w[t][1]}" for t in ("critical", "caution", "safe", "warn") if r_w[t][1])]
        fa = [x[3] for x in wrows if x[3] is not None]; fb = [x[4] for x in wrows if x[4] is not None]
        if fa and fb:
            rep.append(f"- 検出フレーム率 中央値: 静止 {np.median(fa):.3f} / 歩行 {np.median(fb):.3f}（対 {len(wdiffs)} 組の差 歩行−静止 中央値 {np.median(wdiffs):+.3f}）")
        # 変更票 #69 r31 ①（2026-10-23）: 分母の窓 = 相手の中点が道に沿って ±10 に入っている時刻（t_in10〜t_out10）。無い本は旧の窓（t_start〜t_cpa）で出し本数を残す
        rep.append(f"- 検出フレーム率の窓: ±10 の時刻（t_in10〜t_out10・#69 r31 ①）{extras['n_fr_in10']} 件・旧の窓（t_start〜t_cpa・時刻が無い本）{extras['n_fr_start_cpa']} 件"
                   + ("（混在 → 歩行対比の差は ±10 の本だけで読む）" if extras['n_fr_in10'] and extras['n_fr_start_cpa'] else ""))
        rep += ["", "| pair/クラス | 静止: 成功/検出率 | 歩行: 成功/検出率 |", "| --- | --- | --- |"]
        for pid, a, b, fa1, fb1 in wrows:
            fmt = lambda e, f: ("—" if e is None else f"{'○' if e['notified'] else '×'} / {f if f is not None else '—'}")
            rep.append(f"| {pid} | {fmt(a, fa1)} | {fmt(b, fb1)} |")
        for pid, cls, cnt in wamb:
            rep.append(f"| {pid}/{cls} | 未集計: 同じ状態に事象 {cnt} 件（対象が 1 件に決まらない） | |")
    if extras["n_unscored"]:
        rep.append(f"- 横距離 m 欠落で未採点: {extras['n_unscored']} 件")
    if extras["n_maskonly"]:
        rep.append(f"- scored=0（マスク専用）: {extras['n_maskonly']} 件")
    pred_b = _arg("--pred-b")
    if pred_b:
        ev_b, _, _ = evaluate(rows, load_pred7(Path(pred_b)), cfg43, dir_win, min_history, include_front, warn_dir_span)
        mb, _, _ = summarize(ev_b, include_front)
        key = lambda e: (e["clip"], e["trial"], e["event_id"], e["class"])
        da = {key(e): e for e in main_ev}; db = {key(e): e for e in mb}
        common = sorted(set(da) & set(db))
        b_ = sum(1 for k in common if da[k]["notified"] and not db[k]["notified"]); c_ = sum(1 for k in common if not da[k]["notified"] and db[k]["notified"])
        rep += ["", "## 対応あり比較（A=--pred, B=--pred-b・主要評価のみ）", f"- 成功の不一致対: A○B×={b_} / A×B○={c_} → McNemar 厳密 p={V2.mcnemar_exact(b_, c_):.4f}"]
    # 変更票 #69 r20（本人 10/18 22:56・B11 2026-10-20）: 至近の境目 1.5 m を 1.3／1.7 m にずらして区分の動きを並べる（主の結果は 1.5 m のまま・結果に関係なく出す）
    rep += ["", "## 至近の境目の感度（#69 r20・主は 1.5 m・区分の動きを見るだけ）",
            "- 住宅街（幅 4 m・歩道なし）で車が真ん中を走ると頭から車の中心までが約 1.5 m になり、歩く位置が 0.2 m ずれるだけで至近と注意が入れ替わるため。主要評価の事象（前方の扱い・履歴不足・多重車・歩行対比・計画外の除外は主と同じ）",
            "", "| 境目 | critical 本数（到達 強） | caution 本数（到達 中以上） | safe 本数（抑制） | 境界例 | caution への強 | safe への中 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for t3 in (1.3, 1.5, 1.7):
        ev_s, _, ex_s = evaluate(rows, pred, cfg43, dir_win, min_history, include_front, warn_dir_span, t3=t3)
        m_s, side_s, rates_s = summarize(ev_s, include_front)
        r_s = rates_s(m_s)
        rep.append(f"| {t3:.1f} m{'（主）' if t3 == 1.5 else ''} | {r_s['critical'][0]}/{r_s['critical'][1]} | {r_s['caution'][0]}/{r_s['caution'][1]} | "
                   f"{r_s['safe'][0]}/{r_s['safe'][1]} | {len(side_s['boundary'])} | {ex_s['n_strong_on_caution']} | {ex_s['n_mid_on_safe']} |")
    rep += ["", "| clip | event | クラス | n_car | 象限 | GT区分 | 成功 | 発火 | リード[s] | 遅れ[s] | 検出率 | 推定方位 | 象限一致 | 除外理由 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in events:
        mark = "—" if e["notified"] is None else ("○" if e["notified"] else "×")
        excl = "、".join(n for f, n in (("front", "前方"), ("history_short", "履歴不足"), ("multi", "多重車"), ("walk", "歩行対比"), ("unplanned", "計画外")) if e.get(f))
        az_s = ("%.0f°" % e["az_est"]) if e["az_est"] is not None else "—"
        rep.append(f"| {e['clip']} | {e['event_id']} | {e['class']} | {e['n_car']} | {e['quadrant'] or '—'} | {e['gt_tier']} | {mark} | {e['fired_tier'] or '—'} | "
                   f"{e['lead'] if e['lead'] is not None else '—'} | {e['delay'] if e.get('delay') is not None else '—'} | {e['frame_recall'] if e.get('frame_recall') is not None else '—'} | {az_s} | "
                   f"{'○' if e['quad_ok'] else ('×' if e['quad_ok'] is not None else '—')} | {excl or '—'} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(rep) + "\n", encoding="utf-8")
    with open(events_out, "w", newline="", encoding="utf-8") as f:
        fields = EVENT_FIELDS + (GT_DIRWIN_FIELDS if GT_DIRWIN is not None else []) + (V30_FIELDS if WARMUP_V30 else [])
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for e in events:
            w.writerow({k: ("" if e.get(k) is None else e.get(k)) for k in fields})
    print("\n".join(rep[:14])); print("->", out_md); print("->", events_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
