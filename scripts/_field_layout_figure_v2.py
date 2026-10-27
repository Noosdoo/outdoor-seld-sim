# -*- coding: utf-8 -*-
"""実録の線とテープの配置図 v2（2026-09-24・線の値を 0.5 m 刻みの 6 本に改定・本人 20:01「線の値決めよう」台本 §8 4f）。
v1（2026-09-23・7 本）は _field_layout_figure.py と out/field_layout_2026-09-23/ に残す。
上から見た図（2 段）: 上 = 全体（8／12 m の遠い線まで）、下 = マイク近くの拡大（1.0〜5.0 m の線と目印）。
マイクを原点、x = 前（走る向き）、y = 左。線は x に平行に y = 1.0/1.5/2.0/2.5/3.0/5.0（左）、1.0/1.5/2.0（右・キック）、3.0（右・歩行対比の車）、8/12（待ち受け・線なし）。
目印は各線の x = −10/−5/0/+5/+10 に短い横テープ。カメラは地上の三脚。テープで引く（ロープ・巻尺は不可・本人 9/24 12:48）。
出力: out/field_layout_2026-09-24/lines_layout.png
使い方: python scripts/_field_layout_figure_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/field_layout_2026-09-24"
OUT.mkdir(parents=True, exist_ok=True)
for cand in ("Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

RED, BLUE, GREEN, GRAY = "#c8102e", "#1f5fa8", "#2a7f3f", "#555"
LINES_L = [(1.0, "1.0 m  至近: 自転車 B-11〜15・キック E-01〜04, E-13〜20"),
           (1.5, "1.5 m  至近の境界: キック E-09〜12・歩行対比のキック P41〜50（幅 1.4-1.6 で記録・副表）"),
           (2.0, "2.0 m  注意: 車 A-01〜04・EV B-01〜05・発進 B-06〜10・キック E-05〜08"),
           (2.5, "2.5 m  注意: 車 A-05〜07"),
           (3.0, "3.0 m  注意の上側（境界 3.2 まで 0.2 m）: 車 A-08〜10・歩行対比の車 P01〜10・自転車 P31〜40"),
           (5.0, "5.0 m  安全: 車 A-11〜14・歩行対比の車 P21〜30")]
LINES_R = [(-1.0, "右 1.0 m  キックの右側通過（E-02, E-04）"), (-1.5, "右 1.5 m  キックの右側通過（E-10, E-12）"),
           (-2.0, "右 2.0 m  キックの右側通過（E-06, E-08）"), (-3.0, "右 3.0 m  歩行対比の車 P11〜20")]
FAR = [(8.0, "8 m  安全帯の遠い側（待ち受け・線なし。通過位置は巻尺かレーザーで 1 回測る）"), (12.0, "12 m  同上")]
MARKS = [-10, -5, 0, 5, 10]


def lines_and_marks(ax, ys, tick=0.25, lw=2.2, color=RED, ls="-"):
    for y in ys:
        ax.plot([-14, 14], [y, y], color=color, lw=lw, ls=ls, zorder=2)
        for x in MARKS:
            ax.plot([x, x], [y - tick, y + tick], color="#222", lw=2.4, zorder=3)


def panel_overview(ax):
    ax.set_title("① 全体（上から見た図）— 線は道の向き（x）に平行、マイクからの横距離（y）が線の値", fontsize=11, loc="left")
    ax.axhspan(-3.6, 13.2, color="#f3f3f3", zorder=0)
    lines_and_marks(ax, [y for y, _ in LINES_L + LINES_R])
    for y, _ in FAR:
        ax.plot([-14, 14], [y, y], color=RED, lw=1.4, ls=(0, (6, 4)), zorder=2)
    ax.plot([-14, 14], [0, 0], color=BLUE, lw=1.6, ls=":", zorder=2)
    for x in MARKS:
        ax.plot([x, x], [-0.25, 0.25], color=BLUE, lw=2.0, zorder=3)
    ax.plot(0, 0, marker="^", ms=12, color=BLUE, zorder=5)
    ax.text(0.5, -0.9, "マイク（三脚・原点）", fontsize=9, color=BLUE)
    for y, txt, ty in [(12, FAR[1][1], 12), (8, FAR[0][1], 8), (5, "5.0 m 安全", 5.6), (3.0, "3.0 m 注意の上側", 3.7),
                       (2.5, "2.5 / 2.0 / 1.5 / 1.0 m（下の拡大図）", 1.9), (0, "0 m マイクの線（歩行対比で本人が歩く）", 0.3),
                       (-1.0, "右 1.0 / 1.5 / 2.0 m キックの右側通過", -1.5), (-3.0, "右 3.0 m 歩行対比の車", -3.1)]:
        ax.annotate(txt, xy=(14.2, y), xytext=(16.0, ty), fontsize=8.5, va="center",
                    color=BLUE if y == 0 else "#222", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.plot(-8, 6.0, marker="s", ms=9, color=GREEN, zorder=5)
    ax.text(-7.4, 6.0, "カメラ（地上の三脚・高さ 1.5〜2 m・線の斜め横）", fontsize=8.5, color=GREEN, va="center")
    ax.annotate("", xy=(-10, 10.5), xytext=(-14, 10.5), arrowprops=dict(arrowstyle="->", lw=1.8, color=GRAY))
    ax.text(-14, 11.0, "車は車体の中心で線をまたぐ（ボンネット中央の印）。二輪・キックは前輪を線の上に。x = +14 → −14（前→後）または逆", fontsize=8.5, color=GRAY)
    ax.annotate("", xy=(14, -3.1), xytext=(19.8, -3.1), arrowprops=dict(arrowstyle="<-", lw=1.4, color="#888"))
    ax.text(14.2, -3.5, "助走 約 5.8 m（15 km/h まで）", fontsize=8, color="#666")
    ax.annotate("", xy=(-19.5, -3.1), xytext=(-14, -3.1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#888"))
    ax.text(-19.5, -3.5, "停止 約 3.5 m", fontsize=8, color="#666")
    ax.set_xlim(-20.5, 38); ax.set_ylim(-4.0, 13.2); ax.set_aspect("equal")
    ax.set_xlabel("x [m]（前 = ＋）"); ax.set_ylabel("y [m]（左 = ＋）"); ax.grid(True, color="#ddd", lw=0.5)


def panel_zoom(ax):
    ax.set_title("② マイク近くの拡大 — 左 6 本・右 4 本と、各線の目印（x = −10/−5/0/+5/+10 に 50 cm の横テープ）", fontsize=11, loc="left")
    ax.axhspan(-3.5, 5.8, color="#f3f3f3", zorder=0)
    lines_and_marks(ax, [y for y, _ in LINES_L + LINES_R], tick=0.12)
    ax.plot([-14, 14], [0, 0], color=BLUE, lw=1.6, ls=":", zorder=2)
    for x in MARKS:
        ax.plot([x, x], [-0.12, 0.12], color=BLUE, lw=2.0, zorder=3)
        ax.text(x, -0.4, f"x = {x:+d}" if x else "x = 0", ha="center", va="top", fontsize=8.5, color="#333")
    ax.plot(0, 0, marker="^", ms=12, color=BLUE, zorder=5)
    ax.text(0.4, 0.25, "マイク（三脚・原点・高さ 1.5〜2 m）", fontsize=9, color=BLUE)
    labels = LINES_L + LINES_R + [(0.0, "0 m  マイクの線（歩行対比で本人が y = 0 を歩く）")]
    order = sorted(labels, key=lambda t: t[0])
    ty = {-3.0: -3.1, -2.0: -2.3, -1.5: -1.5, -1.0: -0.7, 0.0: 0.1, 1.0: 0.9, 1.5: 1.7, 2.0: 2.5, 2.5: 3.3, 3.0: 4.1, 5.0: 5.0}
    for y, txt in order:
        ax.annotate(txt, xy=(14.2, y), xytext=(15.6, ty[y]), fontsize=8.6, va="center", color=BLUE if y == 0 else "#222",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.plot(-8, 4.6, marker="s", ms=9, color=GREEN, zorder=5)
    ax.text(-7.4, 4.6, "カメラ（地上の三脚）。画角に 線・目印・マイク が全部入る", fontsize=8.5, color=GREEN, va="center")
    ax.set_xlim(-15, 40); ax.set_ylim(-3.5, 5.8); ax.set_aspect("equal")
    ax.set_xlabel("x [m]（前 = ＋）"); ax.set_ylabel("y [m]（左 = ＋）"); ax.grid(True, color="#ddd", lw=0.5)


def main() -> int:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12.5), dpi=130, gridspec_kw={"height_ratios": [1.3, 1.0]})
    panel_overview(ax1)
    panel_zoom(ax2)
    fig.suptitle("実録の線とテープの配置 v2（2026-09-24 本人決定: 0.5 m 刻み・左 6 本＋右 4 本・8/12 m は待ち受け）", fontsize=13)
    fig.text(0.01, 0.005,
             "テープ = 幅 5 cm の養生テープ（ロープ・巻尺は他の車両が通るので不可）。線は道の向きに平行に 28 m（x = −14〜+14）引く。線の横距離は巻尺でマイクから測る。"
             "各線の x = −10/−5/0/+5/+10 に 50 cm の横テープ（目印）。3.2 m の線は引かない（採点の境界そのもの）。1.5 m 線のキックは幅 1.4-1.6 で記録して境界例の副表へ。",
             fontsize=8, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    p = OUT / "lines_layout.png"
    fig.savefig(p)
    print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
