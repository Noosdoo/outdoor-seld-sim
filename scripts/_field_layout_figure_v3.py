# -*- coding: utf-8 -*-
"""実録の線とテープの配置図 v3（2026-09-28・変更票 #38: 学内 195 本・待ち受けなし・右側通過は三脚 180° 回転の案）。
v2（9/24・左 6 本＋右 4 本＋8/12 m）は _field_layout_figure_v2.py と out/field_layout_2026-09-24/ に残す。
上から見た図（2 段）: 上 = 全体（左 6 本と目印・カメラ・走る向き）、下 = マイク近くの拡大（線ごとに誰が使うか）。
マイクを原点、x = 前（走る向き）、y = 左。線は x に平行に y = 1.0/1.5/2.0/2.5/3.0/5.0（左 6 本）。8/12 m は無し。
右側通過は三脚を 180° 回して左の線を使う案（新台本 §5・本人の決定待ち。採らないなら右 1.0/1.5/2.0/3.0 を引く）。
出力: out/field_layout_2026-09-28/lines_layout.png
使い方: python scripts/_field_layout_figure_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/field_layout_2026-09-28"
OUT.mkdir(parents=True, exist_ok=True)
for cand in ("Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP"):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

RED, BLUE, GREEN, GRAY = "#c8102e", "#1f5fa8", "#2a7f3f", "#555"
LINES_L = [(1.0, "1.0 m  至近圏: 自転車 C-01〜04（ベル）・キック F-01〜12"),
           (1.5, "1.5 m  至近圏の端（採点の境界）: キック F-13〜16（幅 1.4-1.6 で記録・副表）"),
           (2.0, "2.0 m  注意圏: 車 A-01〜04・発進 B-01〜05・キック F-17〜20・歩く比較のキック P41〜50"),
           (2.5, "2.5 m  注意圏: 車 A-05〜08"),
           (3.0, "3.0 m  注意圏（境界 3.2 まで 0.2 m）: 車 A-09〜12・自転車 C-05〜08・キック F-21〜22・歩く比較の車 P01〜20・自転車 P31〜40"),
           (5.0, "5.0 m  安全圏: 車 A-13〜22・キック F-23〜24・歩く比較の車 P21〜30")]
MARKS = [-10, -5, 0, 5, 10]


def lines_and_marks(ax, ys, tick=0.25, lw=2.2, color=RED, ls="-"):
    for y in ys:
        ax.plot([-14, 14], [y, y], color=color, lw=lw, ls=ls, zorder=2)
        for x in MARKS:
            ax.plot([x, x], [y - tick, y + tick], color="#222", lw=2.4, zorder=3)


def panel_overview(ax):
    ax.set_title("① 全体（上から見た図）— 線は道の向き（x）に平行、マイクからの横距離（y）が線の値。左 6 本だけ", fontsize=11, loc="left")
    ax.axhspan(-2.0, 7.0, color="#f3f3f3", zorder=0)
    lines_and_marks(ax, [y for y, _ in LINES_L])
    ax.plot([-14, 14], [0, 0], color=BLUE, lw=1.6, ls=":", zorder=2)
    for x in MARKS:
        ax.plot([x, x], [-0.25, 0.25], color=BLUE, lw=2.0, zorder=3)
    ax.plot(0, 0, marker="^", ms=12, color=BLUE, zorder=5)
    ax.text(0.5, -0.9, "マイク（三脚・原点・高さ約 1.6 m）", fontsize=9, color=BLUE)
    for y, txt, ty in [(5, "5.0 m 安全圏", 5.3), (3.0, "3.0 m 注意圏の上側", 3.5), (2.5, "2.5 m 注意圏", 2.6), (2.0, "2.0 m 注意圏", 1.9),
                       (1.5, "1.5 m 至近圏の端", 1.2), (1.0, "1.0 m 至近圏", 0.6), (0, "0 m マイクの線（歩く比較で本人が歩く）", -0.1)]:
        ax.annotate(txt, xy=(14.2, y), xytext=(16.0, ty), fontsize=8.5, va="center",
                    color=BLUE if y == 0 else "#222", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.plot(-8, 6.0, marker="s", ms=9, color=GREEN, zorder=5)
    ax.text(-7.4, 6.0, "カメラ（地上の三脚・高さ 1.5〜2 m・線の斜め横）", fontsize=8.5, color=GREEN, va="center")
    ax.annotate("", xy=(-10, 6.6), xytext=(-14, 6.6), arrowprops=dict(arrowstyle="->", lw=1.8, color=GRAY))
    ax.text(-14, 6.9, "車は車体の中心で線をまたぐ（ボンネット中央の印）。二輪・キックは前輪を線の上に。x = +14 → −14（前→後）または逆（後→前）", fontsize=8.5, color=GRAY)
    ax.annotate("", xy=(14, -1.5), xytext=(19.8, -1.5), arrowprops=dict(arrowstyle="<-", lw=1.4, color="#888"))
    ax.text(14.2, -1.9, "助走 約 5.8 m（15 km/h まで）", fontsize=8, color="#666")
    ax.annotate("", xy=(-19.5, -1.5), xytext=(-14, -1.5), arrowprops=dict(arrowstyle="->", lw=1.4, color="#888"))
    ax.text(-19.5, -1.9, "停止 約 3.5 m", fontsize=8, color="#666")
    ax.text(-19.5, -1.3, "ブザー D-01〜08 はマイクから 2 m・クラクション E-01〜08 は 3 m（前・左・後・右・線なし・巻尺で測る）", fontsize=8.5, color="#333")
    ax.set_xlim(-20.5, 38); ax.set_ylim(-2.2, 7.4); ax.set_aspect("equal")
    ax.set_xlabel("x [m]（前 = ＋）"); ax.set_ylabel("y [m]（左 = ＋）"); ax.grid(True, color="#ddd", lw=0.5)


def panel_zoom(ax):
    ax.set_title("② マイク近くの拡大 — 左 6 本と、各線の目印（x = −10/−5/0/+5/+10 に 50 cm の横テープ）・線ごとに誰が使うか", fontsize=11, loc="left")
    ax.axhspan(-0.8, 5.8, color="#f3f3f3", zorder=0)
    lines_and_marks(ax, [y for y, _ in LINES_L], tick=0.12)
    ax.plot([-14, 14], [0, 0], color=BLUE, lw=1.6, ls=":", zorder=2)
    for x in MARKS:
        ax.plot([x, x], [-0.12, 0.12], color=BLUE, lw=2.0, zorder=3)
        ax.text(x, -0.3, f"x = {x:+d}" if x else "x = 0", ha="center", va="top", fontsize=8.5, color="#333")
    ax.plot(0, 0, marker="^", ms=12, color=BLUE, zorder=5)
    ax.text(0.4, 0.25, "マイク（三脚・原点）", fontsize=9, color=BLUE)
    labels = LINES_L + [(0.0, "0 m  マイクの線（歩く比較で本人が y = 0 を歩く）")]
    ty = {0.0: 0.1, 1.0: 0.9, 1.5: 1.6, 2.0: 2.3, 2.5: 3.0, 3.0: 3.8, 5.0: 5.0}
    for y, txt in sorted(labels, key=lambda t: t[0]):
        ax.annotate(txt, xy=(14.2, y), xytext=(15.6, ty[y]), fontsize=8.6, va="center", color=BLUE if y == 0 else "#222",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.plot(-8, 4.6, marker="s", ms=9, color=GREEN, zorder=5)
    ax.text(-7.4, 4.6, "カメラ（地上の三脚）。画角に 線・目印・マイク が全部入る", fontsize=8.5, color=GREEN, va="center")
    ax.set_xlim(-15, 40); ax.set_ylim(-0.8, 5.8); ax.set_aspect("equal")
    ax.set_xlabel("x [m]（前 = ＋）"); ax.set_ylabel("y [m]（左 = ＋）"); ax.grid(True, color="#ddd", lw=0.5)


def main() -> int:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), dpi=130, gridspec_kw={"height_ratios": [1.0, 1.0]})
    panel_overview(ax1)
    panel_zoom(ax2)
    fig.suptitle("実録の線とテープの配置 v3（2026-09-28 変更票 #38: 学内 195 本・左 6 本だけ・8/12 m の待ち受けなし）", fontsize=13)
    fig.text(0.01, 0.005,
             "テープ = 幅 5 cm の養生テープ（ロープ・巻尺は他の車両が通るので不可）。線は道の向きに平行に 28 m（x = −14〜+14）。線の横距離は巻尺でマイクから測る。"
             "右側通過（自転車 C-03/04/07/08・キック F-02/04/14/16/18/20・歩く比較の右）は三脚を 180° 回して左の線を使う案（本人の決定待ち。採らないなら右 1.0/1.5/2.0/3.0 m も引く）。"
             "回したら指パッチンの 4 方向の点検をやり直す。",
             fontsize=8, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    p = OUT / "lines_layout.png"
    fig.savefig(p)
    print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
