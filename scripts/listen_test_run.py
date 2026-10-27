# -*- coding: utf-8 -*-
"""人と比べる測定（聞き取り）の道具（2026-10-27 Opus）。scripts/listen_test_prepare.py が作った wav を順に鳴らし、気づいた時刻を記録する。

流れ（1 画面・キーだけ）:
  1. 音量の確かめ（Enter で背景を 4 秒鳴らす・ヘッドホンの音量をここで決めて、あとは変えない）
  2. 反応の速さ 6 本: 「ピッ」と鳴ったらすぐスペース（ヘッドホンと道具の遅れごと測り、あとで差し引く）
  3. 本番の聞き取り 18 本: 自転車の走る音かベルに気づいたらすぐスペース（何度押してもよい・最初の 1 回を使う）。
     鳴り終わったら、どちら側を通ったかを ←（左）・→（右）・↓（わからない）。押さなかった本は聞かない。
  Enter = 次を鳴らす・Esc = 保存して終わる（続きから再開できる）。
結果: <listen>/results_<人>.csv（block,order,file,presses_s,first_press_s,side,played_at）。1 本ごとに追記する。
使い方: ..\\DynamicSound\\.venv\\Scripts\\python.exe scripts/listen_test_run.py --reader A
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
import tkinter as tk
import wave
from pathlib import Path

E2E = Path(__file__).resolve().parents[1]
DEFAULT_DIR = E2E / "out" / "_demo_field_unused_149" / "listen"
COLS = ["block", "order", "file", "presses_s", "first_press_s", "side", "played_at"]

TEXT = {
    "intro": "人と比べる測定（デモの組）\n\nヘッドホンを付けてください。\nEnter で背景の音を 4 秒鳴らします。\nちょうど聞きやすい音量にして、そのあとは変えないでください。",
    "rt_intro": "反応の速さ（6 本）\n\n「ピッ」と鳴ったら、すぐスペース。\nEnter で始めます。",
    "main_intro": "本番の聞き取り（{n} 本）\n\n自転車の走る音か、自転車のベルに気づいたら、すぐスペース。\n確信できた瞬間で大丈夫です。何度押してもかまいません（最初の 1 回を使います）。\n鳴り終わったら、どちら側を通ったかを ←（左）・→（右）・↓（わからない）で答えます。\nEnter で始めます。",
    "ready": "{block} {k} / {n}\n\nEnter で鳴らします",
    "playing": "{block} {k} / {n}\n\n聞いています…（気づいたらスペース）",
    "side": "どちら側を通りましたか？\n\n←  左　　　→  右　　　↓  わからない",
    "done": "終わりました。ありがとうございました。\n結果: {path}\n\nEsc で閉じます",
}


def wav_seconds(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_trials(d: Path) -> list:
    return list(csv.DictReader(open(d / "trials.csv", encoding="utf-8-sig")))


def read_done(res: Path) -> set:
    if not res.is_file():
        return set()
    return {(r["block"], r["file"]) for r in csv.DictReader(open(res, encoding="utf-8-sig"))}


def append_result(res: Path, row: dict) -> None:
    new = not res.is_file()
    with open(res, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


class App:
    def __init__(self, root: tk.Tk, d: Path, reader: str):
        import winsound
        self.ws = winsound
        self.root, self.d = root, d
        self.res = d / f"results_{reader}.csv"
        done = read_done(self.res)
        tr = read_trials(d)
        self.rt = [t for t in tr if t["block"] == "rt" and ("rt", t["file"]) not in done]
        self.main = [t for t in tr if t["block"] == "main" and ("main", t["file"]) not in done]
        self.n_rt = sum(t["block"] == "rt" for t in tr)
        self.n_main = sum(t["block"] == "main" for t in tr)
        self.label = tk.Label(root, text="", font=("Meiryo", 20), justify="center", wraplength=1100)
        self.label.pack(expand=True, fill="both", padx=40, pady=40)
        root.bind("<Key>", self.on_key)
        self.state, self.cur, self.presses, self.t0 = "intro", None, [], 0.0
        self.queue = []
        self.show(TEXT["intro"])

    def show(self, s: str) -> None:
        self.label.config(text=s)

    def play(self, p: Path) -> None:
        self.ws.PlaySound(None, self.ws.SND_PURGE)
        self.t0 = time.perf_counter()
        self.ws.PlaySound(str(p), self.ws.SND_FILENAME | self.ws.SND_ASYNC)

    def next_ready(self) -> None:
        if not self.queue:
            if self.state_block == "rt":
                self.state_block = "main"
                self.queue = list(self.main)
                self.state = "main_intro"
                self.show(TEXT["main_intro"].format(n=self.n_main))
                return
            self.state = "done"
            self.show(TEXT["done"].format(path=self.res))
            return
        self.cur = self.queue.pop(0)
        n = self.n_rt if self.state_block == "rt" else self.n_main
        self.state = "ready"
        self.show(TEXT["ready"].format(block="反応の速さ" if self.state_block == "rt" else "聞き取り", k=self.cur["order"], n=n))

    def start_trial(self) -> None:
        p = self.d / self.cur["file"]
        self.presses = []
        n = self.n_rt if self.state_block == "rt" else self.n_main
        self.state = "playing"
        self.show(TEXT["playing"].format(block="反応の速さ" if self.state_block == "rt" else "聞き取り", k=self.cur["order"], n=n))
        self.play(p)
        self.root.after(int((wav_seconds(p) + 0.4) * 1000), self.end_trial)

    def end_trial(self) -> None:
        if self.state != "playing":
            return
        if self.state_block == "main" and self.presses:
            self.state = "side"
            self.show(TEXT["side"])
            return
        self.save("")

    def save(self, side: str) -> None:
        c = self.cur
        append_result(self.res, {"block": c["block"], "order": c["order"], "file": c["file"],
                                 "presses_s": ";".join(f"{x:.3f}" for x in self.presses),
                                 "first_press_s": (f"{self.presses[0]:.3f}" if self.presses else ""), "side": side,
                                 "played_at": datetime.datetime.now().isoformat(timespec="seconds")})
        self.next_ready()

    def on_key(self, e) -> None:
        k = e.keysym
        if k == "Escape":
            self.ws.PlaySound(None, self.ws.SND_PURGE)
            self.root.destroy()
            return
        if self.state == "intro" and k == "Return":
            self.play(self.d / "volume.wav")
            self.state = "volume"
            self.show("背景の音（4 秒）\n\n音量が決まったら Enter（もう一度聞くなら Space）")
        elif self.state == "volume" and k == "space":
            self.play(self.d / "volume.wav")
        elif self.state == "volume" and k == "Return":
            self.ws.PlaySound(None, self.ws.SND_PURGE)
            if self.rt:
                self.state_block, self.queue = "rt", list(self.rt)
                self.state = "rt_intro"
                self.show(TEXT["rt_intro"])
            else:
                self.state_block, self.queue = "main", list(self.main)
                self.state = "main_intro"
                self.show(TEXT["main_intro"].format(n=self.n_main))
        elif self.state in ("rt_intro", "main_intro") and k == "Return":
            self.next_ready()
        elif self.state == "ready" and k == "Return":
            self.start_trial()
        elif self.state == "playing" and k == "space":
            self.presses.append(time.perf_counter() - self.t0)
            if len(self.presses) == 1:
                self.label.config(text=self.label.cget("text") + "\n\n● 押しました")
        elif self.state == "side" and k in ("Left", "Right", "Down"):
            self.save({"Left": "L", "Right": "R", "Down": "?"}[k])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="A")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    a = ap.parse_args(argv)
    if not (a.dir / "trials.csv").is_file():
        print(f"先に scripts/listen_test_prepare.py を（{a.dir} に trials.csv が無い）"); return 2
    root = tk.Tk()
    root.title("人と比べる測定")
    root.geometry("1200x700")
    App(root, a.dir, a.reader)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
