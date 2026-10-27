# -*- coding: utf-8 -*-
"""動画のコマ送り＋キー 1 つで「拍手」「目印の通過」の時刻を記録する道具（2026-09-26・Fable）。
本人決定（2026-09-24 17:29）「正解の時刻は人（本人）が動画で読む。AI・画像認識には読ませない」→ 読む人の手間だけ減らす。
仕様 = `md/handoff/実録_現在地と未決_2026-09-24.md` §5-4。

  - 動画（スマホ 30 fps 前後・MOV/MP4）をコマ送りで表示（←/→ = 1 コマ、Shift+←/→ = 10 コマ、↑/↓ = 1 秒、Space = 再生/停止）
  - キー 1 つで記録: c = 拍手（最初）、v = 拍手（最後・任意）、1〜9・0・- = 11 点の目印（進行方向の順。画面に「1: x=+10 …」と出る）、m = 次の目印、
    z = 拡大（1 → 3 → 6 倍・拡大中は元の 4K の画素のまま・2026-10-22）、マウスのクリック = その場所を中心に拡大、
    r = この本の目印の順を逆にする（1 番目のキー = 最初に横切る目印・2026-10-22）、
    h = 隠れ始め・j = 見え始め（足元や前輪が隠れた区間・B4 R17-01・2026-10-20: 1 秒以内なら前後の読みから線で埋める・超えたら欠測。出力の hidden 列 start／end・x は空）、u = 取り消し
  - n / p = 次 / 前のテイク、s = 保存、q = 保存して終了。途中で閉じても状態ファイル（.state.json）から続きを再開できる
  - 出力 = step19f の入力 CSV（take,frame,cls,track,x,y,t）。t = 音の時刻 [s] = 動画の時刻 − 拍手_動画側秒 + 拍手_録音側秒（takes CSV の clap_audio_s・
    本人が波形で読んだ値〔events.csv の 拍手_録音側秒・#59 r7〕だけ。step19j の自動の値は step19d が候補の列 clap_candidate_s に入れるだけで同期には使わない＝NT12-01。記録シートのボタンの秒 拍手_音側秒 は使わない・変更票 #59）（実数。step19f は t があれば 0.1 s に丸めない）
    ＋ 同期の CSV（take,clap,audio_s,video_s・step19g の sync 用。2 回目の拍手の音側秒は takes CSV の clap2_audio_s があれば）

takes CSV（記録シートから作る）の列:
  take        テイク名（出力の take 列・例 20261005_A1_t03）
  video       動画ファイルの場所（絶対パスか takes CSV からの相対）
  cls         car / kick / bike_bell / bike など（step19f と同じ名前）
  track       同じ相手の番号（省略時 1）
  line_m      線の値 [m]＝現場の y（東が正・西が負）。右側の線は三脚を回して作る（#44・#69 r4）
  direction   前→後 か 後→前（目印の順を決める。前→後 = x が +10 → −10）。省略時 前→後
  clap_audio_s 拍手_録音側秒（波形で確認した原本秒。ストップウォッチ値は代用しない）
  sync_group   任意。同じ原録音（WAV）と同じ拍手を共有する本の印。無ければ take 名（step19d が出す take は原本の WAV 名）。
               1 本の録音に通過が何回もある測定（take 名が通過ごとに違う）は、全行に同じ値（例 原本の WAV 名）を入れる。
  clap2_audio_s 2 回目の拍手の音側秒（任意・同期試験用）
  markers     任意。目印の x を進行方向の順にコンマ区切り（既定 "10,8,6,4,2,0,-2,-4,-6,-8,-10" を direction で並べ替え）
  src_z       任意（step19f の --src-z 用。出力の z 列に写す）
  備考        任意（2026-10-26）。あれば画面の一番上に【備考: …】と出す（読むときの注意。出力には入らない）

使い方:
  python scripts/step19h_video_marks.py --takes <takes.csv> --out <出力dir> --reader A [--size 960x540]
      → <出力dir>/reads_A.csv（step19f の入力）、<出力dir>/sync_A.csv、<出力dir>/reads_A.state.json（再開用）
  python scripts/step19h_video_marks.py --takes <takes.csv> --out <出力dir> --reader A --export   # 画面を開かず、状態ファイルから reads/sync を書き出す（2026-10-12・#59 r6。
      夜に読んだあとで clap_audio_s を埋めた本も、読み直さずに出せる。拍手の秒が無い本は出さず一覧に出す）
  python scripts/step19h_video_marks.py --takes <takes.csv> --out <出力dir> --reader A --audio-dir raw   # 動画の下に録音の波形（W）を出す（2026-10-22・本人「元のウィンドウで完結」）:
      上の細い帯 = 録音全体（クリック = その場所へ）・下 = 拡大した範囲（クリック = カーソル）・波形の上でホイール か i／o = 拡大／縮小・
      , ／ . = カーソルを 0.5 ms（Shift 5 ms）・a = カーソルを録音の拍手に・b = ベルの鳴り始めに・e = ベルの鳴り終わりに（2026-10-27・最後に鳴り終わった所）・
      d = この録音の読みを消す・Home = カーソルを真ん中に・
      l = カーソル（無ければ動画のいまのコマ）から 1.5 秒聞く・L = 表示している範囲を聞く（最大 8 秒・聞いた範囲を黄色で示す）・Esc = 止める。
      動画の拍手（c）と録音の拍手（a）がそろうと、動画のいまのコマの録音の時刻（オレンジ）を真ん中に追いかける。
      読みは <出力dir>/audio_reads_<人>.json と .csv に残り、録音の拍手はこの本の clap_audio_s（拍手_録音側秒）として同期に使う（#59 r7）。自動の候補（step19j）は出さない（NT12-01）。
  python scripts/step19h_video_marks.py --selftest <一時dir>   # 合成の動画で、時刻の読みと出力の形を確かめる（画面なし）
  その後: python scripts/step19f_video_to_gt.py <出力dir>/reads_A.csv <正解dir> --mic-z 1.6
          python scripts/step19g_pilot_uncertainty.py readers reads_A.csv reads_B.csv <出力dir>   # 2 人の読みの一致
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

LABEL_RES = 0.1
MARKER_KEYS = tuple("1234567890") + ("minus",)
DEFAULT_MARKERS = [float(x) for x in range(10, -11, -2)]     # 前→後（x が減る向き）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 動画
ZOOM_STEPS = (1, 3, 6)   # 2026-10-22 本人「画質落ちてませんか」: 4K を窓に縮めると遠い線のテープが 1〜2 画素になる → z とクリックで元の画素のまま拡大（表示だけ・読みの記録は変えない）


def zoom_crop(w: int, h: int, zoom: float, cx: float, cy: float):
    """元の絵（w×h）から、(cx, cy) を中心に 1/zoom の大きさを切り出す範囲 (x0, y0, cw, ch)。端では内側に寄せる。zoom ≤ 1 は全体。"""
    if zoom <= 1:
        return 0, 0, w, h
    cw, ch = max(1, int(round(w / zoom))), max(1, int(round(h / zoom)))
    x0 = int(round(min(max(cx - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), h - ch)))
    return x0, y0, cw, ch


class Video:
    """cv2 で 1 コマずつ読む。時刻は容器のタイムスタンプ（CAP_PROP_POS_MSEC・可変フレームレートでも正しい）を優先し、無ければ 番号/fps。"""

    def __init__(self, path: Path):
        import cv2
        self.cv2 = cv2
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"動画を開けない: {self.path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.cur = -1
        self._cache: dict[int, tuple] = {}
        self._order: list[int] = []

    def get(self, idx: int):
        """(idx, 画像 BGR ndarray, 動画の時刻 [s], 時刻の出どころ) を返す。範囲外は端に丸める。"""
        idx = max(0, min(idx, self.n - 1))
        if idx in self._cache:
            return self._cache[idx]
        cv2 = self.cv2
        if idx != self.cur + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"コマ {idx} が読めない（{self.path.name}）")
        msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        # cv2 の POS_MSEC は「読んだコマの次」の位置を返す実装があるので、1 コマ分戻して読んだコマの時刻にする
        t_pts = (msec / 1000.0) - (1.0 / self.fps)
        if not (t_pts >= -1e-6):
            t_pts = idx / self.fps; src = "idx/fps"
        else:
            src = "pts"
        self.cur = idx
        rec = (idx, img, max(0.0, t_pts), src)
        self._cache[idx] = rec; self._order.append(idx)
        if len(self._order) > 90:
            old = self._order.pop(0); self._cache.pop(old, None)
        return rec

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------- 記録（画面なしで検査できる部分）
def markers_for(direction: str, custom: str | None) -> list[float]:
    if custom:
        return [float(v) for v in custom.split(",") if v.strip()]
    m = list(DEFAULT_MARKERS)
    if "後" in (direction or "") and (direction or "").index("後") < (direction or "").find("前"):   # 後→前 = x が増える向き
        m = sorted(m)
    return m


def wav_envelope(x, n_cols: int):
    """波形 x（1 次元）を n_cols 列に分け、列ごとの（最小, 最大）を返す（描画用・2026-10-22）。"""
    import numpy as np
    n_cols = max(1, int(n_cols)); out = np.zeros((n_cols, 2), dtype=np.float32); n = len(x)
    if n == 0:
        return out
    edges = np.linspace(0, n, n_cols + 1).astype(int)
    for i in range(n_cols):
        a = min(edges[i], n - 1); b = max(a + 1, edges[i + 1])
        seg = x[a:b]
        out[i] = (seg.min(), seg.max())
    return out


def make_play_wav(x, sr: int, t0: float, t1: float, path: Path, max_s: float = 8.0, max_gain: float = 50.0) -> tuple[float, float]:
    """波形 x の [t0, t1)（最大 max_s 秒）を、聞くための 16 bit の wav にする（96 kHz は 2 つずつ平均して 48 kHz に）。
    いちばん大きい所が 0.5 になるように持ち上げる（ただし max_gain 倍まで＝背景だけの所を大きくしすぎない）。返り値 = 実際に書いた (t0, t1)。"""
    import numpy as np
    import wave as _wave
    t0 = max(0.0, float(t0)); t1 = min(float(t1), len(x) / sr, t0 + max_s)
    seg = np.asarray(x[int(t0 * sr):int(t1 * sr)], dtype=np.float32)
    if sr >= 88200:
        seg = seg[:len(seg) // 2 * 2].reshape(-1, 2).mean(axis=1); sr2 = sr // 2
    else:
        sr2 = sr
    pk = float(np.abs(seg).max()) if len(seg) else 0.0
    g = min(0.5 / pk, max_gain) if pk > 0 else 1.0
    pcm = (np.clip(seg * g, -1, 1) * 32767).astype("<i2").tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr2)); w.writeframes(pcm)
    return t0, t1


class AudioReads:
    """録音ごとの本人の読み（拍手の立ち上がり clap_audio_s・ベルの鳴り始め bell_onset_s・ベルの鳴り終わり bell_end_s）。<出力dir>/audio_reads_<人>.json と .csv（2026-10-22）。
    bell_end_s（2026-10-27・本人 12:35 (c)）: 実録の警告音の正解の区間 = 波形で読んだ鳴り始め〜鳴り終わり（何回か鳴らした本は最初の鳴り始め〜最後の鳴り終わり）。"""

    KINDS = ("clap_audio_s", "bell_onset_s", "bell_end_s")

    def __init__(self, out_dir: Path, reader: str):
        self.out_dir = Path(out_dir); self.p_json = self.out_dir / f"audio_reads_{reader}.json"; self.p_csv = self.out_dir / f"audio_reads_{reader}.csv"
        self.d = json.loads(self.p_json.read_text(encoding="utf-8")) if self.p_json.exists() else {}

    def set(self, wav: str, kind: str, sec: float):
        assert kind in self.KINDS
        self.d.setdefault(wav, {})[kind] = round(float(sec), 4); self.save()

    def clear(self, wav: str):
        self.d.pop(wav, None); self.save()

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.p_json.write_text(json.dumps(self.d, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(self.p_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["wav", *self.KINDS])
            for wav in sorted(self.d):
                w.writerow([wav, *[("%.4f" % self.d[wav][k]) if k in self.d[wav] else "" for k in self.KINDS]])

    def apply(self, takes: list[dict]):
        """読んだ録音の拍手を、同じ録音（take = WAV 名）の行の clap_audio_s に入れる（同期に使う・#59 r7）。"""
        for t in takes:
            v = self.d.get(t["take"], {}).get("clap_audio_s")
            if v is not None:
                t["clap_audio_s"] = "%.4f" % v


def load_takes(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v or "").strip() for k, v in r.items()}
            if not r.get("take"):
                continue
            v = Path(r["video"])
            if not v.is_absolute():
                v = (path.parent / v).resolve()
            r["video_path"] = str(v)
            r["track"] = r.get("track") or "1"
            r["markers_list"] = markers_for(r.get("direction", "前→後"), r.get("markers"))
            rows.append(r)
    if not rows:
        raise ValueError(f"takes CSV に行が無い: {path}")
    return rows


class Session:
    """テイクごとの記録。state = {take: {"clap": {"frame","t_video"}, "clap2": {...}, "marks": [{"x","frame","t_video"}...], "done": bool}}"""

    def __init__(self, takes: list[dict], state_path: Path, reader: str):
        self.takes = takes
        self.state_path = state_path
        self.reader = reader
        self.state = {}
        if state_path.exists():
            self.state = json.loads(state_path.read_text(encoding="utf-8")).get("takes", {})
        keys = [self.key(t) for t in takes]
        if len(set(keys)) != len(keys):
            raise ValueError("同じ take・cls・track の設定行が重複")
        for t in takes:
            key = self.key(t)
            if t["take"] in self.state:
                if sum(x["take"] == t["take"] for x in takes) > 1:
                    raise ValueError(f"{t['take']}: 旧保存は相手と mic の読みを区別できません。旧 state を別名で保存し、別々に読み直してください")
                self.state[key] = self.state.pop(t["take"])
            self.state.setdefault(key, {"clap": None, "clap2": None, "marks": [], "done": False, "history": []})
            if self.state[key].get("rev"):                                   # r で逆にした本（2026-10-22）
                t["markers_list"] = list(reversed(t["markers_list"]))

    def key(self, take):
        if isinstance(take, str):
            matches = [t for t in self.takes if t["take"] == take]
            if len(matches) != 1:
                raise ValueError("take 名だけでは読みの対象を特定できません。設定行を渡してください")
            take = matches[0]
        return json.dumps([take["take"], take["cls"], str(take.get("track") or "1")], ensure_ascii=False)

    def rec(self, take: str, kind: str, frame: int, t_video: float, x: float | None = None):
        s = self.state[self.key(take)]
        if kind in ("clap", "clap2"):
            s["history"].append((kind, s[kind]))
            s[kind] = {"frame": frame, "t_video": t_video}
        elif kind in ("hide", "show"):                                       # B4: 隠れ始め／見え始め
            s.setdefault("hidden", []).append({"kind": kind, "frame": frame, "t_video": t_video})
            s["hidden"].sort(key=lambda m: m["t_video"]); s["history"].append(("hidden", kind))
        else:
            s["marks"] = [m for m in s["marks"] if m["x"] != x]          # 同じ目印を読み直したら置き換える
            s["marks"].append({"x": x, "frame": frame, "t_video": t_video})
            s["marks"].sort(key=lambda m: m["t_video"])
            s["history"].append(("mark", x))

    def undo(self, take: str):
        s = self.state[self.key(take)]
        if not s["history"]:
            return
        kind, val = s["history"].pop()
        if kind in ("clap", "clap2"):
            s[kind] = val
        elif kind == "hidden":
            for i in range(len(s.get("hidden", [])) - 1, -1, -1):
                if s["hidden"][i]["kind"] == val:
                    del s["hidden"][i]; break
        else:
            s["marks"] = [m for m in s["marks"] if m["x"] != val]

    def toggle_rev(self, t: dict):
        """この本の目印の順を逆にする（キーの割り当てだけ・読みの x は変わらない・2026-10-22）。"""
        s = self.state[self.key(t)]; s["rev"] = not s.get("rev", False)
        t["markers_list"] = list(reversed(t["markers_list"]))

    def next_marker(self, t: dict) -> float | None:
        """次の目印 = 最後に記録した目印（並びの位置）より後ろで、まだ記録していないもの（2026-10-22: 写っていない端の目印や、歩く人が通らない ±12 を
        飛ばしたあとに m を押しても、飛ばした目印として記録しない）。何も記録していなければ並びの先頭。"""
        marks = self.state[self.key(t)]["marks"]; done = {m["x"] for m in marks}; lst = t["markers_list"]
        last = max((lst.index(m["x"]) for m in marks if m["x"] in lst), default=-1)
        for x in lst[last + 1:]:
            if x not in done:
                return x
        return None

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"reader": self.reader, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "takes": self.state},
                                              ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def sync_id(t: dict) -> str:
        """同じ原録音の印。sync_group 列があればそれ、無ければ take 名（step19d では原本の WAV 名）。"""
        return str(t.get("sync_group") or t.get("take") or "").strip()

    def clap_of(self, t: dict) -> dict | None:
        """この本の拍手。無ければ、**同じ原録音（sync_id）・同じ動画・同じ拍手_録音側秒**の別の行で押した拍手を使う
        （同じ WAV の相手と mic の行、1 本の録音に通過が何回もある測定用・2026-10-07）。
        2026-10-08 Astra 監査 D08-03: 録音側秒が違えば共用しない。
        2026-10-08 Astra 再監査 RD08-01: 録音側秒が同じでも、原録音が違えば共用しない（録音開始の約 2 秒後に拍手する別の本で衝突する）。
        原録音が同じかわからない（sync_group も take も違う）なら自動では共用しない。"""
        s = self.state[self.key(t)]
        if s["clap"]:
            return s["clap"]
        for u in self.takes:
            if (u is not t and self.sync_id(u) == self.sync_id(t) and str(u.get("video_path")) == str(t.get("video_path"))
                    and str(u.get("clap_audio_s", "")).strip() == str(t.get("clap_audio_s", "")).strip() and str(t.get("clap_audio_s", "")).strip()
                    and self.state[self.key(u)]["clap"]):
                return self.state[self.key(u)]["clap"]
        return None

    def audio_time(self, t: dict, t_video: float) -> float | None:
        c = self.clap_of(t)
        if not c or not t.get("clap_audio_s"):
            return None
        return t_video - float(c["t_video"]) + float(t["clap_audio_s"])

    def rows(self) -> tuple[list[dict], list[dict], list[str]]:
        """(step19f の行, 同期の行, 足りないテイクの注意) を返す"""
        out, sync, warn = [], [], []
        for t in self.takes:
            s = self.state[self.key(t)]
            c = self.clap_of(t)
            if not c or not t.get("clap_audio_s"):
                # Astra FR11-08（2026-10-12）: 足りないのは takes CSV の clap_audio_s（拍手_録音側秒）か動画の拍手 c。廃止したボタンの秒（拍手_音側秒）を埋めるようには案内しない
                warn.append(f"{t['take']}: 動画の拍手（c）が未記録か、takes CSV の clap_audio_s（拍手_録音側秒 = 本人が波形で読んだ値・step19j の候補は使わない・NT12-01）が空 → 出力しない"
                            "（ボタンの秒 拍手_音側秒 は使わない・変更票 #59）"); continue
            if len(s["marks"]) < 2:
                warn.append(f"{t['take']}: 目印が {len(s['marks'])} 点（2 点以上必要） → 出力しない"); continue
            kinds = [x["kind"] for x in sorted(s.get("hidden", []), key=lambda m: m["t_video"])]        # R72-05: h／j は 隠れ始め → 見え始め の順でそろっていること
            if kinds != ["hide", "show"] * (len(kinds) // 2) or len(kinds) % 2:
                warn.append(f"{t['take']}: 隠れの印が不完全（h／j の並び {kinds}・隠れ始めと見え始めを交互に・押し忘れは u で戻す） → 出力しない"); continue
            y = float(t["line_m"])
            for m in s["marks"]:
                ta = self.audio_time(t, float(m["t_video"]))
                out.append({"take": t["take"], "frame": int(round(ta / LABEL_RES)), "cls": t["cls"], "track": t["track"],
                            "x": m["x"], "y": y, "z": t.get("src_z", ""), "t": f"{ta:.3f}"})
            for h in s.get("hidden", []):                                            # B4: 隠れた印（x は空・hidden = start／end）
                ta = self.audio_time(t, float(h["t_video"]))
                out.append({"take": t["take"], "frame": int(round(ta / LABEL_RES)), "cls": t["cls"], "track": t["track"],
                            "x": "", "y": y, "z": "", "t": f"{ta:.3f}", "hidden": ("start" if h["kind"] == "hide" else "end")})
            sync.append({"take": t["take"], "clap": 1, "audio_s": t["clap_audio_s"], "video_s": f"{float(c['t_video']):.3f}"})
            if s["clap2"] and t.get("clap2_audio_s"):
                sync.append({"take": t["take"], "clap": 2, "audio_s": t["clap2_audio_s"], "video_s": f"{float(s['clap2']['t_video']):.3f}"})
        return out, sync, warn

    def write_outputs(self, out_dir: Path) -> tuple[Path, Path, list[str]]:
        rows, sync, warn = self.rows()
        out_dir.mkdir(parents=True, exist_ok=True)
        p1 = out_dir / f"reads_{self.reader}.csv"; p2 = out_dir / f"sync_{self.reader}.csv"
        with open(p1, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["take", "frame", "cls", "track", "x", "y", "z", "t", "hidden"]); w.writeheader(); w.writerows({**r, "hidden": r.get("hidden", "")} for r in rows)
        with open(p2, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["take", "clap", "audio_s", "video_s"]); w.writeheader(); w.writerows(sync)
        return p1, p2, warn


# ---------------------------------------------------------------- 画面
class WavePane:
    """動画の窓の下に出す録音の波形（W）（2026-10-22・本人「元のウィンドウで完結できませんか」）。上の細い帯 = 録音全体、下 = 拡大した範囲。
    動画の拍手（c）と録音の拍手（a）がそろうと、動画のいまのコマの録音の時刻（オレンジ）を真ん中に追いかける。
    音はカーソルの位置から鳴らす（Windows の再生の遅れで線がずれて見えないように、動く線は出さず、聞いた範囲を黄色で示す）。"""

    KEYS = ("a", "b", "e", "d", "i", "o", "Home", "comma", "period", "l", "L", "Escape")

    def __init__(self, parent, width: int, audio_dir: Path, reads: AudioReads, on_change, height: int = 120):
        import tkinter as tk
        self.W, self.H, self.OV = int(width), int(height), 24
        self.cv = tk.Canvas(parent, width=self.W, height=self.H, bg="white", highlightthickness=0); self.cv.pack()
        self.lab = tk.Label(parent, text="", justify="left", anchor="w", font=("Meiryo", 9), wraplength=self.W); self.lab.pack(fill="x")
        self.audio_dir = Path(audio_dir); self.reads = reads; self.on_change = on_change
        self.wav = None; self.x = None; self.sr = 1; self.env = None; self.bell = False
        self.center = 5.0; self.span = 10.0; self.cursor = None; self.vline = None; self.played = None; self.n_play = 0
        self.cv.bind("<Button-1>", self.click)

    def load(self, wav: str, bell: bool):
        self.bell = bell
        if wav == self.wav:
            self.draw(); return
        self.stop()
        self.wav = wav; p = self.audio_dir / wav
        if not p.exists():
            self.x = None; self.draw(); return
        import soundfile as sf
        y, self.sr = sf.read(str(p), dtype="float32", always_2d=True)
        self.x = y[:, 0].copy(); del y                                      # AmbiX の 1 ch 目 = W
        self.env = wav_envelope(self.x, self.W)
        dur = len(self.x) / self.sr; r = self.reads.d.get(wav, {})
        self.cursor = r.get("clap_audio_s"); self.vline = None
        if self.cursor is not None:
            self.center, self.span = self.cursor, 2.0
        else:                                                                # 拍手はふつう録音の最初の 10 秒の中
            self.span = min(10.0, dur); self.center = self.span / 2
        self.draw()

    def set_video_time(self, ta):
        self.vline = ta
        if ta is not None and self.x is not None:
            self.center = ta                                                 # 動画のいまのコマを真ん中に
        self.draw()

    def _dur(self):
        return len(self.x) / self.sr if self.x is not None else 1.0

    def _px(self, t):
        return (t - (self.center - self.span / 2)) / self.span * self.W

    def draw(self):
        import numpy as np
        cv = self.cv; cv.delete("all")
        if self.x is None:
            cv.create_text(10, 10, anchor="nw", text=f"録音が見つからない: {self.audio_dir / str(self.wav)}"); self.lab.config(text=""); return
        dur = self._dur(); mid = self.OV / 2; m = float(np.abs(self.env).max()) or 1.0
        t0 = self.center - self.span / 2; t1 = t0 + self.span
        y0 = self.OV + 4; hd = self.H - y0 - 2; midd = y0 + hd / 2
        if self.played is not None:
            a, b = self.played
            cv.create_rectangle(a / dur * self.W, 1, b / dur * self.W, self.OV - 1, fill="#fff3b0", outline="")
            cv.create_rectangle(max(0, self._px(a)), y0, min(self.W, self._px(b)), self.H, fill="#fff3b0", outline="")
        for i, (lo, hi) in enumerate(self.env):
            cv.create_line(i, mid - hi / m * mid, i, mid - lo / m * mid + 1, fill="#999")
        cv.create_rectangle(max(0, t0 / dur * self.W), 1, min(self.W - 1, t1 / dur * self.W), self.OV - 1, outline="#06c")
        i0, i1 = max(0, int(t0 * self.sr)), min(len(self.x), int(np.ceil(t1 * self.sr)))
        seg = self.x[i0:i1]; mm = 1.0
        if len(seg) > 1:
            mm = float(np.abs(seg).max()) or 1.0                             # 見えている範囲で縦を合わせる
            pa, pb = self._px(i0 / self.sr), self._px(i1 / self.sr)
            if len(seg) > (pb - pa):
                e2 = wav_envelope(seg, max(1, int(pb - pa)))
                for k, (lo, hi) in enumerate(e2):
                    cv.create_line(pa + k, midd - hi / mm * hd / 2, pa + k, midd - lo / mm * hd / 2 + 1, fill="#333")
            else:
                pts = []
                for k, v in enumerate(seg):
                    pts += [pa + k * (pb - pa) / (len(seg) - 1), midd - float(v) / mm * hd / 2]
                cv.create_line(*pts, fill="#333")
        cv.create_line(0, midd, self.W, midd, fill="#ddd")
        if self.vline is not None and not (0.0 <= self.vline <= dur):           # 動画の方が長い本（2026-10-22・本人「波形が途中で消えてる」）
            cv.create_text(self.W / 2, midd, text=f"このコマは録音の外（録音は 0〜{dur:.2f} s・いまのコマは {self.vline:.2f} s）", fill="#c00", font=("Meiryo", 10))
        r = self.reads.d.get(self.wav, {})
        for t, col, lab in ((r.get("clap_audio_s"), "#0a0", "拍手"), (r.get("bell_onset_s"), "#00c", "ベル"), (r.get("bell_end_s"), "#90c", "ベル終"),
                            (self.vline, "#f80", "動画のコマ"), (self.cursor, "#d00", "")):
            if t is None:
                continue
            cv.create_line(t / dur * self.W, 1, t / dur * self.W, self.OV - 1, fill=col)
            if t0 <= t <= t1:
                px = self._px(t); cv.create_line(px, y0, px, self.H, fill=col)
                if lab:
                    cv.create_text(px + 3, y0 + 2, anchor="nw", text=lab, fill=col, font=("Meiryo", 8))
        f = lambda v: "—" if v is None else f"{v:.4f} s"
        self.lab.config(text=(
            f"録音 {self.wav} 表示 {t0:.2f}〜{t1:.2f} s（{self.span * 1000:.0f} ms）カーソル {f(self.cursor)} ｜ 拍手 {f(r.get('clap_audio_s'))}"
            + (f"・ベル {f(r.get('bell_onset_s'))}〜{f(r.get('bell_end_s'))}" if self.bell else "") + "\n"
            "波形: クリック=カーソル（上の帯=その場所へ） ホイール／i o=拡大 , .=0.5 ms a=拍手 b=ベルの鳴り始め e=ベルの鳴り終わり d=消す l=カーソルから聞く L=範囲を聞く Esc=止める"))

    def click(self, e):
        if self.x is None:
            return
        if e.y < self.OV:
            self.center = min(max(e.x / self.W * self._dur(), 0.0), self._dur())
        else:
            self.cursor = round(self.center - self.span / 2 + e.x / self.W * self.span, 5)
        self.draw()

    def zoom(self, factor, at_px=None):
        if self.x is None:
            return
        at_px = self.W / 2 if at_px is None else at_px
        tm = self.center - self.span / 2 + at_px / self.W * self.span
        self.span = min(max(self.span * factor, 0.005), max(0.01, self._dur()))
        self.center = tm - (at_px / self.W - 0.5) * self.span; self.draw()

    def handle_key(self, k: str, sh: bool) -> bool:
        """波形のキーなら処理して True（動画の窓のキーと重ならない: a b e d i o Home , . l L Esc）。"""
        if k not in self.KEYS or self.x is None:
            return k in self.KEYS
        if k in ("comma", "period"):
            if self.cursor is not None:
                self.cursor = round(self.cursor + (0.005 if sh else 0.0005) * (1 if k == "period" else -1), 5); self.draw()
        elif k == "i": self.zoom(0.5)
        elif k == "o": self.zoom(2.0)
        elif k == "Home":
            if self.cursor is not None:
                self.center = self.cursor; self.draw()
        elif k == "a":
            if self.cursor is not None:
                self.reads.set(self.wav, "clap_audio_s", self.cursor); self.on_change()
        elif k == "b":
            if self.cursor is not None:
                self.reads.set(self.wav, "bell_onset_s", self.cursor); self.on_change()
        elif k == "e":
            if self.cursor is not None:
                self.reads.set(self.wav, "bell_end_s", self.cursor); self.on_change()
        elif k == "d":
            self.reads.clear(self.wav); self.on_change()
        elif k == "l":
            a = self.cursor if self.cursor is not None else (self.vline if self.vline is not None else self.center - self.span / 2)
            self.listen(a, a + 1.5)
        elif k == "L":
            self.listen(self.center - self.span / 2, self.center + self.span / 2)
        elif k == "Escape":
            self.stop(); self.draw()
        return True

    def listen(self, a: float, b: float):
        """[a, b) を鳴らす（winsound・Windows の標準）。ファイルは <出力dir>/_play0.wav と _play1.wav を交互に使う。"""
        import winsound
        self.stop()
        self.n_play += 1; p = self.reads.out_dir / f"_play{self.n_play % 2}.wav"
        a, b = make_play_wav(self.x, self.sr, a, b, p)
        if b - a < 0.01:
            return
        winsound.PlaySound(str(p), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.played = (a, b); self.draw()

    def stop(self):
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self.played = None


def pid_alive(pid: int) -> bool:
    """その番号のプロセスが動いているか（Windows は OpenProcess・ほかは os.kill(pid, 0)）。"""
    import os
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong(); ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259                              # STILL_ACTIVE
    try:
        os.kill(int(pid), 0); return True
    except OSError:
        return False


def take_lock(lock: Path) -> str | None:
    """鍵を取る。ほかの道具が同じ読みを開いていれば、その説明を返して鍵は取らない（2026-10-22）。"""
    import os
    if lock.exists():
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        if pid_alive(int(info.get("pid", 0))):
            return f"この読み（{lock.parent} の {lock.stem}）は、ほかの読み取りの道具（{info.get('started', '?')} に開いた）がもう開いています。"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    return None


def first_unread(sess, takes: list[dict]) -> int:
    """目印も拍手もまだ記録していない最初の行の番号（全部読んでいれば 0）。"""
    for i, t in enumerate(takes):
        s = sess.state[sess.key(t)]
        if not s["marks"] and not s["clap"]:
            return i
    return 0


def run_gui(takes: list[dict], out_dir: Path, reader: str, size: str = "", audio_dir: str | None = None) -> int:
    import tkinter as tk
    from PIL import Image, ImageTk
    lock = out_dir / f"reads_{reader}.lock"
    busy = take_lock(lock)
    if busy:                                                                 # 2 つ同時に開くと、あとで保存した方が先の読みを消す（2026-10-22）
        print("[止めた] " + busy + " その窓で続けるか、その窓を q で閉じてから開き直してください。")
        try:
            from tkinter import messagebox
            _r = tk.Tk(); _r.withdraw(); messagebox.showerror("読み取りの道具", busy + "\n\n2 つ同時に開くと、読みが消えます。その窓で続けるか、その窓を q で閉じてから開き直してください。"); _r.destroy()
        except Exception:
            pass
        return 3
    sess = Session(takes, out_dir / f"reads_{reader}.state.json", reader)
    root = tk.Tk(); root.title(f"目印の読み取り — 読む人 {reader}")
    # 絵の大きさ: --size WxH で指定。無ければ画面の 6 割（Windows の拡大率 150% でもタスクバーに隠れない）（2026-10-07・本人「でかくてギリギリ」）
    if size:
        CW, CH = (int(v) for v in size.lower().split("x"))
    else:
        CW = max(640, int(root.winfo_screenwidth() * (0.5 if audio_dir else 0.6))); CH = int(CW * 9 / 16)   # 波形の帯があるときは少し小さく（2026-10-22）
    canvas = tk.Canvas(root, width=CW, height=CH, bg="black"); canvas.pack()
    status = tk.Label(root, text="", justify="left", anchor="w", font=("Meiryo", 9), wraplength=CW); status.pack(fill="x")
    st = {"i": 0, "idx": 0, "video": None, "playing": False, "photo": None, "zoom": 1, "zc": None, "view": (0, 0, 1.0, 0.0, 0.0)}
    _lz = getattr(Image, "Resampling", Image).LANCZOS
    reads = AudioReads(out_dir, reader); reads.apply(takes)                # 波形で読んだ録音の拍手を同期に（2026-10-22）
    wave = None

    def open_take():
        t = takes[st["i"]]
        same = st["video"] is not None and str(st["video"].path) == str(Path(t["video_path"]))
        if st["video"] is not None and not same:
            st["video"].release()
        if not same:
            st["video"] = Video(Path(t["video_path"]))
        if wave is not None:
            wave.load(t["take"], bell=any(u["take"] == t["take"] and u.get("cls") in ("bike_bell", "horn") for u in takes))   # 歩く人の行も同じ録音のベルを出す（2026-10-23）・走りながらのクラクション E3 も b／e で鳴り始めと鳴り終わり（10/27）
        s = sess.state[sess.key(t)]
        if s["marks"] or s["clap"]:
            st["idx"] = (s["marks"][-1]["frame"] if s["marks"] else s["clap"]["frame"])
        elif not same:
            st["idx"] = 0
        # 同じ動画で記録がまだ無い本（走行音の測定のように 1 本の動画に通過が 10 回）は、いまのコマから続ける（2026-10-07・本人「n を押すたびに最初から？」）
        show()

    def show():
        t = takes[st["i"]]; v = st["video"]
        idx, img, tv, src = v.get(st["idx"]); st["idx"] = idx
        h, w = img.shape[:2]
        if st["zc"] is None:
            st["zc"] = (w / 2, h / 2)
        x0, y0, cw_, ch_ = zoom_crop(w, h, st["zoom"], *st["zc"])   # 拡大中は元の画素から切り出す（2026-10-22）
        sc = min(CW / cw_, CH / ch_); dw, dh = max(1, int(cw_ * sc)), max(1, int(ch_ * sc))
        im = Image.fromarray(img[y0:y0 + ch_, x0:x0 + cw_, ::-1]).resize((dw, dh), _lz)
        st["view"] = (x0, y0, sc, (CW - dw) / 2, (CH - dh) / 2)
        st["photo"] = ImageTk.PhotoImage(im)
        canvas.delete("all"); canvas.create_image(CW // 2, CH // 2, image=st["photo"])
        s = sess.state[sess.key(t)]
        ta = sess.audio_time(t, tv)
        keys = "  ".join(f"{MARKER_KEYS[k] if k < len(MARKER_KEYS) else 'm'}: x={x:+g}" for k, x in enumerate(t["markers_list"]))
        marks = ", ".join(f"x={m['x']:+g}@{m['t_video']:.2f}s" for m in s["marks"]) or "なし"
        nxt = sess.next_marker(t)
        status.config(text=(
            (f"【備考: {t['備考']}】\n" if t.get("備考") else "")          # 2026-10-26: 表に 備考 の列があれば出す（デモの組の読む表・無ければ今のまま）
            + ("【歩く人の行: 足元（体の真下）が十字を越えたコマで押す・拍手は自転車の行のを使う】\n" if t.get("cls") == "mic" else "")
            + f"[{st['i'] + 1}/{len(takes)}] {t['take']}  {Path(t['video_path']).name}  線 {t['line_m']} m  {t.get('direction', '前→後')}  cls={t['cls']} track={t.get('track', '1')}\n"
            f"コマ {idx}/{v.n - 1}  動画 {tv:.3f} s（{src}）  音 {'' if ta is None else f'{ta:.3f} s'}   拍手: {'なし' if not sess.clap_of(t) else f'{sess.clap_of(t)['t_video']:.3f}s' + ('' if s['clap'] else '（同じ動画の別の本の）')}"
            f"{'' if not s['clap2'] else f'  拍手2: {s['clap2']['t_video']:.3f}s'}\n"
            f"目印 {keys}   次 = {'済' if nxt is None else f'x={nxt:+g}（m キー）'}   記録: {marks}\n"
            f"←/→ 1 コマ  Shift 10 コマ  ↑/↓ 1 秒  Space 再生  c 拍手  v 拍手2  m 次の目印  u 取り消し  n/p 次/前  s 保存  q 保存して終了\n"
            f"拡大 {st['zoom']}×（z で 1→3→6・クリックした所を中心に・拡大中は元の 4K の画素）  r = この本の目印の順を逆に"))
        if wave is not None:
            wave.set_video_time(ta)

    def step(d):
        st["idx"] = max(0, min(st["idx"] + d, st["video"].n - 1)); show()

    def play():
        if st["playing"]:
            step(1); root.after(66, play)

    def key(e):
        k = e.keysym; sh = bool(e.state & 0x1)
        if wave is not None and wave.handle_key(k, sh):                  # 波形のキー（a b d i o Home , . l L Esc）
            return
        t = takes[st["i"]]; v = st["video"]
        idx, img, tv, src = v.get(st["idx"])
        if k == "Right": step(10 if sh else 1)
        elif k == "Left": step(-10 if sh else -1)
        elif k == "Up": step(int(round(v.fps)))
        elif k == "Down": step(-int(round(v.fps)))
        elif k == "space":
            st["playing"] = not st["playing"]; play()
        elif k == "c": sess.rec(t, "clap", idx, tv); show()
        elif k == "v": sess.rec(t, "clap2", idx, tv); show()
        elif k in MARKER_KEYS:
            j = MARKER_KEYS.index(k)
            if j < len(t["markers_list"]):
                sess.rec(t, "mark", idx, tv, t["markers_list"][j]); show()
        elif k == "m":
            nx = sess.next_marker(t)
            if nx is not None:
                sess.rec(t, "mark", idx, tv, nx); show()
        elif k == "r":
            sess.toggle_rev(t); show()
        elif k == "z":
            st["zoom"] = ZOOM_STEPS[(ZOOM_STEPS.index(st["zoom"]) + 1) % len(ZOOM_STEPS)]; show()
        elif k == "h": sess.rec(t, "hide", idx, tv); show()
        elif k == "j": sess.rec(t, "show", idx, tv); show()
        elif k == "u": sess.undo(t); show()
        elif k == "n" and st["i"] < len(takes) - 1:
            sess.save(); st["i"] += 1; open_take()
        elif k == "p" and st["i"] > 0:
            sess.save(); st["i"] -= 1; open_take()
        elif k == "s":
            sess.save(); p1, p2, warn = sess.write_outputs(out_dir); print("保存:", p1, p2, *warn, sep="\n  ")
        elif k == "q":
            close()

    def click(e):
        x0, y0, sc, ox, oy = st["view"]
        st["zc"] = (x0 + (e.x - ox) / sc, y0 + (e.y - oy) / sc)   # 押した所を元の絵の座標に直して中心にする
        if st["zoom"] == 1:
            st["zoom"] = ZOOM_STEPS[1]
        show()

    def close():                                                             # q でも × でも、保存して鍵を返す（2026-10-22）
        sess.save(); p1, p2, warn = sess.write_outputs(out_dir); print("保存:", p1, p2, *warn, sep="\n  ")
        try:
            lock.unlink()
        except OSError:
            pass
        if wave is not None:
            wave.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    st["i"] = first_unread(sess, takes)                                      # 開いたら、まだ何も読んでいない最初の行から（2026-10-22）
    root.bind("<Key>", key)
    canvas.bind("<Button-1>", click)
    if audio_dir:
        wave = WavePane(root, CW, Path(audio_dir), reads, on_change=lambda: (reads.apply(takes), show()))

        def on_wheel(e):                                                     # 波形の上でだけホイールで拡大／縮小
            if root.winfo_containing(e.x_root, e.y_root) is wave.cv:
                wave.zoom(0.5 if e.delta > 0 else 2.0, e.x_root - wave.cv.winfo_rootx())
        root.bind_all("<MouseWheel>", on_wheel)
    try:
        open_take()
        root.mainloop()
    finally:
        try:
            if lock.exists() and json.loads(lock.read_text(encoding="utf-8")).get("pid") == __import__("os").getpid():
                lock.unlink()
        except Exception:
            pass
    return 0


def export_outputs(takes_path: Path, out_dir: Path, reader: str) -> int:
    """画面を開かずに、状態ファイル（reads_<人>.state.json）から reads_<人>.csv・sync_<人>.csv を書き出す（2026-10-12・変更票 #59 r6）。
    夜に読んだあとで clap_audio_s をあとから埋めた本も、読み直さずに出せる。拍手の秒が無い本は出さず、その一覧を出す（終了コード 1）。"""
    takes = load_takes(takes_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    AudioReads(out_dir, reader).apply(takes)                                 # 波形で読んだ録音の拍手（2026-10-22）
    sess = Session(takes, out_dir / f"reads_{reader}.state.json", reader)
    p1, p2, warn = sess.write_outputs(out_dir)
    for w in warn:
        print("[WARN] " + w)
    print(f"書き出し: {len(takes) - len(warn)}/{len(takes)} 本 -> {p1}・{p2}" + (f"（{len(warn)} 本は出していない・上の [WARN]）" if warn else ""))
    return 0 if not warn else 1


# ---------------------------------------------------------------- 自己検査（画面なし）
def selftest(tmp: Path) -> int:
    import numpy as np
    import cv2
    tmp.mkdir(parents=True, exist_ok=True)
    fps, n = 30.0, 120
    vp = tmp / "synthetic.mp4"
    w = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(n):
        img = np.zeros((240, 320, 3), np.uint8); x = int(300 - i * 2.4)
        cv2.rectangle(img, (x, 100), (x + 15, 130), (255, 255, 255), -1); w.write(img)
    w.release()
    v = Video(vp)
    idx, img, t0, src = v.get(0); _, _, t45, _ = v.get(45); _, _, t44, _ = v.get(44); _, _, t46, _ = v.get(46)
    ok = [("動画のコマ数", v.n == n), ("時刻が単調（44 < 45 < 46）", t44 < t45 < t46),
          ("コマ 45 の時刻 ≈ 1.5 s（±1 コマ）", abs(t45 - 45 / fps) <= 1.0 / fps + 1e-6)]
    # takes CSV と記録
    takes_p = tmp / "takes.csv"
    takes_p.write_text("take,video,cls,track,line_m,direction,clap_audio_s,clap2_audio_s\n"
                       "t1,synthetic.mp4,car,1,2.0,前→後,3.200,\n"
                       "t2,synthetic.mp4,kick,1,-1.5,後→前,2.000,25.000\n", encoding="utf-8")
    takes = load_takes(takes_p)
    ok.append(("目印の順（前→後 = +10 → −10）", takes[0]["markers_list"] == [float(x) for x in range(10, -11, -2)]))
    ok.append(("目印の順（後→前 = −10 → +10）", takes[1]["markers_list"] == [float(x) for x in range(-10, 11, 2)]))
    ok.append(("拡大の切り出し: 1 倍は全体・3 倍は真ん中 1280×720・端は内側に寄せる（2026-10-22）",
               zoom_crop(3840, 2160, 1, 0, 0) == (0, 0, 3840, 2160) and zoom_crop(3840, 2160, 3, 1920, 1080) == (1280, 720, 1280, 720)
               and zoom_crop(3840, 2160, 3, 0, 0) == (0, 0, 1280, 720) and zoom_crop(3840, 2160, 6, 3840, 2160) == (3200, 1800, 640, 360)))
    _e = wav_envelope(np.array([0.0, 1.0, -2.0, 3.0, 0.5, -0.5], dtype=np.float32), 3)
    ok.append(("波形の包絡: 列ごとの最小と最大（2026-10-22）", _e.tolist() == [[0.0, 1.0], [-2.0, 3.0], [-0.5, 0.5]]))
    _ar = AudioReads(tmp / "ar", "T"); _ar.set("x_001.WAV", "clap_audio_s", 5.01234); _ar.set("x_001.WAV", "bell_onset_s", 12.5)
    _tk = [{"take": "x_001.WAV", "clap_audio_s": ""}, {"take": "x_001.WAV", "cls": "mic", "clap_audio_s": ""}, {"take": "x_002.WAV", "clap_audio_s": ""}]
    AudioReads(tmp / "ar", "T").apply(_tk)
    _csv = (tmp / "ar" / "audio_reads_T.csv").read_text(encoding="utf-8")
    ok.append(("録音の読み: 保存して読み直せる・同じ録音の行（相手と mic）に拍手が入る・ほかの録音は空のまま・csv に 2 つ",
               [t["clap_audio_s"] for t in _tk] == ["5.0123", "5.0123", ""] and "x_001.WAV,5.0123,12.5000" in _csv))
    _ar.set("x_001.WAV", "bell_end_s", 14.25)
    _csv2 = (tmp / "ar" / "audio_reads_T.csv").read_text(encoding="utf-8")
    ok.append(("録音の読み: ベルの鳴り終わり（e・2026-10-27）も残る・csv の 3 列目",
               AudioReads(tmp / "ar", "T").d["x_001.WAV"].get("bell_end_s") == 14.25 and "wav,clap_audio_s,bell_onset_s,bell_end_s" in _csv2
               and "x_001.WAV,5.0123,12.5000,14.2500" in _csv2 and "e" in WavePane.KEYS))
    (tmp / "rev.state.json").unlink(missing_ok=True)
    _sr = Session([dict(takes[1])], tmp / "rev.state.json", "T"); _t = _sr.takes[0]; _m0 = list(_t["markers_list"]); _sr.toggle_rev(_t)
    ok.append(("r: 目印の順が逆になり、次の目印も逆側から", _t["markers_list"] == _m0[::-1] and _sr.next_marker(_t) == _m0[-1]))
    _sr.save(); _t2 = dict(takes[1]); _t2["markers_list"] = list(_m0); _sr2 = Session([_t2], tmp / "rev.state.json", "T")
    ok.append(("r: 逆にしたことが状態ファイルに残り、開き直しても逆のまま", _sr2.takes[0]["markers_list"] == _m0[::-1]))
    _x = (np.sin(np.arange(96000 * 3) * 2 * np.pi * 1000 / 96000) * 0.01).astype(np.float32)
    _a, _b = make_play_wav(_x, 96000, 0.5, 20.0, tmp / "play.wav", max_s=2.0)
    import wave as _wv
    with _wv.open(str(tmp / "play.wav"), "rb") as _w:
        _fr, _nf = _w.getframerate(), _w.getnframes(); _pk = np.abs(np.frombuffer(_w.readframes(_nf), "<i2")).max() / 32767
    ok.append(("聞く: 96 kHz → 48 kHz・最大の長さで切る・持ち上げは 0.5 まで（2026-10-22）", _fr == 48000 and abs(_nf - 96000) <= 2 and (_a, _b) == (0.5, 2.5) and 0.45 < _pk <= 0.5))
    make_play_wav(_x * 0.0001, 96000, 0, 1, tmp / "play2.wav")
    with _wv.open(str(tmp / "play2.wav"), "rb") as _w:
        _pk2 = np.abs(np.frombuffer(_w.readframes(_w.getnframes()), "<i2")).max() / 32767
    ok.append(("聞く: 小さい音は 50 倍まで（背景だけの所を大きくしすぎない）", _pk2 < 0.01))
    (tmp / "skip.state.json").unlink(missing_ok=True)
    _sk = Session([dict(takes[1])], tmp / "skip.state.json", "T"); _u = _sk.takes[0]; _l = _u["markers_list"]
    _sk.rec(_u, "mark", 10, 0.3, _l[1]); _sk.rec(_u, "mark", 20, 0.6, _l[2])
    ok.append(("m: 先頭を飛ばしても、最後に記録した目印の次を選ぶ（飛ばした目印にしない・2026-10-22）", _sk.next_marker(_u) == _l[3]))
    (tmp / "fu.state.json").unlink(missing_ok=True)
    _fu = Session([dict(takes[0]), dict(takes[1])], tmp / "fu.state.json", "T")
    _a0 = first_unread(_fu, _fu.takes); _fu.rec(_fu.takes[0], "clap", 5, 0.2)
    _lk = tmp / "lk" / "reads_T.lock"; _lk.unlink(missing_ok=True)
    _b1 = take_lock(_lk); _b2 = take_lock(_lk)                                 # 2 回目は自分（動いている）が持っているので止まる
    _lk.write_text(json.dumps({"pid": 999999, "started": "x"}), encoding="utf-8"); _b3 = take_lock(_lk)   # 死んだ番号の鍵は取り直せる
    ok.append(("鍵: 1 つ目は開ける・同じ読みの 2 つ目は止める・閉じ忘れの古い鍵は取り直せる（2026-10-22）", _b1 is None and _b2 is not None and _b3 is None))
    _lk.unlink(missing_ok=True)
    ok.append(("開いたら、まだ読んでいない最初の行から（2026-10-22）", _a0 == 0 and first_unread(_fu, _fu.takes) == 1))
    ok.append(("波形のキーが動画の窓のキーと重ならない（2026-10-22）", not (set(WavePane.KEYS) & (set(MARKER_KEYS) | set("cvmhjunpsqzr") | {"Left", "Right", "Up", "Down", "space"}))))
    ok.append(("z と r が目印のキーとほかのキーに重ならない", all(k not in MARKER_KEYS and k not in "cvmhjunpsq" for k in "zr")))
    ok.append(("11 点すべてに異なるショートカット", len(MARKER_KEYS) == len(DEFAULT_MARKERS) == len(set(MARKER_KEYS))))
    sess = Session(takes, tmp / "reads_T.state.json", "T")
    sess.rec("t1", "clap", 10, 10 / fps)
    for j, fr in enumerate((20, 35, 50, 65, 80)):
        sess.rec("t1", "mark", fr, fr / fps, [10.0, 5.0, 0.0, -5.0, -10.0][j])
    sess.rec("t2", "clap", 5, 5 / fps); sess.rec("t2", "clap2", 100, 100 / fps)
    sess.rec("t2", "mark", 30, 30 / fps, -10.0); sess.rec("t2", "mark", 60, 60 / fps, 0.0)
    sess.rec("t2", "mark", 60, 60 / fps, 0.0)        # 読み直し → 置き換え
    sess.undo("t2"); sess.rec("t2", "mark", 61, 61 / fps, 0.0)
    sess.rec("t1", "hide", 40, 40 / fps); sess.rec("t1", "show", 45, 45 / fps)      # B4: t1 の隠れた区間（0.17 s・短い）
    sess.save()
    sess2 = Session(takes, tmp / "reads_T.state.json", "T")     # 再開
    ok.append(("再開で記録が戻る", sess2.state[sess2.key("t1")]["marks"] == sess.state[sess.key("t1")]["marks"] and sess2.state[sess2.key("t2")]["clap2"] is not None))
    p1, p2, warn = sess2.write_outputs(tmp)
    rows = list(csv.DictReader(open(p1, encoding="utf-8")))
    ok.append(("出力 7 行（t1 5 点＋t2 2 点）＋隠れた印 2 行（hidden start／end・x 空）", len(rows) == 9 and not warn
               and [r["hidden"] for r in rows if r["take"] == "t1" and r["hidden"]] == ["start", "end"] and all(r["x"] == "" for r in rows if r["hidden"])))
    rows = [r for r in rows if not r["hidden"]]
    r = [x for x in rows if x["take"] == "t1" and x["x"] == "10.0"][0]
    ta = 20 / fps - 10 / fps + 3.2
    ok.append(("音の時刻 = 動画 − 拍手_動画 + 拍手_音（t1 x=+10・ms 単位）", abs(float(r["t"]) - ta) < 5e-4 and int(r["frame"]) == round(ta / LABEL_RES)))
    ok.append(("y = 線の値（右は負）", [x for x in rows if x["take"] == "t2"][0]["y"] == "-1.5"))
    sync = list(csv.DictReader(open(p2, encoding="utf-8")))
    ok.append(("同期 csv: t1 は拍手 1、t2 は拍手 1 と 2", [(s["take"], s["clap"]) for s in sync] == [("t1", "1"), ("t2", "1"), ("t2", "2")]))
    # 2026-10-08 D08-03・RD08-01: 拍手の共用は「同じ原録音（sync_group か take）・同じ動画・同じ拍手_録音側秒」のときだけ
    tA = {"take": "a", "sync_group": "W1", "cls": "bicycle", "video_path": str(vp), "clap_audio_s": "2.0"}; tB = dict(tA, take="b")
    tC = dict(tA, take="c", clap_audio_s="3.0")
    tD = {"take": "d", "cls": "bicycle", "video_path": str(vp), "clap_audio_s": "2.0"}                       # 別の原録音・同じ秒
    w3 = {"take": "261008_003.WAV", "cls": "bicycle", "video_path": str(vp), "clap_audio_s": "2.00"}
    w4 = dict(w3, take="261008_004.WAV")                                                                      # 再監査の再現: 別 WAV・同じ動画・同じ秒
    w3m = dict(w3, cls="mic", track="0")                                                                      # 同じ WAV の mic 行
    s3 = Session([tA, tB, tC, tD, w3, w4, w3m], tmp / "share.state.json", "Z")
    s3.rec(tA, "clap", 300, 10.0); s3.rec(w3, "clap", 300, 10.0)
    ok.append(("拍手の共用: 同じ原録音（sync_group）・同じ動画・同じ録音側秒の本は借りる", s3.clap_of(tB) is not None and abs(s3.audio_time(tB, 16.0) - 8.0) < 1e-9))
    ok.append(("拍手の共用: 同じ原録音でも録音側秒が違う本は借りない", s3.clap_of(tC) is None and s3.audio_time(tC, 16.0) is None))
    ok.append(("拍手の共用: 録音側秒が同じでも原録音が違う本は借りない（RD08-01）", s3.clap_of(tD) is None and s3.clap_of(w4) is None and s3.audio_time(w4, 75.0) is None))
    ok.append(("拍手の共用: 同じ WAV の mic 行は借りる", s3.clap_of(w3m) is not None and abs(s3.audio_time(w3m, 16.0) - 8.0) < 1e-9))
    # step19f に通る（t 列あり）
    import subprocess
    r19 = subprocess.run([sys.executable, str(Path(__file__).with_name("step19f_video_to_gt.py")), str(p1), str(tmp / "gt"), "--mic-z", "1.6"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", env={"PYTHONIOENCODING": "utf-8"})
    ok.append(("step19f が t 列つきの出力を読める", r19.returncode == 0 and (tmp / "gt" / "t1.csv").exists()))
    # 2026-10-12 #59 r6: --export（画面なしの書き出し）と、FR11-08 の案内の文言
    rc_e = export_outputs(takes_p, tmp, "T")
    ok.append(("--export: 状態ファイルから reads/sync を書き出せる（rc 0）", rc_e == 0 and p1.exists()))
    takes_nc = tmp / "takes_noclap.csv"
    takes_nc.write_text("take,video,cls,track,line_m,direction,clap_audio_s,clap2_audio_s\n"
                        "t9,synthetic.mp4,bicycle,1,2.0,後→前,,\n", encoding="utf-8")
    s9 = Session(load_takes(takes_nc), tmp / "reads_N.state.json", "N")
    s9.rec("t9", "clap", 10, 10 / fps); s9.rec("t9", "mark", 20, 20 / fps, -10.0); s9.rec("t9", "mark", 40, 40 / fps, -8.0)
    _, _, w9 = s9.write_outputs(tmp)
    ok.append(("FR11-08: clap_audio_s（拍手_録音側秒）が空の本の案内は clap_audio_s を示し、廃止した 拍手_音側秒 を埋めるようには読めない",
               len(w9) == 1 and "clap_audio_s" in w9[0] and "拍手_音側秒 は使わない" in w9[0]))
    ok.append(("--export: 拍手の秒が無い本があると rc 1（出していない本の一覧つき）", export_outputs(takes_nc, tmp, "N") == 1))
    for name, cond in ok:
        print(("PASS " if cond else "FAIL ") + name)
    if r19.returncode != 0:
        print(r19.stderr[-600:])
    k = sum(c for _, c in ok)
    print(f"{k}/{len(ok)} {'ALL PASS' if k == len(ok) else 'FAILED'}")
    v.release()
    return 0 if k == len(ok) else 1


def main() -> int:
    argv = sys.argv[1:]
    def opt(k, d=None):
        if k in argv:
            i = argv.index(k); v = argv[i + 1]; del argv[i:i + 2]; return v
        return d
    st = opt("--selftest")
    if st:
        return selftest(Path(st))
    export = "--export" in argv
    if export:
        argv.remove("--export")
    takes_p = opt("--takes"); out = opt("--out"); reader = opt("--reader", "A"); size = opt("--size", ""); audio_dir = opt("--audio-dir")
    if not takes_p or not out:
        print(__doc__); return 2
    if export:
        return export_outputs(Path(takes_p), Path(out), reader)
    return run_gui(load_takes(Path(takes_p)), Path(out), reader, size, audio_dir)


if __name__ == "__main__":
    sys.exit(main())
