# -*- coding: utf-8 -*-
"""クリップ識別子の規則（2026-09-22・本人決定・md/design/識別子の改名_手順書_2026-09-22.md）。

新: ``<用途>_<出所>_clip<番号 5 桁>``   例 ``train_syn_clip28028``・``eval_syn_clip00001``・``eval_field_clip00001``
  用途 = train（学習）/ val（検証・ckpt 選択器の入力）/ eval（評価: 合成は封印した確定評価、実録は収録したもの）/ tune（調整用・予約）
  出所 = syn（合成）/ field（実録）
旧: ``fold<F>_room<R>_mix<番号 4〜5 桁>``（DCASE 形式。fold1 = 学習、fold2 = 検証、fold20 = 確定評価。room は屋外では常に 1）

対応は 1 対 1（番号の値は不変・5 桁にゼロ埋め）。旧名の集合のうち v2 の本線が使うのは fold1/fold2/fold20 だけで、
fold3（v11 の test）と fold10〜13（v12_eval）は改名せず履歴として残す（to_new は受け付けない）。
"""
from __future__ import annotations

import re

SPLITS = ("train", "val", "eval", "tune")
SOURCES = ("syn", "field")
OLD_TO_SPLIT = {"fold1": "train", "fold2": "val", "fold20": "eval"}
SPLIT_TO_OLD = {v: k for k, v in OLD_TO_SPLIT.items()}
RE_NEW = re.compile(r"^(train|val|eval|tune)_(syn|field)_clip(\d{5})$")
RE_OLD = re.compile(r"^(fold\d+)_room(\d+)_mix(\d{4,5})$")
PREFIX_NEW = {s: f"{s}_syn_" for s in SPLITS}          # 例: 学習用の合成クリップは "train_syn_" で始まる


def is_new(clip_id: str) -> bool:
    return RE_NEW.match(str(clip_id)) is not None


def is_old(clip_id: str) -> bool:
    return RE_OLD.match(str(clip_id)) is not None


def parse(clip_id: str) -> tuple[str, str, int]:
    """新名 → (用途, 出所, 番号)。旧名なら to_new してから読む。"""
    m = RE_NEW.match(str(clip_id))
    if m is None:
        m = RE_NEW.match(to_new(clip_id))
    return m.group(1), m.group(2), int(m.group(3))


def make(split: str, source: str, number: int) -> str:
    assert split in SPLITS and source in SOURCES and 0 <= int(number) <= 99999, (split, source, number)
    return f"{split}_{source}_clip{int(number):05d}"


def make_field(session_no: int, n: int, split: str = "eval") -> str:
    """実録（本人決定 2026-09-22 18:57・案 A）: 番号 = セッション番号（2 桁・01〜99・session.csv の登録順）× 1000 ＋ セッション内の通し番号（3 桁・001〜999・切り出し順）。
    例 eval_field_clip03012 = セッション 3 の 12 本目。テイク・事象との対応は切り出しの記録表と eval_field_index.md。"""
    s, k = int(session_no), int(n)
    assert 1 <= s <= 99 and 1 <= k <= 999, f"セッション番号 1〜99・通し番号 1〜999 の範囲外: {session_no}, {n}"
    return make(split, "field", s * 1000 + k)


def parse_field(clip_id: str) -> tuple[int, int]:
    """実録の新名 → (セッション番号, セッション内の通し番号)。"""
    split, source, num = parse(clip_id)
    if source != "field":
        raise ValueError(f"実録の名前でない: {clip_id}")
    return num // 1000, num % 1000


def to_new(clip_id: str) -> str:
    """旧名 → 新名。新名はそのまま返す。対応の無い旧名（fold3・fold10〜13 など）は ValueError。"""
    s = str(clip_id)
    if RE_NEW.match(s):
        return s
    m = RE_OLD.match(s)
    if m is None:
        raise ValueError(f"識別子の形が違う: {s}")
    fold, room, num = m.group(1), m.group(2), int(m.group(3))
    if fold not in OLD_TO_SPLIT:
        raise ValueError(f"新名に対応しない区分（履歴として旧名のまま残す）: {s}")
    if room != "1":
        raise ValueError(f"room が 1 でない旧名は対象外: {s}")
    return make(OLD_TO_SPLIT[fold], "syn", num)


def to_old(clip_id: str) -> str:
    """新名 → 旧名（合成のみ。実録は旧名が無い）。番号の桁は旧名の慣習に合わせる（10000 未満は 4 桁、以上は 5 桁）。"""
    s = str(clip_id)
    if RE_OLD.match(s):
        return s
    split, source, num = parse(s)
    if source != "syn" or split not in SPLIT_TO_OLD:
        raise ValueError(f"旧名が無い: {s}")
    return f"{SPLIT_TO_OLD[split]}_room1_mix{num:04d}" if num < 10000 else f"{SPLIT_TO_OLD[split]}_room1_mix{num:05d}"


def compat(clip_id: str) -> str:
    """旧名でも新名でも受け、新名を返す（過去の結果ファイルとの突き合わせ用）。対応の無い名前は ValueError。"""
    return to_new(clip_id)


def normalize(clip_id: str) -> str:
    """読み取り側の緩い正規化: 新名はそのまま、対応のある旧名は新名に、それ以外（履歴の fold3 など・検査用の仮名）はそのまま返す。
    予測 CSV や manifest を読むところで使う（過去の走行の結果を新名の plan・GT と突き合わせられる）。"""
    s = str(clip_id)
    if RE_NEW.match(s):
        return s
    try:
        return to_new(s)
    except ValueError:
        return s


def split_to_new(split: str) -> str:
    """plan の split 列の値（fold1/fold2 → train/val）。新しい値はそのまま。"""
    if split in SPLITS:
        return split
    if split in OLD_TO_SPLIT:
        return OLD_TO_SPLIT[split]
    raise ValueError(f"split の値が対応表に無い: {split}")


def number(clip_id: str) -> int:
    """名前の番号（旧名・新名どちらでも）。"""
    return parse(clip_id)[2]
