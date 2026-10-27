# -*- coding: utf-8 -*-
"""8 クラスの初期化 ckpt（v12_init_from_run3ep84.ckpt）から 9 クラス用の初期化 ckpt を作る（2026-10-08・変更票 #48 D22 bicycle）。

出力層 tscam_conv は out_channels = C × 3 枠 × 4 軸（x, y, z, 距離）で、チャンネルの並びは (track*4 + axis)*C + class（sde_patch.py 冒頭の規約・
クラスが一番内側）。8 クラス = 96 → 9 クラス = 108 に広げる。既存 8 クラスの重みとバイアスはそのまま写し、9 番目（bicycle）の行は 0 で始める
（tanh(0) = 0 = 「鳴っていない」から学習が始まる。並べ替えはしない）。ほかの層は全部そのまま。
使い方: .venv/bin/python server_sde/_make_init9.py <in.ckpt> <out.ckpt> [--c-old 8] [--c-new 9]
検査: 読み戻して 8 クラス分の各ブロックが元と一致し、9 番目の行が 0 であることを確かめる。sha256 を表示。
"""
import hashlib
import sys

import torch

TRACKS, AXES = 3, 4


def expand(w: torch.Tensor, c_old: int, c_new: int) -> torch.Tensor:
    n = w.shape[0]
    assert n == TRACKS * AXES * c_old, (n, c_old)
    blocks = w.reshape(TRACKS * AXES, c_old, *w.shape[1:])
    pad = torch.zeros((TRACKS * AXES, c_new - c_old, *w.shape[1:]), dtype=w.dtype)
    return torch.cat([blocks, pad], dim=1).reshape(TRACKS * AXES * c_new, *w.shape[1:])


def main() -> int:
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    c_old = int(a[a.index("--c-old") + 1]) if "--c-old" in a else 8
    c_new = int(a[a.index("--c-new") + 1]) if "--c-new" in a else 9
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    keys = [k for k in sd if k.endswith("tscam_conv.weight") or k.endswith("tscam_conv.bias")]
    assert len(keys) == 2, keys
    for k in keys:
        old = sd[k]
        new = expand(old, c_old, c_new)
        # 検査: 各 (track, axis) ブロックの先頭 c_old 行が元と一致・残りが 0
        nb = new.reshape(TRACKS * AXES, c_new, *new.shape[1:]); ob = old.reshape(TRACKS * AXES, c_old, *old.shape[1:])
        assert torch.equal(nb[:, :c_old], ob) and float(nb[:, c_old:].abs().sum()) == 0.0
        sd[k] = new
        print(f"{k}: {tuple(old.shape)} -> {tuple(new.shape)}")
    ck["state_dict"] = sd
    ck.setdefault("v22_init9", {})["note"] = f"expanded tscam_conv from {c_old} to {c_new} classes (class-innermost layout, new rows zero) on 2026-10-08"
    torch.save(ck, dst)
    print("sha256", hashlib.sha256(open(dst, "rb").read()).hexdigest(), dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
