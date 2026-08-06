# -*- coding: utf-8 -*-
"""run3(SDE 6クラス) ckpt → v12(8クラス)ウォームスタートckpt変換。

tscam_conv を C=6 → C=8 へクラス次元拡張する（軸4=xyz+dist・トラック3は不変）:
- 旧チャンネル (track*4+axis)*6 + class を新 (track*4+axis)*8 + class へコピー
- 新クラス行 (class=6,7 = Kickboard/Motorcycle) は**ゼロ初期化**
  → 初期状態は「run3と同一挙動＋新クラス出力0」（既存6クラスの劣化ゲートが初期値で成立）

使い方(サーバ、PSELDNetsのvenvで):
    python _make_v12_init_ckpt.py <run3のepoch_084.ckpt> <出力パス>
"""
import sys

import torch

N_AXES = 4
N_TRACKS = 3
C_OLD, C_NEW = 6, 8


def expand_classes(w):
    out_old = w.shape[0]
    assert out_old == N_TRACKS * N_AXES * C_OLD, f"想定外のout_ch: {out_old}"
    new_shape = (N_TRACKS * N_AXES * C_NEW,) + tuple(w.shape[1:])
    w_new = torch.zeros(new_shape, dtype=w.dtype)
    for t in range(N_TRACKS):
        for a in range(N_AXES):
            src = slice((t * N_AXES + a) * C_OLD, (t * N_AXES + a) * C_OLD + C_OLD)
            dst = slice((t * N_AXES + a) * C_NEW, (t * N_AXES + a) * C_NEW + C_OLD)
            w_new[dst] = w[src]
    return w_new


def main(src_path, dst_path):
    ckpt = torch.load(src_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    hit = 0
    for k in list(sd.keys()):
        if "tscam_conv" in k and (k.endswith(".weight") or k.endswith(".bias")):
            w_new = expand_classes(sd[k])
            print(f"  {k}: {tuple(sd[k].shape)} -> {tuple(w_new.shape)} "
                  f"(class 6,7 = 0初期化)")
            sd[k] = w_new
            hit += 1
    assert hit >= 2, f"tscam_convが見つからない/少なすぎる: {hit}"
    for key in ("epoch", "global_step"):
        if key in ckpt:
            ckpt[key] = 0
    ckpt.pop("optimizer_states", None)
    ckpt.pop("lr_schedulers", None)
    ckpt.pop("loops", None)
    torch.save(ckpt, dst_path)
    print(f"saved -> {dst_path} (置換 {hit} テンソル)")


if __name__ == "__main__":
    assert len(sys.argv) == 3, __doc__
    main(sys.argv[1], sys.argv[2])
