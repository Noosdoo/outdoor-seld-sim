# -*- coding: utf-8 -*-
"""run2 ckpt → SDE用ウォームスタートckpt変換。

出力ヘッド tscam_conv を 3軸(out=C*9) から 4軸(out=C*12) へ拡張する:
- 旧チャンネル (track*3+axis)*C + class を新 (track*4+axis)*C + class へコピー
- 距離軸 (track*4+3)*C + class は**ゼロ初期化**
  → 初期状態は「run2と同一挙動＋距離出力0」（劣化検証ゲートが初期値で成立）

使い方(サーバ、PSELDNetsのvenvで):
    python _make_sde_init_ckpt.py <run2のepoch_094.ckpt> <出力パス>
"""
import sys

import torch


def expand_head(w, n_axes_old=3, n_axes_new=4):
    """[C*3*n_old, ...] -> [C*3*n_new, ...]（トラック3固定・距離行ゼロ）。"""
    out_old = w.shape[0]
    assert out_old % (3 * n_axes_old) == 0, f"想定外のout_ch: {out_old}"
    C = out_old // (3 * n_axes_old)
    new_shape = (C * 3 * n_axes_new,) + tuple(w.shape[1:])
    w_new = torch.zeros(new_shape, dtype=w.dtype)
    for t in range(3):
        for a in range(n_axes_old):
            src = slice((t * n_axes_old + a) * C, (t * n_axes_old + a + 1) * C)
            dst = slice((t * n_axes_new + a) * C, (t * n_axes_new + a + 1) * C)
            w_new[dst] = w[src]
    return w_new, C


def main(src_path, dst_path):
    ckpt = torch.load(src_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    hit = 0
    for k in list(sd.keys()):
        if "tscam_conv" in k and (k.endswith(".weight") or k.endswith(".bias")):
            w_new, C = expand_head(sd[k])
            print(f"  {k}: {tuple(sd[k].shape)} -> {tuple(w_new.shape)} "
                  f"(C={C}, 距離行=0初期化)")
            sd[k] = w_new
            hit += 1
    assert hit >= 2, f"tscam_convが見つからない/少なすぎる: {hit}"
    # 学習の途中再開ではなく初期値として使う（epoch等の再開情報を無効化）
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
