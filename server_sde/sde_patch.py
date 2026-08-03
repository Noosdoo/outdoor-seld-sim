# -*- coding: utf-8 -*-
"""役割③: PSELDNetsに距離推定(SDE)を後付けするモンキーパッチ集。

pin版(8092a14)のコードは書き換えず、importして属性を差し替える。
設計の正: outdoor_seld_e2e/md/design/役割③SDE実装_設計_2026-08-03.md

出力レイアウト(確定):
  ヘッドout_channels = classes*3トラック*4軸、チャンネル = (track*4+axis)*C + class
  axis: 0=x,1=y,2=z,3=dist(ラベル単位 = m * DIST_SCALE、活動マスク済み)
ラベル: adpit_dist h5 = se/azi/ele/dist 各[frames,6,C]（既存adpitのdist追加版）
損失: 置換選択はxyzのみ(既存と同一) + 選ばれた置換の距離MSE×W_DIST
使い方: _train_sde.py / _preproc_sde.py から apply_*() を呼ぶ。
"""
from __future__ import annotations

import os

import numpy as np

DIST_SCALE = 0.1                      # linearモード時: m -> ラベル単位 (d/10)
W_DIST = float(os.environ.get("SDE_W_DIST", "2.0"))
# 【run3】距離スケール: log(既定)=log1p(d)/log1p(100)。遠距離の誤差を圧縮して
# SELD本体への勾配圧迫を抑えつつ、近距離の分解能を保つ（run2の0.092劣化への対応）
DIST_MODE = os.environ.get("SDE_DIST_MODE", "log")
LOG_K = float(np.log1p(100.0))
XYZ_IDX = [0, 1, 2, 4, 5, 6, 8, 9, 10]
D_IDX = [3, 7, 11]


def dist_encode(d):
    """m -> ラベル単位（配列可）。"""
    if DIST_MODE == "log":
        return np.log1p(d) / LOG_K
    return d * DIST_SCALE


def dist_decode(y):
    """ラベル単位 -> m（スカラ・非負）。"""
    y = max(float(y), 0.0)
    if DIST_MODE == "log":
        return float(np.expm1(y * LOG_K))
    return y / DIST_SCALE


# ---------------------------------------------------------------- 共通(ローダ)
def apply_loader_patch():
    """load_output_format_file: 空ファイル対応({99:[]}) + 6列のdist保持。"""
    import pandas as pd
    import utils.data_utilities as du

    def load_output_format_file(_output_format_file):
        if os.path.getsize(_output_format_file) == 0:
            return {99: []}                     # 空ラベル=設計データ(v11の589本)
        out = {}
        df = pd.read_csv(_output_format_file, header=None).values
        for it in df:
            f = int(it[0])
            out.setdefault(f, [])
            if len(it) == 5:      # [frame, class, track, az, el]
                out[f].append([int(it[1]), float(it[3]), float(it[4])])
            elif len(it) == 6:    # [frame, class, track, az, el, dist] ← dist保持
                out[f].append([int(it[1]), float(it[3]), float(it[4]),
                               float(it[5])])
            elif len(it) == 4:
                out[f].append([int(it[1]), float(it[2]), float(it[3])])
            elif len(it) == 7:
                out[f].append([int(it[1]), float(it[3]), float(it[4]),
                               float(it[5])])
        return out

    du.load_output_format_file = load_output_format_file
    # from-import済みの名前空間にも張り替え
    for modname in ("data.components.data", "preproc.preprocess",
                    "models.components.model_module"):
        try:
            import importlib
            m = importlib.import_module(modname)
            if hasattr(m, "load_output_format_file"):
                m.load_output_format_file = load_output_format_file
        except Exception:
            pass


# ---------------------------------------------------------------- 学習側
def apply_train_patches():
    """データセット(ラベル5軸化)・モデル(4軸ヘッド)・損失(距離項)を差し替え。"""
    import h5py
    import torch
    import torch.nn as nn
    from pathlib import Path

    apply_loader_patch()

    # --- 1) データセット: adpit_dist を読み、ラベルを [T,6,5,C] に拡張 -------
    import data.components.data as dcomp
    import data.data as ddata

    _orig_init = dcomp.BaseDataset.__init__

    def _init_sde(self, *a, **kw):
        _orig_init(self, *a, **kw)
        # label/adpit/<stage> -> label/adpit_dist/<stage>
        p = Path(str(self.adpit_label_dir))
        self.adpit_label_dir = p.parent.parent / "adpit_dist" / p.name

    dcomp.BaseDataset.__init__ = _init_sde

    _orig_getitem = ddata.DatasetMultiACCDOA.__getitem__

    def _getitem_sde(self, idx):
        sample = _orig_getitem(self, idx)
        if "adpit_label" not in sample:
            return sample                        # testステージ(ラベルなし)
        assert self.cfg.adapt.method != "mono_adapter", \
            "SDEパッチはmono_adapter未対応"
        lab = sample["adpit_label"]              # [T,6,4,C] (padded済み)
        clip_indexes = self.segments_list[idx]
        path, segments = clip_indexes[0], clip_indexes[1:]
        fn = Path(path).stem
        dataset = path.split("/")[-3] if self.audio_feature in ["logmelIV"] \
            else path.split("/")[-2]
        i0 = int(segments[0] / self.points_per_predictions)
        i1 = int(segments[1] / self.points_per_predictions)
        meta_path = self.adpit_label_dir.joinpath(f"{dataset}.h5")
        with h5py.File(meta_path, "r") as hf:
            dist = hf[f"{fn}/adpit/dist"][i0:i1, ...].astype(np.float32)
        se = lab[:dist.shape[0], :, 0, :]
        d_axis = np.zeros((lab.shape[0], lab.shape[1], 1, lab.shape[3]),
                          dtype=np.float32)
        d_axis[:dist.shape[0], :, 0, :] = se * dist_encode(dist)
        sample["adpit_label"] = np.concatenate((lab, d_axis), axis=2)  # 5軸
        return sample

    ddata.DatasetMultiACCDOA.__getitem__ = _getitem_sde

    # --- 2) モデル: HTSATヘッドを 3*4軸 に --------------------------------
    import models.accdoa as accdoa_mod
    import models.multi_accdoa as ma_mod
    from utils.utilities import get_pylogger
    log = get_pylogger(__name__)

    class HTSAT_SDE(accdoa_mod.HTSAT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tscam_conv = nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.num_classes * 3 * 4,   # 3トラック×(xyz+dist)
                kernel_size=(self.encoder.SF, 3),
                padding=(0, 1))
            self.fc = nn.Identity()
            log.info("[SDE] head out = classes*3*4 = %d",
                     self.num_classes * 12)
            # 【監査対応v2】tanh天井の修正: final_actを外し、forwardでxyz行のみ
            # tanhを掛ける（xyz挙動は従来と完全同一・距離行は線形=表現域無制限）。
            # 旧版はDIST_SCALE=0.1でtanhにより出力可能距離が(-10m,+10m)に固定されていた
            self.final_act = nn.Identity()
            # 既存のload_ckptsはtscam_convを必ずスキップする(accdoa.py L198)ため、
            # ウォームスタートのヘッドはここで自前ロードする(距離行=0の変換済ckpt)
            init_ckpt = os.environ.get("SDE_INIT_CKPT", "")
            if init_ckpt:
                sd = torch.load(init_ckpt, map_location="cpu")
                sd = sd.get("state_dict", sd)
                sd = {k.replace("net.", "").replace("_orig_mod.", ""): v
                      for k, v in sd.items()}
                self.tscam_conv.weight.data.copy_(sd["tscam_conv.weight"])
                self.tscam_conv.bias.data.copy_(sd["tscam_conv.bias"])
                log.info("[SDE] head warm-start <- %s", init_ckpt)

        def forward(self, x):
            out = super().forward(x)["accdoa"]      # 生値(final_act=Identity)
            B, T, _ = out.shape
            o = out.reshape(B, T, 12, self.num_classes).clone()
            o[:, :, XYZ_IDX] = torch.tanh(o[:, :, XYZ_IDX])  # xyzは従来と同一
            # 距離行(D_IDX)は線形のまま=天井なし
            return {"multi_accdoa": o.reshape(B, T, -1)}

    ma_mod.HTSAT = HTSAT_SDE

    # 【監査対応v2.1】SDE未対応の経路を塞ぐガード。
    # 注意: hydraはaugment.type=[]でも設定ツリー上のオブジェクトを生成するため、
    # 生成(__init__)は許し「実際に使われた瞬間」だけ例外にする
    # （v2はここで生成時に例外を投げてジョブ194が起動7秒で死んだ。教訓）
    try:
        import augment as aug_mod

        def _make_blocked(name):
            class _BlockedAug:
                def __init__(self, *a, **kw):
                    pass

                def __call__(self, *a, **kw):
                    raise NotImplementedError(
                        f"[SDE] augment.{name} は5軸ラベル未対応（ガード）")

            _BlockedAug.__name__ = f"Blocked{name}"
            return _BlockedAug

        for _name in ("Rotation", "TrackMix"):
            if hasattr(aug_mod, _name):
                setattr(aug_mod, _name, _make_blocked(_name))
    except Exception:
        pass

    # --- 3) 損失: xyzで置換選択 + 距離MSE(W_DIST) --------------------------
    import loss.multi_accdoa as loss_mod

    class Losses_SDE(object):
        def __init__(self, loss_fn, loss_type):
            self.losses = nn.MSELoss(reduction="none")
            self.names = ["loss_all", "loss_adpit", "loss_other"]
            self.loss_type = loss_type
            self.loss_dict_keys = ["loss_all", "loss_adpit", "loss_other"]

        def __call__(self, output, target, epoch_it=0):
            out = output["multi_accdoa"]         # [B,T,12*C]
            tgt = target["adpit_label"]          # [B,T,6,5,C]
            tok = [tgt[:, :, i, 0:1, :] * tgt[:, :, i, 1:, :]
                   for i in range(6)]            # 各 [B,T,4,C]
            A0, B0, B1, C0, C1, C2 = tok
            perms = [
                (A0, A0, A0), (B0, B0, B1), (B0, B1, B0), (B0, B1, B1),
                (B1, B0, B0), (B1, B0, B1), (B1, B1, B0),
                (C0, C1, C2), (C0, C2, C1), (C1, C0, C2), (C1, C2, C0),
                (C2, C0, C1), (C2, C1, C0)]
            catA = torch.cat((A0, A0, A0), 2)
            catB = torch.cat((B0, B0, B1), 2)
            catC = torch.cat((C0, C1, C2), 2)
            pads = [catB + catC] + [catA + catC] * 6 + [catA + catB] * 6
            out = out.reshape(out.shape[0], out.shape[1], 12, tgt.shape[-1])
            xyz_losses, d_losses = [], []
            for (t0, t1, t2), pad in zip(perms, pads):
                t_full = torch.cat((t0, t1, t2), 2) + pad     # [B,T,12,C]
                mse = self.losses(out, t_full)
                xyz_losses.append(mse[:, :, XYZ_IDX, :].mean(dim=2))
                d_losses.append(mse[:, :, D_IDX, :].mean(dim=2))
            loss_min = torch.min(torch.stack(xyz_losses, dim=0), dim=0).indices
            sel_xyz = sum(l * (loss_min == n)
                          for n, l in enumerate(xyz_losses)).mean()
            sel_d = sum(l * (loss_min == n)
                        for n, l in enumerate(d_losses)).mean()
            return {"loss_all": sel_xyz + W_DIST * sel_d,
                    "loss_adpit": sel_xyz,
                    "loss_other": W_DIST * sel_d}

    loss_mod.Losses = Losses_SDE

    # --- 4) 推論デコード: sed判定はxyzのみ、doaに距離ブロックを同乗 --------
    import utils.data_utilities as du
    import models.components.model_module as mm

    def get_multi_accdoa_labels_sde(accdoa_in, nb_classes=13,
                                    sed_threshold=0.5):
        seds, doas = [], []
        for t in range(3):
            b = 4 * t * nb_classes
            x = accdoa_in[:, :, b:b + nb_classes]
            y = accdoa_in[:, :, b + nb_classes:b + 2 * nb_classes]
            z = accdoa_in[:, :, b + 2 * nb_classes:b + 3 * nb_classes]
            d = accdoa_in[:, :, b + 3 * nb_classes:b + 4 * nb_classes]
            seds.append(torch.sqrt(x**2 + y**2 + z**2) > sed_threshold)
            doas.append(torch.cat((x, y, z, d), dim=-1))   # 4Cブロック
        return torch.stack(seds, axis=0), torch.stack(doas, axis=0)

    def multi_accdoa_to_dcase_format_sde(sed_pred, doa_pred,
                                         threshold_unify=15, nb_classes=13):
        from utils.data_utilities import \
            distance_between_cartesian_coordinates as _dcc

        def sim(p0, p1):
            return 1 if _dcc(p0[0], p0[1], p0[2],
                             p1[0], p1[1], p1[2]) < threshold_unify else 0

        def merge(e0, e1):
            return [e0[0]] + [(a + b) / 2 for a, b in zip(e0[1:], e1[1:])]

        def merge3(e0, e1, e2):  # 【監査対応v2】原本と同じ等重み(1/3ずつ)
            return [e0[0]] + [(a + b + c) / 3 for a, b, c
                              in zip(e0[1:], e1[1:], e2[1:])]

        temp, outd = {}, {}
        tl, fl, cl = np.where(sed_pred == 1.)
        for t, f, c in zip(tl, fl, cl):
            temp.setdefault(f, []).append([
                c, doa_pred[t, f, c], doa_pred[t, f, c + nb_classes],
                doa_pred[t, f, c + 2 * nb_classes],
                dist_decode(doa_pred[t, f, c + 3 * nb_classes])])  # dist[m]
        for f, evs in temp.items():
            evs.sort(key=lambda x: x[0])
            outd[f] = []
            grp = []
            for i, ev in enumerate(evs):
                grp.append(ev)
                if i == len(evs) - 1 or ev[0] != evs[i + 1][0]:
                    if len(grp) == 1:
                        outd[f].append(grp[0])
                    elif len(grp) == 2:
                        if sim(grp[0][1:4], grp[1][1:4]):
                            outd[f].append(merge(grp[0], grp[1]))
                        else:
                            outd[f] += [grp[0], grp[1]]
                    else:
                        f01 = sim(grp[0][1:4], grp[1][1:4])
                        f12 = sim(grp[1][1:4], grp[2][1:4])
                        f02 = sim(grp[0][1:4], grp[2][1:4])
                        s = f01 + f12 + f02
                        if s == 0:
                            outd[f] += [grp[0], grp[1], grp[2]]
                        elif s == 1:
                            # 【監査対応v2】f02分岐は原本(data_utilities.py)と
                            # ビット互換にする（原本はgrp[1]を捨てgrp[0]を二重計上
                            # する挙動。上流バグだが、run2との比較可能性を優先して
                            # 同一挙動に合わせる。別途上流に報告対象）
                            if f01:
                                outd[f] += [merge(grp[0], grp[1]), grp[2]]
                            elif f12:
                                outd[f] += [grp[0], merge(grp[1], grp[2])]
                            else:
                                outd[f] += [grp[0], merge(grp[0], grp[2])]
                        else:
                            outd[f].append(merge3(grp[0], grp[1], grp[2]))
                    grp = []
        return outd

    def convert_cart_to_polar_sde(in_dict):
        out = {}
        for f, vals in in_dict.items():
            out[f] = []
            for v in vals:
                x, y, z = v[1], v[2], v[3]
                az = np.arctan2(y, x) * 180 / np.pi
                el = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                row = [v[0], az, el] + ([v[4]] if len(v) > 4 else [])
                out[f].append(row)
        return out

    def write_output_format_file_sde(path, out_dict):
        # 【監査対応v2】dist付きは6列 [frame,class,track=0,az,el,dist] で書く。
        # 旧5列だと自前ローダのlen==5分岐([frame,class,track,az,el])に化けて
        # az←el, el←dist と誤読される往復事故があった
        with open(path, "w") as fid:
            for f in out_dict.keys():
                for v in out_dict[f]:
                    if len(v) > 3:
                        fid.write(f"{int(f)},{int(v[0])},0,{int(v[1])},"
                                  f"{int(v[2])},{float(v[3]):.2f}\n")
                    else:
                        fid.write(f"{int(f)},{int(v[0])},{int(v[1])},"
                                  f"{int(v[2])}\n")

    for m in (du, mm):
        m.get_multi_accdoa_labels = get_multi_accdoa_labels_sde
        m.multi_accdoa_to_dcase_format = multi_accdoa_to_dcase_format_sde
        m.convert_output_format_cartesian_to_polar = convert_cart_to_polar_sde
    du.write_output_format_file = write_output_format_file_sde
    import models.model_module as top_mm
    top_mm.write_output_format_file = write_output_format_file_sde

    print(f"[SDE] train patches applied (W_DIST={W_DIST}, "
          f"DIST_SCALE={DIST_SCALE})")
