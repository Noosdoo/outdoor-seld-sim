# -*- coding: utf-8 -*-
"""入力の倍率（本体 §3-2・変更票 #75 ②・2026-10-26）: PSELDNets の特徴量の直前 1 か所（`BaseModelModule.standardize(batch_x)`・`af_extractor` の直前・
波形 4ch）で `batch_x * 10^(g/20)` を 1 回だけ掛ける。g = 環境変数 `SDE_INPUT_GAIN_DB`（既定 0 = 何もしない＝旧と bit 一致）。

通る道（全部 standardize を通る）: 学習 `src/models/model_module.py:58`（_train_sde17・_train_causal17）・因果推論 `_causal_infer_v17.py`・長尺 `_causal_infer_long18.py`・
scores `_causal_infer_scores.py`（実録の推論 field_eval_infer も _causal_infer_v17 を呼ぶ）。各台本は `sde_patch.apply_train_patches()` の直後に `gain_patch.apply()`。
相手 D（`_baseline_D_pretrain170_infer.py`）は自分の `BASELINE_GAIN_DB` を波形に掛ける別の道で、同じ standardize も通るので、**同じプロセスで
SDE_INPUT_GAIN_DB と BASELINE_GAIN_DB が両方 0 以外なら止める**（二重に掛けない）。感度の道具 `_gain_scale_foa.py` はファイルを書き換える型で別（併用しない）。

記録（来歴の約束・Fable 10/26 16:2x）: **環境変数 SDE_INPUT_GAIN_DB が明示されていれば 0 でも書く・未設定なら書かない**（v30 の鎖は倍率の決定が 0 に
なりうるので「鍵が無い = 旧の run」と「v30 で 0 を選んだ」を区別する。旧の run は未設定の経路なので identity は変わらない。v30 の sbatch は必ず明示する）。
ckpt に `sde_input_gain_db`（on_save_checkpoint で書く・無い ckpt は 0 と読む）。推論は ckpt の値と環境の値を照合し、違えば止める（`check_ckpt(ck)`）。
run_meta の settings.sde に `input_gain_db`。倍率 0 を明示したときは記録の包みだけで standardize は包まない（計算は旧と bit 一致）。
"""
from __future__ import annotations

import os

GAIN_DB = float(os.environ.get("SDE_INPUT_GAIN_DB", "0") or 0.0)
EXPLICIT = os.environ.get("SDE_INPUT_GAIN_DB", "").strip() != ""   # 明示（0 でも）= 来歴を書く・未設定（空も）= 旧のまま
_APPLIED = {"on": False}


def _baseline_gain_db() -> float:
    return float(os.environ.get("BASELINE_GAIN_DB", "0") or 0.0)


def guard_double() -> None:
    """主の倍率と相手 D の倍率を同じプロセスで重ねない。"""
    if GAIN_DB != 0.0 and _baseline_gain_db() != 0.0:
        raise RuntimeError(f"SDE_INPUT_GAIN_DB={GAIN_DB:g} と BASELINE_GAIN_DB={_baseline_gain_db():g} が両方 0 以外（二重に掛ける）。"
                           "主の倍率は相手 D に掛けない（D は自分の 40 dB のまま・本体 §3-2）")


def apply() -> bool:
    """standardize を包む（GAIN_DB = 0 なら包まない＝計算は旧と bit 一致）。明示されていれば ckpt に倍率を書く on_save_checkpoint を包む（0 でも）。
    未設定なら何もしない。返り値 = 倍率を掛けたか。"""
    guard_double()
    if not EXPLICIT:
        return False
    from models.components.model_module import BaseModelModule
    if not getattr(BaseModelModule.on_save_checkpoint, "_gain_recorded", False):
        orig_save = BaseModelModule.on_save_checkpoint

        def on_save_checkpoint(self, checkpoint):
            out = orig_save(self, checkpoint)
            checkpoint["sde_input_gain_db"] = GAIN_DB
            return out
        on_save_checkpoint._gain_recorded = True
        BaseModelModule.on_save_checkpoint = on_save_checkpoint
    if GAIN_DB == 0.0:
        print("[gain_patch] 入力の倍率 0 dB（明示）: 掛けない・ckpt と run_meta に 0 を記録", flush=True)
        return False
    if _APPLIED["on"]:
        return True
    orig = BaseModelModule.standardize
    g = 10.0 ** (GAIN_DB / 20.0)

    def standardize(self, batch_x):
        return orig(self, batch_x * g)
    standardize._gain_db = GAIN_DB
    BaseModelModule.standardize = standardize
    _APPLIED["on"] = True
    print(f"[gain_patch] 入力の倍率 {GAIN_DB:+g} dB（×{g:.4g}）を standardize の直前に掛ける", flush=True)
    return True


def check_ckpt(ck: dict) -> float:
    """推論: ckpt の sde_input_gain_db（無ければ 0 = 旧の学習）と環境の SDE_INPUT_GAIN_DB が同じことを確かめる（違えば止める）。返り値 = 値。"""
    v = float(ck.get("sde_input_gain_db", 0.0) or 0.0)
    if abs(v - GAIN_DB) > 1e-9:
        raise RuntimeError(f"ckpt の入力の倍率 {v:g} dB と SDE_INPUT_GAIN_DB={GAIN_DB:g} が違う（学習と推論で倍率をそろえる・本体 §3-2）")
    return v


def meta_settings(sde: dict) -> dict:
    """run_meta の settings.sde に input_gain_db を足す（明示されていれば 0 でも・未設定なら足さない＝旧の run の identity は変えない）。"""
    if EXPLICIT:
        sde = dict(sde, input_gain_db=GAIN_DB)
    return sde
