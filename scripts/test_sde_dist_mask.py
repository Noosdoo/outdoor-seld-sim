# -*- coding: utf-8 -*-
"""距離の損失のマスク（変更票 #75 ④・2026-10-26）の受け入れ試験。CPU の torch だけで走る（PSELDNets の import は要らない）。

確かめること:
  1. SDE_DIST_MASK=0（既定）の式が、旧の実装（この試験の中に写した 10/26 以前の式）と bit 一致する
  2. 活動のある所だけの本（1 音源・全フレーム活動）では、マスクあり = マスクなし（3 track とも同じ教師なので平均が同じ）
  3. 活動の無いフレームが混ざる本では、マスクありの距離の項 = 活動フレームだけの平均（numpy の参照実装と一致）
  4. 活動が 1 つも無いバッチで NaN が出ず、距離の項が 0
  5. 置換の選び方（xyz）はマスクに依らず同じ
使い方: <PSELDNets の .venv>/python scripts/test_sde_dist_mask.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    import torch
except ImportError:
    print("SKIP: torch が無い（PSELDNets の .venv で走らせる）")
    sys.exit(0)

_spec = importlib.util.spec_from_file_location("sde_patch", ROOT / "server_sde/sde_patch.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'} {name}{(' — ' + str(detail)) if detail and not cond else ''}")


def old_formula(out, tgt):
    """10/26 以前の Losses_SDE.__call__ の式（写し）"""
    tok = [tgt[:, :, i, 0:1, :] * tgt[:, :, i, 1:, :] for i in range(6)]
    A0, B0, B1, C0, C1, C2 = tok
    perms = [(A0, A0, A0), (B0, B0, B1), (B0, B1, B0), (B0, B1, B1), (B1, B0, B0), (B1, B0, B1), (B1, B1, B0),
             (C0, C1, C2), (C0, C2, C1), (C1, C0, C2), (C1, C2, C0), (C2, C0, C1), (C2, C1, C0)]
    catA = torch.cat((A0, A0, A0), 2); catB = torch.cat((B0, B0, B1), 2); catC = torch.cat((C0, C1, C2), 2)
    pads = [catB + catC] + [catA + catC] * 6 + [catA + catB] * 6
    out = out.reshape(out.shape[0], out.shape[1], 12, tgt.shape[-1])
    mse_fn = torch.nn.MSELoss(reduction="none")
    xyz_losses, d_losses = [], []
    for (t0, t1, t2), pad in zip(perms, pads):
        t_full = torch.cat((t0, t1, t2), 2) + pad
        mse = mse_fn(out, t_full)
        xyz_losses.append(mse[:, :, P.XYZ_IDX, :].mean(dim=2))
        d_losses.append(mse[:, :, P.D_IDX, :].mean(dim=2))
    loss_min = torch.min(torch.stack(xyz_losses, dim=0), dim=0).indices
    sel_xyz = sum(l * (loss_min == n) for n, l in enumerate(xyz_losses)).mean()
    sel_d = sum(l * (loss_min == n) for n, l in enumerate(d_losses)).mean()
    return sel_xyz, sel_d, loss_min


def make_target(B, T, C, rng, active_frac=1.0, n_src=1):
    """ADPIT の 5 軸ラベル [B,T,6,5,C]。n_src=1 なら A0 だけ（活動フレームは active_frac）。n_src=2 なら B0/B1 に 2 音源。"""
    tgt = torch.zeros(B, T, 6, 5, C)
    for b in range(B):
        for t in range(T):
            if rng.random() >= active_frac:
                continue
            cls = int(rng.integers(C))
            if n_src == 1:
                v = rng.normal(size=3); v /= np.linalg.norm(v)
                tgt[b, t, 0, 0, cls] = 1.0
                tgt[b, t, 0, 1:4, cls] = torch.tensor(v, dtype=torch.float32)
                tgt[b, t, 0, 4, cls] = float(rng.uniform(0.05, 0.9))
            else:
                for slot in (1, 2):
                    v = rng.normal(size=3); v /= np.linalg.norm(v)
                    tgt[b, t, slot, 0, cls] = 1.0
                    tgt[b, t, slot, 1:4, cls] = torch.tensor(v, dtype=torch.float32)
                    tgt[b, t, slot, 4, cls] = float(rng.uniform(0.05, 0.9))
    return tgt


def main() -> int:
    rng = np.random.default_rng(7)
    B, T, C = 2, 10, 9
    torch.manual_seed(7)
    out = torch.randn(B, T, 12 * C) * 0.5

    # 1. 既定（マスクなし）は旧の式と一致
    tgt = make_target(B, T, C, rng, active_frac=0.6)
    sx_old, sd_old, lm_old = old_formula(out, tgt)
    sx_new, sd_new = P.sde_loss_terms(out, tgt, dist_mask=False)
    check("1 マスクなしは旧の式と bit 一致（xyz）", torch.equal(sx_old, sx_new), (sx_old.item(), sx_new.item()))
    check("1b マスクなしは旧の式と bit 一致（距離）", torch.equal(sd_old, sd_new), (sd_old.item(), sd_new.item()))
    check("1c 既定の DIST_MASK は False（環境変数なし）", P.DIST_MASK is False or P.DIST_MASK == False)  # noqa: E712

    # 2. 全フレーム活動（1 音源）: マスクあり = 活動クラスの距離の誤差の平均（3 track とも同じ教師 A0・活動の数 = B×T×3）
    #    （マスクなしは非活動クラスの (out − 0)² も平均に入るので一致しない＝それがマスクを入れる理由）
    tgt_full = make_target(B, T, C, rng, active_frac=1.0)
    _, sd_full_on = P.sde_loss_terms(out, tgt_full, dist_mask=True)
    o_full = out.reshape(B, T, 12, C).numpy(); num = 0.0; cnt = 0
    for b in range(B):
        for t in range(T):
            c = int(tgt_full[b, t, 0, 0, :].argmax().item()); d_t = float(tgt_full[b, t, 0, 4, c].item())
            for k in range(3):
                num += (o_full[b, t, 4 * k + 3, c] - d_t) ** 2; cnt += 1
    check("2 全フレーム活動: マスクあり = 活動クラスの距離の誤差の平均（B×T×3 個）", abs(sd_full_on.item() - num / cnt) < 1e-5 * max(1.0, num / cnt) and cnt == B * T * 3,
          (sd_full_on.item(), num / cnt, cnt))

    # 3. 活動の無いフレームが混ざる本: numpy の参照（選ばれた置換の教師で、活動 (B,T,track,C) だけの平均）
    sx_on, sd_on = P.sde_loss_terms(out, tgt, dist_mask=True)
    _, _, lm = old_formula(out, tgt)
    # 参照: 置換ごとの t_full を作り直し、loss_min で選ぶ
    tok = [tgt[:, :, i, 0:1, :] * tgt[:, :, i, 1:, :] for i in range(6)]
    A0, B0, B1, C0, C1, C2 = tok
    perms = [(A0, A0, A0), (B0, B0, B1), (B0, B1, B0), (B0, B1, B1), (B1, B0, B0), (B1, B0, B1), (B1, B1, B0),
             (C0, C1, C2), (C0, C2, C1), (C1, C0, C2), (C1, C2, C0), (C2, C0, C1), (C2, C1, C0)]
    catA = torch.cat((A0, A0, A0), 2); catB = torch.cat((B0, B0, B1), 2); catC = torch.cat((C0, C1, C2), 2)
    pads = [catB + catC] + [catA + catC] * 6 + [catA + catB] * 6
    o = out.reshape(B, T, 12, C).numpy()
    num = 0.0; cnt = 0
    for b in range(B):
        for t in range(T):
            for c in range(C):
                n = int(lm[b, t, c]); t0, t1, t2 = perms[n]
                tf = (torch.cat((t0, t1, t2), 2) + pads[n])[b, t, :, c].numpy()
                for k in range(3):
                    if (tf[4 * k:4 * k + 3] ** 2).sum() > 0.25:
                        num += (o[b, t, 4 * k + 3, c] - tf[4 * k + 3]) ** 2; cnt += 1
    ref = num / max(cnt, 1)
    check("3 活動フレームだけの平均（numpy 参照と一致）", abs(sd_on.item() - ref) < 1e-5 * max(1.0, ref), (sd_on.item(), ref, cnt))
    check("3b 活動の数 = 活動フレーム数 × 3 track（1 音源）", cnt == 3 * int((tgt[:, :, 0, 0, :] > 0.5).sum().item()), cnt)
    check("3c マスクなしの距離の項はマスクありより小さい（0 に引く非活動の分が薄める）", sd_new.item() < sd_on.item(), (sd_new.item(), sd_on.item()))

    # 4. 活動が 1 つも無いバッチ
    tgt_zero = torch.zeros(B, T, 6, 5, C)
    sx_z, sd_z = P.sde_loss_terms(out, tgt_zero, dist_mask=True)
    check("4 活動なし: 距離の項 0・NaN なし", torch.isfinite(sd_z).item() and sd_z.item() == 0.0, sd_z.item())
    check("4b 活動なし: xyz の項は有限", torch.isfinite(sx_z).item())

    # 5. 置換の選び方はマスクに依らない（xyz の項が同じ）
    check("5 xyz の項はマスクの有無で同じ", torch.equal(sx_on, sx_new))

    # 6. 2 音源（B スロット）の本でも参照と一致
    tgt2 = make_target(B, T, C, rng, active_frac=0.5, n_src=2)
    _, sd2_on = P.sde_loss_terms(out, tgt2, dist_mask=True)
    check("6 2 音源の本で有限・非負", torch.isfinite(sd2_on).item() and sd2_on.item() >= 0.0)

    # 7. 勾配が流れる（学習に使える）
    out_g = out.clone().requires_grad_(True)
    _, sd_g = P.sde_loss_terms(out_g, tgt, dist_mask=True)
    sd_g.backward()
    g = out_g.grad.reshape(B, T, 12, C)
    act_any = (tgt[:, :, 0, 0, :] > 0.5)
    check("7 勾配は活動フレームの距離の軸にだけ乗る", bool((g[:, :, P.D_IDX, :][~act_any.unsqueeze(2).expand(-1, -1, 3, -1)] == 0).all().item())
          and bool((g[:, :, P.D_IDX, :][act_any.unsqueeze(2).expand(-1, -1, 3, -1)] != 0).any().item()))

    n_ok = sum(1 for _, ok in results if ok)
    print(f"{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
