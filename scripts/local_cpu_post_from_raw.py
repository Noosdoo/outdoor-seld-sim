"""PC の推論（scripts/local_cpu_infer.sh）が残した生の出力（raw_<出力名>/<clip>.npy）から、閾値を変えた予測（7 列・val_all_causal.csv と同じ形）を、
推論の台本（server_sde/_causal_infer_v17.py の 207〜229 行）と同じ関数で作る（2026-10-25 Opus・切り分け専用）。
  sde_patch.apply_train_patches → du.get_multi_accdoa_labels（入り）→ hysteresis_patch.apply（保持 < 入りのとき）→ multi_accdoa_to_dcase_format → 極座標 → 書き出し
使い方: <PSELDNets の .venv の python> scripts/local_cpu_post_from_raw.py <raw の置き場> <出力の csv> <入り> <保持>
  （PS は環境変数 PS か、既定の out/_local_cpu_infer/PS＝local_cpu_infer.sh が作るサーバの configs と PC の src の置き場）
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

E2E = Path(__file__).resolve().parents[1]
PS = Path(os.environ.get("PS", E2E / "out" / "_local_cpu_infer" / "PS"))
HERE = E2E / "server_sde"
os.environ.setdefault("SDE_W_DIST", "3.0")
os.environ.setdefault("SDE_DIST_MODE", "log")
sys.path.insert(0, str(PS / "src"))
sys.path.insert(0, str(PS))
sys.path.insert(0, str(HERE))
import sde_patch  # noqa: E402

sde_patch.apply_train_patches()
import utils.data_utilities as du  # noqa: E402
import hysteresis_patch  # noqa: E402

ASSOC = 20.0
NCLS = 9


def main():
    raw, out, th, stay = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    lines = []
    for p in sorted(raw.glob("*.npy")):
        causal = torch.from_numpy(np.load(p))
        sed, doa = du.get_multi_accdoa_labels(causal, NCLS, th)
        sed_np, doa_np = sed[:, 0].numpy(), doa[:, 0].numpy()
        if stay < th:
            sed_np = hysteresis_patch.apply(doa_np, NCLS, th, stay, ASSOC)
        dcase = du.multi_accdoa_to_dcase_format(sed_np, doa_np, nb_classes=NCLS)
        polar = du.convert_output_format_cartesian_to_polar(in_dict=dcase)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "x.csv"
            du.write_output_format_file(str(tmp), polar)
            for l in tmp.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    lines.append(f"{p.stem},{l.strip()}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:      # 公式と同じ LF（Windows の write_text は CRLF になり、中身は同じでも cmp が違うと言う・10/25）
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"{len(lines)} 行 → {out}（入り {th}・保持 {stay}）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
