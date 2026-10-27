# -*- coding: utf-8 -*-
"""④ハイパラ感度測定の experiment yaml を生成する（2026-09-02）。

基準= configs/experiment/outdoor_siren_v12.yaml（batch8 / lr3e-4 / StepLR(60,0.1) / 100ep）。
armごとに **差分だけ** 変えた outdoor_siren_v12_hp_<arm>.yaml を書く。
宣言= md/design/再学習④_ハイパラ感度_方針宣言_2026-09-02.md §2。

実行場所: PSELDNets リポジトリ直下（configs/ がある所）
    .venv/bin/python ~/research/outdoor_seld_e2e/server_sde/hp_make_configs.py
"""
import os
import sys

assert os.path.isdir("configs/experiment"), "PSELDNets 直下で実行してください"

BASE = """# @package _global_
# ④ハイパラ感度 arm={arm}: {note}
defaults:
 - override /data: outdoor_siren_v12.yaml
 - override /loss: multi_accdoa.yaml
 - _self_

task_name: outdoor_siren_v12

model:
  batch_size: {batch}
  kwargs:
    pretrained_path: {pretrained}
    audioset_pretrain: false
  optimizer:
    kwargs: {{lr: {lr}}}
  lr_scheduler:
    method: {sched}
    kwargs: {sched_kw}

trainer:
  max_epochs: {epochs}
  check_val_every_n_epoch: 5
"""

OFFICIAL = "ckpts/mACCDOA-HTSAT-0.567.ckpt"
D = dict(batch=8, pretrained=OFFICIAL, lr=0.0003, sched="StepLR",
         sched_kw="{step_size: 60, gamma: 0.1}", epochs=100)

# Stage B（基準学習）
ARMS = {
    "lr1e-4": dict(D, lr=0.0001, note="lr 1e-4（StepLR60 → 1e-5）"),
    "cos": dict(D, sched="CosineAnnealingLR", sched_kw="{T_max: 100, eta_min: 3.0e-6}",
                note="CosineAnnealingLR(T_max=100, eta_min=3e-6)"),
    "2step": dict(D, sched="MultiStepLR", sched_kw="{milestones: [40, 70], gamma: 0.1}",
                  note="MultiStepLR [40,70] = 3e-4→3e-5→3e-6"),
    "short": dict(D, epochs=40, sched_kw="{step_size: 20, gamma: 0.1}",
                  note="40ep・StepLR(20)（100epは無駄か）"),
    "bs32": dict(D, batch=32, note="batch 32（lrそのまま）"),
    # 公式ckptから直接（sbatch側で SDE_INIT_CKPT を空にする＝ヘッド乱数初期化）
    "scratch": dict(D, note="warm-start連鎖なし・公式ckptから100ep"),
    # Stage C（因果ft）— pretrained_path は sbatch 側で w3 ep099 に上書きする
    "cseed": dict(D, note="因果ft ft2同一レシピ・seed違い（フロア）"),
    "clr3e-5": dict(D, lr=0.00003, note="因果ft lr 3e-5"),
    "clr1e-4": dict(D, lr=0.0001, note="因果ft lr 1e-4"),
    "crep8": dict(D, epochs=25, sched_kw="{step_size: 15, gamma: 0.1}",
                  note="因果ft REPEAT=8・25ep・StepLR(15)（総サンプル数=REPEAT2×100ep）"),
    # 追試（宣言§7・2026-09-03）
    "clr1e-4s2": dict(D, lr=0.0001, note="因果ft lr1e-4 の別seed再現（seedはsbatch側）"),
    "clr1e-4r8": dict(D, lr=0.0001, epochs=25, sched_kw="{step_size: 15, gamma: 0.1}",
                      note="因果ft lr1e-4 × REPEAT8・25ep・StepLR(15)"),
    "clr1e-4e150": dict(D, lr=0.0001, epochs=150, sched_kw="{step_size: 90, gamma: 0.1}",
                        note="因果ft lr1e-4 × 150ep・StepLR(90)"),
    # D21 の対照つき追加学習 (b)（2026-10-01 本人「(b)〜(c) 今やっちゃいましょう」）: dev2 の因果 ft e149 から同じ種・同じエポック数で 30 ep 続ける（更新回数は 25,500 本 153,030 対 24,000 本 144,030 で 6.25% 違う・Astra 再監査 10/6 RF03）。
    # 25,500 本（D21 あり）と 24,000 本（D21 なし・対照）の 2 本を同じ設定で走らせ、差を D21 のデータの分だけにする。lr 1e-4 固定（30 ep では StepLR(90) は効かない）
    "clr1e-4e030": dict(D, lr=0.0001, epochs=30, sched_kw="{step_size: 90, gamma: 0.1}",
                        note="因果ft の続き lr1e-4 × 30ep（D21 対照実験 (b)・pretrained_path は sbatch 側で dev2 e149 に上書き）"),
    # (b) のやり直し（2026-10-01 22:10）: 本番の因果 ft は StepLR(90, 0.1) で e90〜149 は lr 1e-5。e149 から 1e-4 で再加熱すると両腕とも崩れた
    # （交差 val の前後取り違え 0.03% → 1〜2%・方位 RMSE 10° → 26〜32°）ので、続きは e149 時点の lr 1e-5 で行う
    "clr1e-5e030": dict(D, lr=0.00001, epochs=30, sched_kw="{step_size: 90, gamma: 0.1}",
                        note="因果ft の続き lr1e-5 × 30ep（D21 対照実験 (b) のやり直し・e149 時点の lr を保つ）"),
    # (b) の更新回数を揃えるやり直し（2026-10-06 本人「やり直しましょう」・Astra 再監査 RF03）: noD21 腕（学習セグメント 19,200）を 32 ep にして
    # 更新回数 153,632 ≈ D21 腕 30 ep の 153,030（差 +0.39%）。lr・StepLR は clr1e-5e030 と同じ
    "clr1e-5e032": dict(D, lr=0.00001, epochs=32, sched_kw="{step_size: 90, gamma: 0.1}",
                        note="因果ft の続き lr1e-5 × 32ep（D21 対照実験 (b)・更新回数を D21 腕に揃える）"),
}


def main():
    want = sys.argv[1:] or list(ARMS)
    for arm in want:
        cfg = ARMS[arm]
        path = f"configs/experiment/outdoor_siren_v12_hp_{arm}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write(BASE.format(arm=arm, **cfg))
        print("wrote", path)


if __name__ == "__main__":
    main()
