# outdoor_seld_e2e — 屋外SELD 最小エンドツーエンドパイプライン

DynamicSound（屋外物理シミュレータ）→ FOA 変換 → DCASE 形式ラベル → PSELDNets
ファインチューニング → SELD 評価指標、を **1クリップで貫通** するパイプライン。

```
DynamicSound で移動音源クリップ生成（物理: 伝搬遅延/ドップラー/1/r/大気吸収）
  → 解析エンコードで FOA 4ch (W,Y,Z,X / SN3D, 24 kHz FLAC)
  → DCASE メタデータ CSV (frame,class,track,az,el @0.1 s)
  → PSELDNets (HTS-AT + mACCDOA) の学習が回り ER/F/LE/LR が出る
```

## 実行環境（2026-07 時点で確認済み）

| 役割 | venv | 主要パッケージ |
|---|---|---|
| 生成・検証 | `research/dynamic-sound/.venv` | dynamic-sound 1.0.3, numpy 2.4, librosa, matplotlib, pyroomacoustics 0.7.4 |
| 学習・評価 | `research/PSELDNet/PSELDNets/.venv` | torch 2.2.1+cpu, lightning 2.2.1, hydra, h5py |

チェックポイント: `PSELDNets/ckpts/mACCDOA-HTSAT-0.567.ckpt`
（HuggingFace `Jinbo-HU/PSELDNets` の `model/` から取得、140,516,864 bytes）

## 実行手順（全体で30〜40分、大半はシミュレーションと学習）

```powershell
# 変数（PowerShell）
$GEN = "C:\Users\satos\research\dynamic-sound\.venv\Scripts\python.exe"
$TRN = "C:\Users\satos\research\PSELDNet\PSELDNets\.venv\Scripts\python.exe"
cd C:\Users\satos\research\outdoor_seld_e2e

# Step 0: DynamicSound 検証（各10〜15分、独立に実行可。レポート: out/step0/step0_report.md）
& $GEN scripts/step0_validate.py abce
& $GEN scripts/step0_validate.py d
& $GEN scripts/step0_validate.py report

# 単体テスト（座標・放射時刻・FOA規約。数秒）
& $GEN tests/test_geometry.py
& $GEN tests/test_foa.py

# Step 1: クリップ生成（直接音＋鏡像音源の2シミュ、計12分程度）
& $GEN scripts/step1_generate_clip.py

# Step 2-3: FOA 変換とラベル生成（数秒）
& $GEN scripts/step2_encode_foa.py
& $GEN scripts/step3_make_labels.py

# Step 4: PSELDNets へ配置 → 前処理 → index パス修正 → 学習（CPUで約1分）
& $GEN scripts/step4_place_dataset.py
cd C:\Users\satos\research\PSELDNet\PSELDNets
& .\.venv\Scripts\python.exe src/preproc.py dataset=outdoor1clip
cd C:\Users\satos\research\outdoor_seld_e2e
& $GEN scripts/step4_place_dataset.py --fix-index
cd C:\Users\satos\research\PSELDNet\PSELDNets
& .\.venv\Scripts\python.exe src/train.py experiment=outdoor_e2e_1clip
# → logs/outdoor_e2e_1clip/runs/<日時>/train.log に val/macro: ER, F, LE, LR, SELD_scr

# Step 5: サニティチェックと可視化（1分程度）
cd C:\Users\satos\research\outdoor_seld_e2e
& $GEN scripts/step5_sanity.py all
```

### バリアント: ピーポー型サイレン（`--peepo`）

Step 1〜5 の各スクリプトに `--peepo` を付けると、うねり型（wail）の代わりに
「ピーポー」型（960/770 Hz を各0.65秒で交互、日本の救急車）で同じシーンを生成する。
音程が階段状なのでドップラーが最も見やすい。出力は `out/clip_peepo/`・`out/figures_peepo/`
に分離され、wail 版の成果物は残る。PSELDNets 側の配置先は同じ
`datasets/outdoor1clip/`（＝上書き。最後に配置した版が学習対象になる）。

```powershell
& $GEN scripts/step1_generate_clip.py --peepo
& $GEN scripts/step2_encode_foa.py --peepo
& $GEN scripts/step3_make_labels.py --peepo
& $GEN scripts/step4_place_dataset.py --peepo
# 以降の preproc / --fix-index / train / step5 は wail 版と同じ（step5 は --peepo 付き）
```

## 貫通結果（2026-07-11、1クリップ過学習デモ・数値は参考値）

- wail版: train loss 0.0329 → 0.0204（15エポック単調減少、1エポック=1ステップ）。
  検証指標（epoch 5→10→15）: ER 1.000→0.800→**0.400**、F −→26.7→**66.7%**、
  LE −→21.8→**10.5°**、LR −→50.0→**80.0%**、SELD_scr 1.000→0.539→**0.248**
- peepo版: train loss 0.0329 → 0.0226（単調減少）。検証指標（epoch 15）:
  **ER 0.000、F 100.0%、LE 10.8°、LR 100.0%、SELD_scr 0.015**
  （E_SELD = ¼[ER + (1−F) + LE/180 + (1−LR)] の検算と一致）
- 音とラベルの整合（サニティチェック2）: インテンシティベクトル法DOA vs ラベル
  **方位角 中央絶対誤差 0.31°（最大 0.96°）、仰角 0.26°**
- FOA 規約（サニティチェック1）: az=+90° 静止音源で Y/W=1.000000, X=Z=0（E2E確認）

## ディレクトリ

```
src/outdoor_seld/     geometry.py  座標変換・放射時刻ソルバ（音とラベルの唯一の共通経路）
                      foa.py       解析FOAエンコーダ(W,Y,Z,X/SN3D)＋IV法DOA推定
                      labels.py    DCASE 5列CSV生成（放射時刻補正済みDOA）
                      scene.py     シーン定義＋DynamicSoundラッパ＋デシメータ
                      siren.py     合成サイレン（wail 650-1450 Hz＋倍音）
tests/                単体テスト25項目（DynamicSound内部との照合含む）
scripts/              step0〜step5 実行スクリプト
out/step0/            検証レポート step0_report.md＋図＋step0_results.json
out/clip/             ドライ音源、モノラル2種、FOA 2種(direct/withrefl)、ラベルCSV、
                      scene_config.json（r0等）、clip_stats.json（RMS記録）
out/figures/          doa_labels_vs_iv.png（★最重要成果物）、spectrogram_direct_vs_refl.png
```

## PSELDNets 側に追加したもの（リポジトリ既存ファイルの編集はなし）

- `configs/data/outdoor1clip.yaml`, `configs/experiment/outdoor_e2e_1clip.yaml`（新規）
- `datasets/outdoor1clip/{foa,metadata}/fold0_room0_mix001.*`, `datasets/cls_indices_train.tsv`
- `ckpts/mACCDOA-HTSAT-0.567.ckpt`（HFからDL）
- 生成物: `_hdf5/`（preproc出力）, `logs/outdoor_e2e_1clip/`, `metrics.csv`（PSELDNets自体が
  検証時に cwd へ書く空ファイル。無害）

## 設計上の重要ポイント（規約はすべて PSELDNets 実コードで確認済み）

1. **FOA = W,Y,Z,X（ACN）/ SN3D**。`src/data/data.py generate_spatial_samples` と
   論文 Eq.(3) に一致。ゲインは DOA 単位ベクトル成分そのもの（W=p, Y=p·uy, Z=p·uz, X=p·ux）。
2. **座標**: x=前, y=左, z=上。az=atan2(y,x)（反時計回り正）, el=atan2(z,√(x²+y²))。
3. **DOA は「受信時刻の見かけの方向」**＝放射時刻 te を論文 Eq.(12)(13) の閉形式で解き
   ps(te) から計算。音（FOAゲイン）とラベルが geometry.py の同一経路を通る。
   自作ソルバは DynamicSound `_compute_emission` と**誤差ゼロで一致**（単体テスト）。
4. **地面反射**は鏡映軌道の第2音源を別シミュレーションし、各自の時変DOAで
   FOA 化してから加算（直接音と反射音が異なる到来方向を持つ物理的に正しいFOA）。
   貫通デモは直接音のみ版を使用。反射込み版は比較材料（`foa_withrefl_24k.flac`）。
5. **48 kHz でシミュレーション → 24 kHz へデシメート**（WAVソース遅延の線形補間による
   ドップラー伸縮アーティファクト対策。残留ゴーストは −60 dB 以下を確認）。
6. **r0 = 1 m は DynamicSound 実装に固定**（`attenuations.geometric = 1/distance`）。
   scene_config.json に記録。将来の条件間ラウドネス整合は dry 信号の RMS
   （clip_stats.json に記録）と source_gain_db で行う。

## 既知の注意点（ハマりどころ）

- **大気吸収FIR（513タップ）の群遅延**: 物理遅延に +256/fs 秒（48 kHz で 5.33 ms）
  上乗せされる。ラベル分解能 100 ms に対しては無視できる（Step 0 (a)(c) で定量確認）。
- **Windows パス**: PSELDNets の index CSV は `\` 区切りで書かれるが `data.py` が
  `path.split('/')` を使うため、`step4_place_dataset.py --fix-index` で `/` に正規化する
  （リポジトリ無改変の回避策）。
- **`AudioFile(loop=True)` が既定**: ワンショット音源は `loop=False` 必須（scene.py で対応済み）。
- **ファイル名規約**: `fold0_room0_mix001.flac` のような room トークン必須、
  パス中に `foa` という語はフォルダ名1箇所のみ（valid GT が文字列置換で解決されるため）。
- 学習の Windows/CPU 対応: `compile=false num_workers=0 trainer.accelerator=cpu`
  ＋ logger は csv（tensorboard 未導入）。experiment yaml に設定済み。
- Step 0 (b) の 20 kHz 帯で理論(−178 dB)と乖離(+39 dB)があるのは 513 タップ FIR の
  阻止域実現限界（≈−140 dB）で、物理モデルの誤りではない。

## 次の一手（本パイプラインの上に載るもの）

1. シーン量産スクリプト化（クラス×軌道×速度×物理on/offのグリッド）→ 条件比較 ablation
2. 物理要素の on/off スイッチ（ドップラー: te 固定化 / 大気吸収: FIRバイパス /
   距離減衰: 正規化 / 地面反射: mirror 加算の有無）— mirror 加算は実装済み
3. FSD50K 実音源クラス（SELD-Data-Generator の10クラス辞書と互換）への差し替え
4. 実録屋外テストセットでの sim-to-real 評価
