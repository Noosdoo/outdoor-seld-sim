# outdoor_seld_e2e 進行メモ（作業状態の記録・セッション復帰用）

目的: DynamicSound → FOA → DCASEラベル → PSELDNets 学習 → 評価 の1クリップ貫通。

## 確定した環境事実（2026-07-10 調査済み）
- PSELDNets: `C:\Users\satos\research\PSELDNet\PSELDNets`（venv `.venv`, py3.11.8, torch2.2.1+cpu, GPUなし）
- DynamicSound: `C:\Users\satos\research\dynamic-sound`（venv `.venv`, dynamic-sound 1.0.3, numpy2.4.6,
  librosa, matplotlib, soundfile, **pyroomacoustics 0.7.4 を今回追加**）
- ckpt: `PSELDNets/ckpts/mACCDOA-HTSAT-0.567.ckpt`（140,516,864 bytes, HFから今回DL）

## 確定した規約（PSELDNets 実コードで確認済み）
- FOA: **チャンネル順 W,Y,Z,X（ACN）/ SN3D**。根拠 `src/data/data.py generate_spatial_samples`
  （W=p, ch1=y·p, ch2=z·p, ch3=x·p）＝論文Eq.(3)
- 座標: x=前, y=左, z=上。az=atan2(y,x)[deg] 反時計回り正、el=atan2(z,√(x²+y²))[deg]
  根拠 `src/preproc/preprocess.py:600-601`、DOA復元 `(cos el cos az, cos el sin az, sin el)`
- メタデータCSV: ヘッダなし5列 `frame,class,track,azimuth(int),elevation(int)`、0.1s/frame
  根拠 `src/utils/data_utilities.py load_output_format_file`
- 合成データセット構造: `datasets/<name>/foa/*.flac` + `datasets/<name>/metadata/*.csv`
  + `datasets/cls_indices_train.tsv`（親に置く、3列目=クラス名、行順=クラスID）
  根拠 `src/utils/datasets.py Synthesis`。dataset_dict にない名前は自動で synth 扱い
- 音声: **24kHz 4ch FLAC**（synth系は index csv 内で .wav→.flac 置換されるため flac 必須）
- ファイル名に `fold0_room0_` の形のroomトークン必須（rooms フィルタは部分文字列マッチ）
- パス中に `foa` という語はフォルダ名1箇所のみ（valid GTは文字列replace `foa`→`metadata`）
- preproc: `python src/preproc.py dataset=<name>`（cwd=リポジトリ直下）→ `_hdf5/` にラベルh5+index csv
- **Windows罠**: index csv のパスは `\` 区切りで書かれるが `data.py` は `path.split('/')[-3]`
  → preproc 後に index csv を `/` に正規化する後処理が必要（リポジトリ無改変で回避）
- 学習: `python src/train.py experiment=...`。要上書き: `trainer.accelerator=cpu compile=false
  num_workers=0 model.batch_size=2`。SELD-ckptロードは tscam_conv/head を自動スキップ
  （`src/models/accdoa.py load_ckpts(audioset_pretrain=False)`）→ 任意クラス数でOK
- 検証メトリクスは on_validation_epoch_end で val/macro & val/micro の ER/F/LE/LR/SELD_scr を自動出力

## DynamicSound 実装事実（gitソース確認済み）
- `attenuations.geometric = 1/distance` → **基準距離 r0=1m 固定**（1mでゲイン1）
- 大気吸収: ISO9613-1 係数×距離 → firwin2 513タップFIR（毎サンプル設計）
  → **線形位相FIRの群遅延 256サンプル分、物理遅延に上乗せされる**（48kHzで+5.3ms）
- `_compute_emission`: 論文Eq.12-13の2次方程式（区分等速セグメント毎）。staticメソッドなので照合可能
- `AudioFile(loop=True)` が既定 → **loop=False 必須**。get_sample は線形補間
- te < path開始時刻 は None → 到達前は無音（安全）
- 出力: int32 WAV、±1.0クリップ
- `Microphone(file_path, sample_rate)` = 原点無指向1ch。Path=[[t,x,y,z,qw,qx,qy,qz],...]

## 設計決定
- シーン: マイク(0,0,1.5)静止、サイレン音源 (-50,5,1)→(+50,5,1) 10m/s 10秒、T=20°C/1atm/50%RH
- 48kHzシミュ→scipy resample_poly で24kHzデシメート→FOAエンコード（24kタイムラインでDOA計算）
- 地面反射: z→−z 鏡映軌道の別シミュ（直接音と別々にFOA化して加算）。貫通は直接音のみ版
- クラス体系: SELD-Data-Generator の cls_indices.tsv（屋外10クラス）を流用 → **Siren = class 4**
- データセット名: `outdoor1clip` / ファイル名: `fold0_room0_mix001.flac`
- PSELDNets へは新規ファイル追加のみ（既存yaml編集なし）:
  `configs/data/outdoor1clip.yaml` + `configs/experiment/outdoor_e2e_1clip.yaml`

## 進行状態（2026-07-11 全ステップ完了）
- [x] 調査完了・ckpt取得・pyroomacoustics導入
- [x] srcモジュール＋単体テスト25項目PASS（放射時刻はDynamicSound内部と誤差ゼロ一致）
- [x] Step 0: (a)遅延誤差-0.8ms (b)ISO±1dB@≤16kHz (c)因果性+4ms (d)NormMUSIC 0.31°
      (e)ドップラー誤差≤0.010% → out/step0/step0_report.md
- [x] Step 1: direct/mirror 各モノラル48kHz生成（各約6分）
- [x] Step 2: FOA 2版（direct: peak0.179 / withrefl: peak0.319）24kHz FLAC
- [x] Step 3: ラベル99フレーム（frame1..99, Siren=4, az174→6°, el-6..-1°）
- [x] Step 4: preproc→index修正→学習15ep（train loss 0.0329→0.0204 単調減少）
      → val ER 0.400 / F 66.7% / LE 10.5° / LR 80.0% / SELD 0.248
- [x] Step 5: サニティ1 PASS（Y/W=1.0, X=Z=0）、サニティ2 az中央誤差0.31°/最大0.96°
      図: out/figures/doa_labels_vs_iv.png, spectrogram_direct_vs_refl.png

実行コマンドの正は README.md を参照。
