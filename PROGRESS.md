# outdoor_seld_e2e 進行メモ（作業状態の記録・セッション復帰用）

目的: DynamicSound → FOA → DCASEラベル → PSELDNets 学習 → 評価 の1クリップ貫通。

## ⚡ いま生きている結論の一覧（2026-07-15時点）

本文は時系列記録のため、古い結論が訂正注記付きで残っている。読み違え防止のため**この表が最新の正**。

| テーマ | 生きている結論 | 死んだ/訂正された結論 |
| --- | --- | --- |
| v5→v6の改善 | **学習量＋データ量の複合効果**（学習量を揃えるだけでv5でも0.052-0.093。データ量固有の寄与は0.05→0.03の残差。対照ラン#8で確定） | ~~データ量が主因~~ |
| クラス識別 | **本物**（v8=ジッタ＋側独立化後もsubstitution=0） | ~~同一波形・側手がかりの産物の疑い~~（v8で棄却） |
| 疎発音のDOAドリフト | v5規模の訓練では顕在化、訓練を尽くすとほぼ解消。**純音BackupBeepの初動方向（30%）だけが一貫した残課題** | ~~疎発音ほど単調に悪化~~（val依存・SNR交絡で棄却） |
| 拡散雑音 | valの低SNR2本では壊れなかった（n=2の観察） | ~~SNR 0dBでも壊せない~~（過大） |
| 反射 | 学習に含めれば性能コストほぼゼロ（仰角バイアス中央値-3°を補償） | — |
| スコアの絶対値 | 文献と比較不可（簡単な土俵）。主張は相対比較・解剖指標で | — |
| 実験の土俵 | **v8が現行の正**（train240/val80/test80、側独立・全クラスジッタ・直接音W基準）。v5-v7の数値との直接比較禁止 | ~~v7がablation基準~~ |
| 当事者指標 | イベント見逃し0%・誤通知1.35/分・初動方向35-80%（step8dで測定可能） | — |

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

## バリアント: ピーポー型サイレン（2026-07-11 追加）
- `make_peepo_siren`（960/770 Hz 交互・各0.65s・15msランプで位相連続）を siren.py に追加
- 全ステップスクリプトに `--peepo` フラグ（出力先 out/clip_peepo/, out/figures_peepo/）
- PSELDNets への配置先は同じ datasets/outdoor1clip/（上書き＝最後に配置した版が学習対象）
- 動機: 音程が階段状なのでドップラーシフトが平行移動として最も見やすい（ゼミ説明用）
- 実行結果 (2026-07-11): 貫通成功。train loss 0.0329→0.0226、
  val ER 0.000 / F 100% / LE 10.8° / LR 100% / SELD_scr 0.015（epoch15）
- サニティ2は wail と同一（az中央0.31°/最大0.96°。軌道同一なので当然）
- スペクトログラム: 接近中ピー≈989Hz→後退≈933Hz の段差が明瞭（figures_peepo/）

## データセット v1: outdoor_siren_v1（2026-07-11 完成）
- train 30 / val 10（fold1_room1 / fold2_room1）、サイレン1クラス、条件乱数（シード20260711）
  側とサイレン型は交互割当で均衡。条件全記録 = work/*/scene.json + inspection.csv
- **高速レンダラ fastsim.py 導入**: DynamicSoundと同一物理のベクトル化実装。
  DynamicSound実出力6本との波形一致 rel_rms=3.2〜3.4e-5（約-90dB）で等価性検証済み
  （tests/test_fastsim.py、DS参照波形は work/mix001-003/mono_*_DSref.wav に保全）。
  速度 約330倍（1本2.6秒）。step6 は fastsim 既定、--dynamicsound で旧経路
- 全40本の自動検品 PASS（IV法DOA vs ラベル: az中央誤差0.22〜0.28°、最大1.4°）
- Colab用 zip: out/dataset_outdoor_siren_v1.zip（36.6MB、datasets/構成、tsv同梱）
- Colabノートブック: colab/PSELDNets_outdoor_siren_v1_Colab.ipynb（v3レシピ踏襲、
  Drive永続化＋固定experiment_name＋自動resume、batch_size=8、70ep）
- 次: ユーザーが zip を Drive の MyDrive/PSELDNets_data/ へ→ノートブック実行

## Colab学習結果 v1（2026-07-11 完走）
- T4・70ep・約15分・拡張なし（=論文の「Fine-tune ✗Aug」行に対応）
- val（未見10本）: **ER 0.000 / F 100.0% / LE 1.8° / LR 100.0% / SELD_scr 0.003**
  推移: LE 7.3(ep5)→6.6→5.5→3.3→…→1.8°、train loss 0.0309→0.0017
- **天井効果を確認**: 単一クラス・雑音/残響ゼロ・同分布valでは飽和する
  （IV法物理ベースライン0.3°のデータなので妥当。バグ・リークは検証済みで否定）
- 含意: **ablationで差を出すには難易度軸が必要**。第一候補=雑音/SNR軸
  （ユーザーの6月分析「住宅街では距離・SNRが効く」と整合、軸3（見逃し率）にも直結）
- ckpt/ログは Drive: PSELDNets_logs/outdoor_siren_v1/runs/outdoor_siren_v1_run1/

## データセット v2: outdoor_siren_v2（2026-07-11 完成、難易度軸=雑音/SNR）
- **v1と同一の40シーン**（ジオメトリ乱数共通）＋拡散性ピンク雑音（SNR 0-20dB/クリップ、W基準）
- noise.py: 拡散FOA雑音（4ch独立・パワー比1/3）、20Hz未満遮断、SNR誤差<0.001dB（テスト8項目PASS）
- 検品: クリーン版でゲート（全40本PASS）＋実SNR独立測定（±0.5dB）＋雑音版IV誤差を記録
  → **SNR低下でIV誤差が単調増加**（19dB:0.31°→6dB:0.52°→0.2dB:1.14°）＝難易度軸が効く証拠
- クリーン版FOAは work/*/foa_clean_24k.flac に保全（雑音on/off比較は再生成不要）
- zip: out/dataset_outdoor_siren_v2.zip（82.6MB。雑音は圧縮が効かないためv1より大）
- **Colabノートブックv2はユーザーのDrive「Colab Notebooks」へ直接アップロード済み**
  （Drive MCP経由。URL: https://colab.research.google.com/drive/1Z1uaBmTUkN73fwvhA3SwNqo4224u82WX）
- 学習は毎回基盤ckptから（v1学習済み重みは流用しない＝実験間の初期条件統一）
- step6 は `--v1` でクリーン版の再生成も可能（既定はv2）

## データセット v3: outdoor_siren_v3（2026-07-11 完成、難易度軸の本命）
- **train 60 / val 20**。①サイレンは各クリップ2〜6秒だけ発音（スパース化＝約4〜6割が
  負例フレーム→誤検出・見逃しが初めて測れる、軸3の前提）②車の実録音
  （suv_dirt_road.wav 60s/48k、ランダム切り出し）がラベルなし指向性妨害として
  別軌道を通過（SIR 0-15dB、発音区間W基準）③拡散ピンク雑音（SNR 0-20dB、同基準）
- v2はモデルにほぼ攻略された（最終 ER 0.000/LE 3.4°。序盤は苦戦：ep5でER 0.39/LE 18.8°）
  → 「定常拡散雑音はトーン常時発音源を壊せない」という観察。v3の動機
  （⚠️2026-07-14訂正: v2 valでSNR 3dB未満は2本（0.19/1.6dB）のみ。「0dBでも壊せない」
  はn=2の観察のため「valの低SNR2本では壊れなかった」に格下げ。敵対的レビュー#7）
- 設計はユーザーと対話で確定（妨害音の当初案「合成サイレン」は同クラス無ラベル問題を
  説明して車実録に変更。マイクアレイ不要＝モノ音源を物理エンジンが空間化、を説明済み）
- 実装: labels.pyにsource_active_from追加、step6にvariant機構（--v1/--v2/既定v3）、
  発音窓掛け、妨害音レンダ＋SIR、区間基準SNR、ヘッドルーム保護（post_scale）、
  検品にSIR/SNR独立実測＋発音窓列、previewモード（図＋試聴wav）
- 全80本検品PASS。zip: out/dataset_outdoor_siren_v3.zip（165.5MB）
- ノートブック: Drive「Colab Notebooks」に配置済み
  https://colab.research.google.com/drive/13Ka8USmq-XV52T8dKceMYto39qpqciL7
- ⚠️ step6の既定variantはv3になった。v2再生成は `--v2` が必要

## Colab学習結果 v3（2026-07-11 完走）＝ものさし完成
- 最終(ep70): **ER 0.042 / F 96.4% / LE 7.3° / LR 100% / SELD_scr 0.030**（初めて収束後も誤り残存）
- 推移: ep5 ER 0.427/F 60.2/LE 18.6 → ep15 0.104/91.6/9.6 → 以降 0.042-0.094 で揺らぎつつ収束
- 途中 LR 96.9〜99% に低下＝**見逃しが実際に発生**。最終の残誤差はLR100%なので誤検出/20°外れ側
- v1→v2→v3 比較表（ゼミの主図）: ER 0/0/0.042、LE 1.8/3.4/7.3°、SELD 0.003/0.005/0.030
- 課題: 残誤差が小さめ（ダイナミックレンジ狭い）→ 対策は (a)SNR/SIR層別の報告 (b)複数シード
  (c)必要ならSIR負側拡張のv3.1（高速レンダラで数分）
- 次: infer mode=test でクリップ別予測CSV（Drive submissions/）→ MCP経由で取得し
  ER vs SNR/SIR の層別分析図を作る → ablation実験計画書＋ゼミ合意

## v3 誤り解剖（2026-07-11〜12 完了）
- `scripts/step7_error_anatomy.py`: Colab infer(mode=test)予測CSV（20クリップ）を
  埋め込みで受け取り、正解と突合してフレーム単位分類（miss/fa/dir_err>20°/ok）
- 結果（738正解フレーム）: miss6 / fa10 / dir_err21 / ok711
  → out/figures_v3_analysis/{summary.json, error_anatomy.png}
- **miss は6件全て発音区間の端(±0.3s)で発生**（オンセット/オフセット曖昧性、中間フレームの
  見逃しはゼロ）。fa は発音区間内外ほぼ半々、車方位との整合は10件中5件のみ20°以内
  （中央ずれ57.8°=誤通知は車誤認識だけでは説明不可）

## 新規性サーベイ再検証（2026-07-11、out/survey_novelty_2026-07.md）
- 6月の結論「どの物理が効くかの系統的ablationが空き地」は現在も有効。近傍新規参入2件
  （arXiv:2603.02508=物理ablation手法の他分野先行例、AudibleLight=屋内RIRベースで屋外非対応）
  はいずれも直接競合ではないが要ウォッチ
- 主張は「問い×領域×検証×応用指標」の交差点として言い回しを精密化する方針

## Ablation実験計画（2026-07-12 ドラフト、out/ablation_plan_2026-07.md）
- fastsim.render_mono の4物理要素（ドップラー/大気吸収/1/r/地面反射）をon/offする
  one-factor-at-a-time設計（4条件、フルグリッド16は保留）。データ再生成不要
  （同一シーンで音声だけ差し替え、ラベルは幾何のみで決まるため不変）
- 評価は条件ごとに step7_error_anatomy.py を回し、miss@edge比率・fa@car比率等を横並び
- **次: ゼミでスコープ合意 → Colabで4条件学習（各約15分）→ 誤り解剖を横並び表に**

## v3 ドローン混入版の破棄と復元（2026-07-12）
- ユーザーが車(suv_dirt_road.wav)＋ドローン(flying_drone.wav)混在の妨害音バリアントを試し、
  `dataset_outdoor_siren_v3/`を上書き→80本PASSしたが不採用と判断→
  `dataset_outdoor_siren_v3_BROKEN_carDrone_do_not_use/`にリネーム（車のみ版の音声/zipは消失、
  labelとscene.jsonのみ git commit 43050d1 に残存）
  **破棄の理由（2026-07-12 本人確認）**: 車とドローン(実録音)をクリップごとにランダムで
  混ぜると、結果がどちらの音源に起因するか切り分けられず土俵がぶれるため。
  → **設計原則**: 新しい音源要素を追加するときは、まずクリーン合成音源（siren.py方式）で
  試す。実録音は土俵の解釈可能性を崩すリスクがあるので後回し・慎重導入とする
- 現在の`scripts/step6_batch_scenes.py`はドローンコード無し・車のみに復帰済みだったため、
  `gen 0-79`を再実行するだけで**車のみ版を完全復元**（決定論的乱数のため metadata/inspection.csv は
  git版とバイト単位で一致、zip 165.5MBもPROGRESS記載値と一致）。BROKENフォルダとzipは削除済み
- git commit 済み（f5c2768「fix: v3データセットのdrone混入版を破棄し車のみ版を復元」）

## Colab run1のNaN崩壊とrun2再学習（2026-07-12）
- train.logを精査した結果、run1は7/11 11:04-11:17に正常完走（ER0.042/SELD0.030）していたが、
  **翌日7/12 10:23に同じexperiment_name(run1)で再度学習が走り、val ER 0.604/LR 40.6%に崩壊
  →直後にloss=nan**という事故が発覚。原因は自動resume機構（固定experiment_nameの
  `last.ckpt`があれば自動続行）と、その間にDrive上のzipをcar+drone版で上書きしたことの
  組み合わせ——**車のみで学習したckptに、途中からcar+drone分布のデータを食わせて再学習しようと
  して破綻**したと推定（step6のresumeロジックが原因ではなく、データとckptの世代不一致が原因）
- 対策: EXP_NAME を`outdoor_siren_v3_run2`に変更（1行の差分、ノートブック本体は無変更）して
  ゼロから再学習 → **正常完走、NaN再発なし**
- run2最終結果（epoch70）: **ER 0.042 / F 96.9% / LE 6.5° / LR 100% / SELD_scr 0.027**
  （run1の0.042/96.4%/7.3°/100%/0.030とほぼ同一→車のみ版復元の正しさを再現性で裏付け）
- 教訓: **固定experiment_name＋自動resumeは、Driveのzipを差し替えた後の再実行では危険**。
  データを変えて学習し直すときは必ずexperiment_nameも変える（このプロジェクトのColab運用の
  標準注意点として今後も適用）
- Google Drive上のtrain.logは `mcp__claude_ai_Google_Drive__search_files` (`title contains
  'train.log'`) → `download_file_content` で直接取得可能（ユーザーに貼ってもらう手間を省ける、
  今後もColab結果確認に使える）

次: このv3(car-only)ckpt/結果を基準として、ゼミでablation計画（out/ablation_plan_2026-07.md）の
スコープ合意を取る

## データセット v4: outdoor_siren_v4（2026-07-12 完成、土俵依存性の検証用）
- v3と完全に同一構成（スパース発音2-6秒・車妨害音SIR0-15dB・背景雑音SNR0-20dB・train60/val20）で
  **速度・距離レンジのみ変更**: 5-15m/s→**15-30m/s**、3-15m→**5-20m**（住宅街→幹線道路想定）
- `scripts/step6_batch_scenes.py`に`--v4`フラグ追加（`SPEED_RANGE_MPS`/`OFFSET_RANGE_M`を
  variant依存にして`sample_scene`/`sample_interferer`から参照）。v1/v2/v3のコード・データは無改変
- 同じidxならv3/v4で乱数ストリームの消費順が同じため、側・サイレン型・タイミング・雑音条件は
  v3と揃ったまま速度・距離だけが別レンジに再サンプルされる（意図した設計、疑似ペア構造）
- 全80本検品PASS（az中央誤差0.21-0.34°）。zip: `out/dataset_outdoor_siren_v4.zip`（157.1MB）
- ノートブック: `colab/PSELDNets_outdoor_siren_v4_Colab.ipynb`（v3と同一レシピ、
  DATASET/EXP_NAME差し替え、experiment_name使い回し注意の警告を追記済み）。まだDriveには
  アップロードしていない（ローカルのみ）
- 次: zip+ノートブックをDriveにアップロードして学習（ユーザー実行）。ablationはv3・v4
  それぞれで独立に実施し、物理要素の効き方が土俵で変わるかを比較する

## Colab学習結果 v4（2026-07-12 完走、run1・NaNなし）
- 最終(epoch70): **ER 0.042 / F 97.4% / LE 5.7° / LR 100% / SELD_scr 0.025**
  （v3最終 ER0.042/F96.9%/LE6.5°/SELD_scr0.027 とほぼ同一、わずかに良好）
  → **収束後の到達難易度は速度・距離レンジを変えても同程度**（第一の確認完了）
- **一方、学習の立ち上がりはv3よりv4の方が明確に不安定**:
  epoch5で ER1.000/F-100%/LE180°/LR-100%（ほぼ全滅、v3同時点はER0.427）、
  epoch35でSELD_scr0.309とepoch25(0.228)より悪化する逆戻りもあり（v3では未観測）、
  epoch50以降で急速に持ち直し収束
- **新たな土俵依存性の切り口**: 「収束後の最終難易度」だけでなく「学習の立ち上がり方・
  安定性」も土俵（速度・距離レンジ）で変わる可能性。ablation計画書に追記検討
- ckpt/ログ: Drive `PSELDNets_logs/outdoor_siren_v4/runs/outdoor_siren_v4_run1/`
  （run1のままNaN崩壊なし、experiment_name変更は不要だった）

## データセット v5: outdoor_siren_v5（2026-07-13 完成、マルチクラス拡張）
- v3と同一の土俵（速度5-15m/s・距離3-15m・SNR/SIR/発音区間レンジ）だが、検出対象を
  Siren単独から**Siren/Horn/BackupBeep/BikeBellの4クラス**に拡張（idx%4で均等割当、
  各20本）。クラス辞書は新規 `configs/cls_indices_v5.tsv`（プロジェクト内、外部依存なし）
- **新規合成モジュール**: `src/outdoor_seld/alert_sounds.py`（クラクション・バック警告音・
  自転車ベル）、`src/outdoor_seld/engine.py`（車の走行音、妨害音用）。全てクリーン合成、
  実録音は不使用（`flying_drone.wav`・`suv_dirt_road.wav`とも使わない、本人指示）
  - 初版は簡易な正弦波の合成で実物と質感が大きく乖離（本人指摘）→ 外部調査
    （procedural engine sound synthesis、car horn diaphragm合成、bell modal synthesis
    の技法）を踏まえて作り直し済み。エンジン音=気筒発火の準ノコギリ波+RPMゆらぎ+
    路面ノイズ、クラクション=奇数次倍音のリード楽器的音色、ベル=非整数次倍音+warble
  - プレビュー音源: `out/preview_v5_sounds/`
- **バグ修正**: `make_bike_bell`初版は音がクリップ先頭付近でしか鳴らず、発音区間ウィンドウが
  後半に来ると無音になる不具合があった → クリップ全体で周期的に鳴るよう修正
  （horn/backup_beepは元から周期的で問題なし）
- 全80本検品PASS、クラス分布20本×4クラス確認済み。zip: `out/dataset_outdoor_siren_v5.zip`
  （159.9MB）
- Colabノートブック作成済み: `colab/PSELDNets_outdoor_siren_v5_Colab.ipynb`

## v5 run1の学習不足と200epoch再学習（2026-07-12）
- run1（70epoch、v3と同じ設定を流用）の最終結果: **ER 0.396 / F 65.6% / LE 18.8° /
  LR 96.7% / SELD_scr 0.219**（v3の0.042/96.9%/6.5°/0.027と比べて大幅に悪化）
- **原因は「マルチクラスが本質的に難しい」ではなく学習不足と推定**: run1のSELD_scr推移が
  epoch70時点でもまだ下降し続けており収束していなかった（v3/v4はepoch40-50で横ばいに
  達していたのと対照的）。4クラスに均等分割したことで1クラスあたりの学習データが
  v3(train60本)の1/4(train15本)になり、v3で妥当だった70epochでは足りなかったと推定
- 対策: `EXP_NAME`を`outdoor_siren_v5_run2`に変更（run1のckptを引き継がない）、
  `max_epochs`を70→**200**に変更して再学習（ノートブック更新済み、まだ未実行）
- 次: run2の結果を見て、v3水準に近づけば学習不足が確認され、大差が残ればクラス別の
  難易度差を疑う（誤り解剖のマルチクラス対応が必要になる）

## v5 run2の結果とマルチクラス誤り解剖（2026-07-13）
- run2最終結果（200epoch、ベストckpt=epoch_114）: **ER 0.260 / F 77.3% / LE 14.1° /
  SELD_scr 0.152**。run1（0.219）よりは改善したが v3（0.027）には遠く、
  学習不足だけでは説明できない差が残った → 誤り解剖で内訳を特定
- 手順はv3のときと同じ: Colabで `infer.py mode=test`（`experiment_name=infer_outdoor_siren_v5_run2`）
  → Drive `runs/infer_outdoor_siren_v5_run2/submissions/` に検証20本の予測CSV
  → Drive MCPで取得して `out/predictions_v5_run2/` に保存
- 解剖スクリプト: **`scripts/step8_error_anatomy_mc.py`**（新規。step7はv3の記録として無傷で保存）。
  クラス辞書をtsvから読む汎用設計で、フレームを ok / dir_err(>20°) / substitution
  （別クラス予測がGT方向20°以内＝取り違え）/ miss / fa に分類。
  合成偽予測による自己テストで5カテゴリの計数を検算済み（全PASS）
- 軌跡図: **`scripts/step8b_trajectories.py`**（クラス別にdir_err最多クリップの
  GT方位vs予測方位＋発音帯を描画）。出力はどちらも `out/figures_v5_analysis/`

### 結果: 仮説3は棄却、仮説4は「missではなくdir_err」として形を変えて成立
- **仮説3（周波数帯重複による取り違え）→ 棄却**: substitution は**全738 GTフレーム中0件**。
  混同行列は完全対角（BackupBeep 1000Hz vs Siren peepo 960Hz でも混同なし）。
  クラス識別はクリップ単位でも全20本正解 → SEDのクラス分類は問題ではない
- **miss も主因ではない**: 全体21フレーム(2.8%)で、大半が発音区間の端±0.3s
  （BackupBeep 7/12、BikeBell 4/5）。BikeBellのmiss率2.7%は仮説4が予想した
  「チリンの合間の大量miss」ではなかった（モデルは合間も検出を維持=SEDは補間できている）
- **主因は方向誤差（dir_err >20°）= 166フレーム(22.5%)**。クラス別の集中が明確:

  | クラス | dir_err率 | LE中央値 | 音の性質 |
  | --- | --- | --- | --- |
  | Horn | 0.6% | 4.1° | 断続70%デューティ・倍音豊富 |
  | Siren | 15% | 8.0° | 連続・スイープ |
  | BikeBell | 32% | 14.0° | 2秒周期チリン・減衰（最疎） |
  | BackupBeep | 44% | 19.4° | 0.5s on/off 純音1kHz |

- **dir_errの機構（追加分析で確認）**:
  - dir_errフレームは同クラスのokフレームよりクリーン音エネルギーが低い
    （BackupBeep -6.2dB、BikeBell -9.6dB。連続音のSirenは差 0.6dBのみ）
    → **発音の合間にSED検出は続くがDOAが漂う**
  - dir_errの予測は妨害車の方向を向いていない（車±20°以内はokフレームと同率、
    BikeBellは0%）→ 妨害車への吸い寄せではない
  - 予測方位は「時間シフトしたGT位置」でほぼ説明できる（±3s探索で83-100%が20°以内に解消）。
    BikeBellは**78%が過去位置**（中央値+0.5s遅れ＝最後のチリン方向に留まる）、
    BackupBeep/Sirenは先行気味（スイープの平滑化・外挿）
  - FAは46フレームと少ないが、BackupBeepのFAは10/15が車方向（純音は車騒音に釣られやすい）
- **結論**: v5の劣化はクラス混同ではなく、**時間的に疎/狭帯域な音源の「発音合間の
  DOA追跡」の問題**。移動音源では合間に方位が動くため、鳴った瞬間しか方位情報がない
  BikeBell/BackupBeepでLEとdir_errが増える。これは屋外SELD特有の「移動×疎発音」の
  相互作用で、卒論の考察の柱になる（静止音源のDCASE設定では起きにくい）
- **⚠️2026-07-14訂正（敵対的レビュー#2/#3、SNR層別で再検証済み）**: 上のクラス別の
  物語はval依存だった。v5 valはクラス毎5本でSNR抽選が偏っており（backup_beep中央値
  6.3dB vs bike_bell 14.7dB）、同じv5モデルをv6 val（各20本）で解剖すると順位が
  BackupBeep 27.8% > Siren 18.0% > BikeBell 11.1% に入れ替わる。SNR統制の層別の結果:
  **生き残る主張=「BackupBeep（純音・断続）が最悪」（低SNR層33.4%/高SNR層22.8%、
  SNR相関−0.01でSNRの引きでは説明不可）と「Horn最良」**。
  **棄却=「最疎のBikeBellが2番目に悪い＝疎発音ほど単調に悪化」**（過去位置78%等の
  機構分析はn=5クリップ上の観察、v8で層別・シャッフル対照付きの再分析が必要）。
  新しい芽: Sirenのdir_errは高SNR層でむしろ多く（相関+0.42）、雑音でなく幾何
  （近距離CPAの高速方位スイープ）由来の可能性。詳細=out/adversarial_review_2026-07-14.md
- 対策の候補（未実施）: クラス毎データ増量、発音合間のラベル扱いの再考
  （active窓を鳴っている瞬間だけに絞る等）、時間文脈の長いモデル/トラッキング後処理

## データセット v6: outdoor_siren_v6（2026-07-13 着手、クラス毎データ4倍増量版）
- **目的**: v5誤り解剖の発見（疎発音クラスのdir_err集中）が「クラス毎データ不足
  （v5は各クラスtrain15本=v3の1/4）」で消えるか「本質的限界」かの切り分け。
  ユーザーと選択肢（増量/ポリフォニー/歩行マイク）を比較して増量を採用（最短で
  切り分けができ、実装追加ほぼゼロのため）
- **構成**: v5と完全同一（4クラス・速度/距離/SNR/SIR/発音区間レンジ・クリーン合成妨害音・
  クラス辞書cls_indices_v5.tsv）で、**件数のみ4倍: 各クラスtrain60/val20、計320本**
- 実装: `scripts/step6_batch_scenes.py` に `--v6` 追加。`MULTICLASS = VARIANT in ("v5","v6")`
  フラグを導入してv5分岐を共通化（v1-v5の生成条件はビット単位で不変なことを
  `plan --v5` の出力と既存scene.jsonの突き合わせで回帰確認済み）
- **v5とのデータ関係（再精査で確定、2026-07-13）**: 生成の決定論性により、v5の全80本
  （train60＋val20）はv6のtrain(mix001-080)に**flac/metadataともビット単位で一致**して
  含まれる（13ペアのMD5照合で確認）。v6のval(idx240-319)は未使用のseed領域なので
  **v6内のtrain/valリークはない**。評価時の帰結:
  - v6モデルをv5のvalで評価するのは不可（学習済みのため）
  - v6のval 80本は**v5モデルも未見** → v5 run2 ckptとv6 ckptを同一val 80本で
    ペア比較できる（切り分けの主比較はこれを使う）
- Colabノートブック: `colab/PSELDNets_outdoor_siren_v6_Colab.ipynb`（新規作成、
  EXP_NAME=`outdoor_siren_v6_run1`、**max_epochs=100**（train本数4倍で1epochの
  ステップ数が4倍になるため。v5 run2のベストepoch114相当の総ステップには約30epochで到達、
  余裕を見て100）。Driveへは手動アップロード（v4のMCPアップロード破損事故以降の運用）
- 判定基準: v3水準（SELD_scr 0.027）まで改善→データ不足が主因 / LE14°前後のまま→
  疎発音×移動の本質的限界として考察の柱に。全体指標だけでなく学習後に infer(mode=test)
  → `step8_error_anatomy_mc.py --pred out/predictions_v6_run1 --ds out/dataset_outdoor_siren_v6`
  でv5と横並び比較する
- **生成完了（2026-07-13）**: 320本生成→**検品320本全PASS**（az中央誤差 最大0.576°、
  閾値2.0°）。クラス分布 各80本均等、metadata全320本のクラス列がscene.jsonのclass_idxと
  一致、val新領域(idx240-319)の割当式(idx%4)も全数確認。
  zip: `out/dataset_outdoor_siren_v6.zip`（**637.2MB**、foa320+metadata320+4クラス辞書）
- 再精査（本人依頼）の記録: ①git diffで変更8箇所のみ確認 ②plan --v5出力と既存scene.json
  の一致で回帰確認 ③v5との重複80本のflac/metadata MD5照合13ペア全一致（決定論性の実証）
- 次: zipをDriveの`PSELDNets_data/`へ手動アップロード →
  `colab/PSELDNets_outdoor_siren_v6_Colab.ipynb` を上から実行（EXP_NAME=outdoor_siren_v6_run1、
  100epoch、T4で60-90分見込み）→ infer(mode=test) → step8で誤り解剖をv5と横並び比較。
  余裕があればv5 run2のckptでもv6 valにinferして同一80本のペア比較を取る

## v6 run1 学習結果（2026-07-13 完走）: データ増量で v3 水準まで回復
- 100epoch完走（val評価は5epochごとに20回）。**ベスト: SELD_scr 0.028（ep75/85、
  ER 0.044 / F 97.0% / LE 4.8°）**、最終(ep100) 0.029。ep45以降は0.028-0.035で横ばい＝収束
- 収束の速さも想定どおり: ep20で0.052、ep45で0.031（v5 run2ベスト相当の総ステップは
  約ep30）。100epochで十分だった
- 横並び（val条件は各版の自前val、v6のみ80本）:
  | 版 | クラス | train | ER | F | LE | SELD_scr |
  | --- | --- | --- | --- | --- | --- | --- |
  | v3 run2 | 1 | 60本 | 0.042 | 96.9% | 6.5° | 0.027 |
  | v5 run2 | 4 | 60本(各15) | 0.260 | 77.3% | 14.1° | 0.152 |
  | v6 run1 | 4 | 240本(各60) | 0.044 | 97.0% | 4.8° | 0.028 |
- **暫定結論: v5の劣化の主因は「クラス毎の学習データ不足」**。クラス毎60本に増やすと
  マルチクラスでもシングルクラスv3と同水準に到達し、LEはv3より良い4.8°まで下がった
  → 疎発音クラスのDOAドリフトも（macro指標上は）データで解消された可能性が高い
  （**⚠️2026-07-15訂正: この結論は対照ラン#8で修正された。学習量を揃えるだけで
  v5データでも0.152→0.052-0.093まで改善＝改善の相当部分は学習量の寄与。
  「v8 run1学習」節の対照ラン結果を参照**）
- ただしmacro平均なので断定は不可。**クラス別のdir_err/LEがv5からどう変わったかは
  誤り解剖（infer→step8）で確認するのが次の一手**（BackupBeep/BikeBellのdir_err
  44%/32%が実際に消えたかどうか）。あわせてv5 run2 ckptをv6 val 80本にinferすれば
  同一問題でのペア比較になり「val難易度の差」の可能性も潰せる

## v6 誤り解剖＋v5モデルとのペア比較（2026-07-13 完了）: DOAドリフトは解消（⚠️主因は07-15訂正参照）
- 手順: v6/v5両ckptで v6 val 80本に infer(mode=test) → 予測CSVをColab側で1ファイルに連結
  （`PSELDNets_data/infer_*_all.csv`、80本×2をMCP 2回で取得する省力化。**新手順として有効**）
  → `out/predictions_v6_run1/`・`out/predictions_v5run2_on_v6val/` に分割保存 → step8×2
- **ペア比較（同一のv6 val 80本=3129 GTフレーム、両モデルとも未見）**:

  | クラス | dir_err率 v5→v6モデル | LE中央値 v5→v6モデル |
  | --- | --- | --- |
  | Siren | 18.0% → **0.9%** | 10.0° → 3.2° |
  | Horn | 3.3% → **1.3%** | 4.1° → 3.0° |
  | BackupBeep | 27.8% → **0.6%** | 12.0° → 4.0° |
  | BikeBell | 11.1% → **1.3%** | 6.3° → 4.0° |

  総計: dir_err 463(14.8%)→33(1.1%)、fa 171→76、miss 81→66（両モデルともほぼ全て区間端）、
  substitution は両モデルとも0（クラス識別は60本学習でも完全）
- **結論（v5の疑問への最終回答）**: 疎発音クラスのDOAドリフトは「疎発音×移動の本質的限界」
  ではなく**学習データ量で解決する問題**だった。クラス毎train15→60本で、発音の合間の
  DOA補間をモデルが学習できるようになった。卒論の物語は「疎な音は方向追跡が難しい（v5解剖）
  →ただしデータで解決可能（v6）→合成データ生成パイプラインだからこそ増量が自由自在」
  という筋に整理できる（合成データ研究の意義=学習可能にする手段、の直接の実証例）
- **⚠️2026-07-15訂正（対照ラン#8）: 上の「最終回答」は言い過ぎだった**。v5とv6は
  学習量（総ステップ1600 vs 3000）も違っており、学習量を揃えるだけでv5データでも
  0.052-0.093まで改善する。このdir_err激減（27.8%→0.6%）も「データ量＋学習量の複合効果」
  と読み直すこと。卒論の物語は「訓練を尽くしても残る差がデータ量の寄与（0.05→0.03）」
  という誠実な形に修正する。詳細=「v8 run1学習」節の対照ラン結果
- 残る誤りの筆頭は **BackupBeepの誤通知（fa=47、うち車方向は7のみ）** と **区間端のmiss**
  （オンセット/オフセット±0.3s）。どちらも軸3指標（誤通知率・見逃し率）に直結する題材
- 図: `out/figures_v6_analysis/`（v6解剖3枚＋compare_v5_v6.png=ペア比較棒グラフ、
  scripts/step8c_compare_models.py で生成）、`out/figures_v5run2_on_v6val/`（v5側解剖）
- スコアの位置づけの注意（本人指摘で明文化）: 絶対値はDCASE等の文献と比較不可
  （ポリフォニー1・残響なし・クリーン合成4クラス・同一生成器内の汎化のみ、の簡単な土俵。
  同レシピのDCASE2021 FTはER 0.327/F 75%）。主張は常に版間・条件間の相対比較で立てる。
  SELD_scr 0.03水準は天井が近いので、以後の比較は解剖指標（クラス別dir_err率等）を主に使う

## fastsim物理スイッチ実装＋検証（2026-07-14 完了、ablation本体は未着手=ゼミ合意待ち）
- `render_mono` に `enable_doppler / enable_spreading / enable_air_absorption` を追加
  （既定は全ON）。地面反射のon/offは従来どおり鏡像レンダの加算有無（関数の外）で制御
- **設計判断**:
  - doppler off = 「一定遅延読み出し」（伝搬遅延の中央値で固定）。ピッチ変調だけが消え、
    1/r・吸収は時変のまま（**⚠️2026-07-14訂正P10: 「既存生成器（SpatialScaper等）の世界の
    再現」という同一視は過大。あちらは区分的時変遅延で別物。以後「ピッチ変調を消した条件」
    とのみ主張する**）
  - absorption off = FIRを通さない。ただし線形位相FIRの群遅延256サンプル分のゼロ遅延を
    入れて、条件間で波形タイミングが揃うよう補正（onとxcorrラグ0を確認）
- **検証（scratch test、全PASS）**: ①既定値の出力が**git HEADのfastsimとビット単位一致**
  ②doppler: 1kHz純音・10m/s通過で接近1029.4Hz/後退972.2Hz（理論±29Hzと一致）→off で
  1000.0/1000.0Hz ③spreading: 近5m/遠40mの振幅比7.4（幾何どおり）→off で1.02
  ④absorption: 白色雑音で8-16kHz帯が+1.87dB（200-1k帯は+0.03dB）→高域だけ効いている

## データセット v7: outdoor_siren_v7（2026-07-14 着手、地面反射ON=フル物理版）
- **目的**: ablationの基準（フル物理）土俵。「v7から1要素ずつ外す」一要因設計の参照条件。
  v7は「完全版」ではなく、はしごの次の段（この先の候補: ポリフォニー・歩行マイク・
  非剛体地面・帯域拡張・実録検証セット）
- **構成**: v6と完全同一の320シーン（同一シード・同一分割 train=idx0-239/val=idx240-319）
  で、**地面反射のみON**。two-ray（像音源）・剛体地面R=+1（対象音と妨害車の両方に適用、
  世界の物理を音源種別によらず一貫させる）。ラベルは直接音DOAのまま（反射は入力側の外乱）
- **物理の事前実測（v6のwithreflファイルで確認）**: 反射はIV方位を汚さない（0.3°のまま、
  像音源が同方位のため）が、**仰角を中央値1.4〜3.3°（最大10°）押し下げる**
  → 検品ゲートは方位2.0°据え置き・仰角のみ5.0°に緩和（物理由来のバイアスのため）
- 実装: `--v7` フラグ（step6）。対象音=direct+mirror、妨害車もmirror追加レンダ、
  SNR/SIR基準は「混合に入る形の対象音」（反射込みW）。foa_clean_24k も反射込みに
  （検品SNR/SIR実測・step8エネルギー分析との整合のため）
- スモークテスト（gen 0-1）: PASS。仰角バイアス2.47°/2.94°がv6のwithrefl実測と一致
  （=意図どおり反射が入っている）、方位0.25°、SNR/SIR実測が目標±0.05dB
- **クロス評価の設計メモ**: v6とv7はシーン分割が同一なので、v6モデル↔v7モデルを相互の
  valで評価してもリークなし。「反射を知らないモデルは反射入りでどれだけ崩れるか」
  （sim-to-realの縮図）が測れる
- Colabノートブック: `colab/PSELDNets_outdoor_siren_v7_Colab.ipynb`
  （EXP_NAME=outdoor_siren_v7_run1、100epoch）
- **初回生成の検品で39/320本が仰角ゲート不合格（2026-07-14）→ ゲート設計の誤りと判明**:
  方位は全320本OK（0.2-0.5°）だが、仰角バイアスの裾が事前実測（4本で1.4-3.3°）より大きく、
  近距離・低音源高のクリップで**最大7.9°**に達し緩和閾値5.0°を超えた。これは反射の物理
  そのもの（バグではない）。閾値をさらに緩めるとゲートが骨抜きになるため、
  **ゲートを再設計**: ラベル整合の照合は「直接音のみのFOA（foa_direct_24k.flac、v7で
  新規保存）」に対して行い、閾値は全版共通（az 2.0°/el 2.0°）に戻した。反射込み音声の
  仰角バイアスは既知物理として記録扱い。SNR/SIR実測は従来どおり反射込みクリーン基準。
  v1-v6の検品挙動は不変（直接音ファイルはv7にしか無い）
- **再生成完了（2026-07-14）: 検品320本全PASS**（az_med最大0.5°台/el gate=直接音照合、
  全版共通2.0°基準）。zip: `out/dataset_outdoor_siren_v7.zip`（**647.5MB**）。
  再生成前後で foa/*.flac のMD5が5/5一致（決定論の再確認、音声は初回生成と同一）。
  foa_direct_24k.flac は320/320保存済み
- **反射による仰角バイアスの全数実測（卒論用の物理量）**: 反射込みクリーン音のIV仰角と
  ラベル仰角の差（クリップ毎中央値）は **320/320本すべて負（下向き）、中央値−3.0°、
  p10/p90=−5.1/−1.7°、範囲−7.8〜−0.6°**。像音源が常に真下にあるため方向は一貫、
  大きさは距離・音源高で変わる。v7の「反射を見抜く」タスクの難しさの定量値
- **v7 run1学習結果（2026-07-14完走、100ep）**: ベスト **SELD_scr 0.022**（ep90付近、
  ER 0.036/F 98.0%/LE 4.3°）、最終(ep100) 0.024/ER 0.041/F 97.8%/LE 4.3°。
  v6（0.028/LE 4.8°）と同水準〜わずかに良い（差はrun間ブレ幅Δ~0.003と同程度で
  有意とは言えない）。**反射をONにしても、学習に含めれば性能コストはほぼゼロ**
  （仰角バイアス−3.0°を学習で補償できている）が暫定結論。クラス別解剖と
  v6↔v7クロス評価は未実施

## 研究全体の敵対的レビュー（2026-07-14、独立サブエージェント2名+本体裏取り）
- 手法: 方法論・統計担当と卒論審査員視点の2名の独立審査役に全ファイルを読ませて
  粗探し（物理担当は実行環境のセッション上限で未完了→後日再実行）。
  統合結果は **`out/adversarial_review_2026-07-14.md`**（対処の優先順位付き）
- **最重要発見（本体で事実確認済み）: v5/v6/v7でクラスと通過側(L/R)が100%交絡**。
  side=idx%2 と class=idx%4 が同じidxに依存するため、siren/backup_beepは全数左、
  horn/bike_bellは全数右（inspection.csv全数で確認）。「substitution=0」の解釈と
  DOAヘッドの半空間学習に影響し、**このままablationに進むと全結果がこの欠陥を継承**
  → 対処は側割当を独立化した v8 の設計（ablation前の必須修正）
- 他の確定済み指摘: v5 valのクラス別SNR偏り（backup_beep中央値6.3dB vs bike_bell
  14.7dB、n=5/クラス）→v5のクラス別の物語はv6 valで層別再分析が必要 /
  valid==test（評価集合でckpt選択）/ missの大半が区間端規約の産物 /
  v6-v7間でSNR正規化基準が変わる（反射込みW基準）ため厳密には1要素差でない
- ゼミへ持ち込む決定事項: 実録検証の有無（8月期限）、差なし時の結論文、
  「唯一の手段」等の主張の言い換え、当事者指標（イベント単位）の実装方針

## レビュー指摘の修正第1弾（2026-07-14）: 文言訂正＋イベント単位指標の実装
- 修正済み: #7（「0dBでも壊せない」をn=2観察に格下げ、v2節に訂正注記）/
  #9（surveyの「実録で検証した」主張に要決定マーク、決定まで発表資料使用禁止）/
  #2・#3（v5解剖節にSNR層別再検証付きの訂正注記。生き残る主張=BackupBeep最悪・Horn最良、
  棄却=疎発音単調悪化の物語）
- **#6/#15対応: `scripts/step8d_event_metrics.py` 新規実装（軸3の初の実体）**。
  イベント単位の見逃し率・初検出遅延・初検出時方向正否・誤通知イベント数/分を算出。
  v6 val 80イベントでの初結果:
  | 指標 | v6モデル | v5モデル(同一val) |
  | --- | --- | --- |
  | イベント見逃し率 | **0%（全クラス）** | 0%（全クラス） |
  | 初検出遅延 中央値 | 0.01-0.15s | 0.0-0.09s |
  | **初検出時の方向正解率(20°)** | **35-70%**（BackupBeep35%） | 15-40% |
  | 誤通知イベント | **1.05件/分** | 3.0件/分 |
  - **新発見: 「見逃し」はこの合成世界では既に解決済み（0%）で、当事者指標の実態は
    ①誤通知率と②初動の方向精度**。フレーム平均ではok97%でも、歩行者が反応すべき
    「最初の通知の瞬間」の方向はv6モデルでも35-70%しか合っていない（方向が安定するのは
    検出から数フレーム後）。軸3の物語がここから立てられる
  - 誤通知1.05件/分は実運用換算63件/時＝通知設計（持続フィルタ等の後処理）が将来課題
    という正直な材料にもなる。定義v1はゼミで要合意（TOL=0.15s、FA連結0.3s、20°）

## レビュー後の本人決定（2026-07-14）
- **#8対照ラン=実施する**（v5データのまま総ステップをv6相当に揃えたColab学習1本。
  「v5→v6改善はデータ量が主因」の対立仮説を潰す。v8学習と同セッションで実施予定）
- **実録検証=最小実録を準備する方向**でゼミに臨む（FOAマイク調達可否を先生に相談、
  8月着手期限。survey主張②の扱いはゼミ決定後に確定）
- **優先順位（本人指示 2026-07-14）: 実録評価は後回し、まず土台（v8とその検証）を固める**
- **6月の住宅街実測は「ラフな下見」だったと本人確認（2026-07-14）**: 本格実測は8月以降。
  → 土俵定数（SNR 0-20dB等）の根拠は当面「便宜的な設定値」と正直にラベルし（レビュー#12
  の対処を変更）、8月の本格実測で (a)評価用の実録クリップ収集 と (b)環境統計の実測
  （SNR/騒音レベル→合成レンジの根拠づけ）を兼ねる設計にする
- v8設計束は個別相談で確定していく（下記）

## データセット v8: outdoor_siren_v8（2026-07-14 実装、レビュー修正の集大成版）
- **設計（本人承認済み: ジッタ含む全項目）**: v5-v7と数値の直接比較をしない新しい土俵。
  ①側×クラス独立化（side=(idx//4)%2, siren_type=(idx//8)%2。全24セル完全均衡を検算済み:
  クラス×側×分割 30/30・10/10・10/10、サイレン型×側×分割 15/15・5/5・5/5）
  ②**test fold新設**（400本= train240/val80/**test80=fold3_room1**。testは最終報告1回だけ）
  ③SNR/SIR正規化を直接音W基準に固定（全条件共通、検品の照合も同基準）
  ④検品に記録列: 実効SNR（発音瞬間基準、名目+3〜6dB）・帯域内SIR（名目+12〜21dB=P4の
  定量がそのまま残る）・全長SIR・反射仰角バイアス
  ⑤全クラスジッタ: horn基音±5%/honk・gap±10%、beep周波数±5%/on・off±10%、
  bell f0±5%/チリン間隔±10%、車エンジン回転±20%（scene.jsonのdry_params/f0_engineに全記録）
  ⑥反射仰角バイアスのレンジゲート[−8,0]°（鏡像エンコードのバグ検知）
  ⑦基準物理=反射ON（理想剛面two-ray）
- **回帰確認**: plan --v5 が変更前と全行一致 / v7 mix001を再生成して音声・ラベルが
  MD5一致（scene.jsonの差はsim_seconds=実行時間の記録のみ、シード由来値は全一致）/
  dry_params・f0_engineキーはv8限定（v5-v7のscene.jsonバイト再現性を維持）
- スモークテスト5本（4クラス＋fold3の1本）全PASS。ジッタの実効確認済み
  （horn 428Hz、beep 969Hz、bell 2922Hz、車35.9Hz等が音に反映）
- Colabノートブック: `colab/PSELDNets_outdoor_siren_v8_Colab.ipynb`
  （EXP_NAME=outdoor_siren_v8_run1、100ep。**valinfer設定**=開発用推論はfold2、
  fold3は最終時のみ。**#8対照ランのセル**=v5データ×総ステップ3000・減衰1800ステップ
  =375ep/step_size225 も同梱）
- **生成完了（2026-07-14）: 400本、検品全PASS**。初回検品で2本が反射仰角ゲート
  （当初[-8,0]°=v7の経験レンジそのまま）をわずかに超え（-8.06/-9.66°）、
  ゲートが狭すぎたと判断して下限を-15°に修正（バグ検知力は保持：バグなら正側や
  桁違いになる）→ 再検品で400本全PASS。音声の再生成は不要だった
- 全数確認: クラス×側×分割の24セル完全均衡（30/30・10/10・10/10）、
  反射仰角バイアス400本全て負（-9.66〜-0.67°）、backup_beepの周波数ジッタは
  100本すべて一意（同一波形問題の解消を確認）
- zip: `out/dataset_outdoor_siren_v8.zip`（**795.1MB**、foa400+metadata400+4クラス辞書）
- 次: zipをDrive `PSELDNets_data/` に手動アップロード →
  `colab/PSELDNets_outdoor_siren_v8_Colab.ipynb` を実行（EXP_NAME=outdoor_siren_v8_run1、
  100ep）→ 同セッションで#8対照ラン（セル13）→ val推論（セル12）→ step8/step8dで
  「直した問題が消えたか」の確認（substitutionと左右汎化に注目）

## v8 run1 学習＋誤り解剖（2026-07-15）: 修正の効果を確認
- 学習完走（100ep、best=ep94）: val **ER 0.085 / F 93.7% / LE 6.7° / SELD_scr 0.052**。
  v6/v7の0.028/0.024より悪いのは**想定どおり**（側カンニング排除＋全クラスジッタで
  難化した新土俵。直接比較しない）。ep90以降も微減継続=ほぼ収束
- val(fold2)推論→解剖（out/figures_v8_analysis/ + event_metrics_v8_run1_val.json）:
  - **substitution=0が維持**: 音源ジッタ＋側手がかり排除後もクラス取り違えゼロ
    →「クラス識別は同一波形・側の産物」批判に実験で回答できた（強い結果に格上げ）
  - クラス別dir_err: Siren 0.7% / Horn 0.6% / BackupBeep 7.3% / BikeBell 7.5%
    （LE 4.0/3.6/6.0/6.0°）。純音・疎音源が相対的に難しい構図は残るが健全な水準
  - イベント指標: 見逃し0%（全80イベント検出）、初動方向正解率 Siren75/Horn80/
    BikeBell65%と改善、**BackupBeep30%のまま**（純音の初動定位＝一貫した残課題）、
    誤通知1.35件/分
  - **左右バランス（今回初めて検証可能に）**: Siren/Horn/BikeBellはL≒R。
    BackupBeepのみ L3.1% vs R12.0% の非対称 → 側10クリップずつの小標本なので
    クリップ運（低SNRの引き）の可能性が高いが要観察（次のrun/テストfoldで再確認）
- **対照ラン（#8）完走・結果判明（2026-07-15）**: v5データ（train60本）のまま総ステップを
  v6相当（3000ステップ・減衰1800ステップ=375ep/step_size225）に揃えた1本。
  軌跡: ep40で0.111 → ep180-230でベスト**0.052** → 後半やや不安定化（ep255で0.128）→
  最終(ep375) **0.093**。
  **⚠️結論の訂正（レビュー#8/A-6の交絡が実在した）**: v5 run2(0.152)→v6(0.028)の改善は
  「データ量が主因」と言っていたが、**学習量を揃えるだけでv5データでも0.052-0.093まで
  改善する**。つまり改善の相当部分は学習量（総ステップ）の寄与で、データ量固有の寄与は
  残差分（ベスト比較で0.052→0.028、約2倍）。「クラス毎データ不足が主因」は
  「学習量とデータ量の両方が効いた（学習量の寄与が想定より大）」に書き直す。
  v5→v6のdir_err激減（27.8%→0.6%）の解釈も同様に学習量の寄与を含む
- 対照ラン再実行時の注意: セル13を再実行すると「完走済みのため39秒で即終了」する
  （resume機構の正常動作）。その際の再開時val（0.488）は本来の最終値0.093と乖離しており
  **last.ckptのDrive書き込み破損を疑う**（実験結果には影響なし=結論はログの軌跡から取る。
  このckptからのresumeは以後行わないこと）
- 残: fold3（test）は未使用のまま温存（最終報告時に1回だけ）

## 土台の再精査・第2ラウンド（2026-07-15、本人依頼。審査役2名＋本体で文書修正）
- 統合と全指摘は out/adversarial_review_2026-07-14.md **8節**。要点:
  - **実装バグはゼロ**（v8の均衡・ジッタ・SNR/SIR・ラベル整合・fold3・zipの全数検証PASS）
  - 監査官11件: 文書系10件は即日修正（旧結論への訂正注記、**PROGRESS冒頭に「生きている
    結論の一覧」表を新設**、ablation計画書に**7節=v8基準の改訂第3版**を追記）。
    未実装1件=no-dopplerのラベル一定遅延規約（ablation着手前必須、計画7.3）
  - データ検証官8件: 最重要は**D1=クラス⇔絶対音量の相関**（減衰型bellだけ平均パワーが
    12-14dB低く、クリップ音量だけでbellをAUC0.943判別可。ピーク正規化の副作用）。
    「substitution=0=識別本物」に音量ショートカットの反証可能性が残る →
    **要決定: v9（クリップ全体レベルジッタ±12dB、比率不変・再生成約1時間）vs 限界明記**
  - 収穫: v8解剖の「beep右側だけdir_err 12%」の謎は**データの引きの偏り**
    （右側beepは「CPAが発音区間内」率が28pt高い、p=0.006）でほぼ説明＝モデルの欠陥ではない
  - v8のtrain/val幾何はv7と同一シード（fold3のみ新規）である点を明記（意図的な設計）

## 物理・信号処理審査の完了と統合（2026-07-14、レビュー第3の審査役）
- 14件の指摘（統合と対処は out/adversarial_review_2026-07-14.md 7節）。
  総評:「**実装済みデータセットv1-v7自体に致命的な物理バグは無い。深刻な問題は全て
  これから実行するablationに集中**」→ 着手前に発見できた
- **P2（即修正済み）**: doppler-offスイッチで音の未到達区間が素通しになるバグ
  （吸収offで物理レベル+40dB）。fastsimに `valid &= isfinite(te)` を追加、
  前到達区間rms=0を検証、既定経路はgit HEADとビット一致を維持
- **P1（ablation実装要件として確定）**: doppler-off条件は一定遅延読み出しのため
  音のオンセットがラベル窓から最大0.16-0.29sずれる → doppler-off条件のデータセットを
  作る際は**ラベルも同じ一定遅延規約で生成する**こと（fastsim docstringに要件記載）
- P3: v6↔v7のSNR基準シフトは実測−2.2〜+5.2dB（クリップ依存・定数補正不能）
  → #5の「直接音W基準に固定」を裏付け
- **P5**: siren以外の3クラスはドライ波形が全80本同一（beep/bellは完全決定論）
  → 「クラス識別完全」を自明化している疑い。**v8で全クラスに周波数・時間ジッタ導入**
- P7: 大気吸収は本土俵（≲50m・≤4.6kHz）では高々1-1.5dBで「差なしがほぼ確定」
  → ablationの結論文に「物理的予測どおりの無効果」として事前に織り込む
- その他: P4（帯域内SIRは名目+11〜25dB→検品に記録列追加）、P6（「フル物理」呼称は
  「理想剛面two-ray」へ）、P8（v7検品に反射込みIV仰角のレンジゲート追加）、
  P9（対流増幅1.5dB@30m/s未実装→限界明記、等価性維持を優先）、P10（SpatialScaper
  同一視の文言修正済み）、P11/P12/P13/P14（記録・限界記載・v8時確認）
- **ablation本体（offの各条件のデータ生成・学習）はまだ始めない**（本人指示 2026-07-14、
  ゼミ合意後に着手）。スイッチは実装・検証済みだが未使用
