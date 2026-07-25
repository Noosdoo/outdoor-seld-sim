# 参考文献・出典の項目別一覧（2026-07-17 作成、卒論の文献リストの元台帳）

各項目 = 研究のどの決定を支えているか。詳細な使い方は「詳細」列の文書を参照。
リンク切れに備え、8月にScholar追跡と併せて書誌情報（著者・年・誌名）を確定させる。

## 1. 先行研究・新規性（must-cite）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [PAWS (IEEE IPSN 2018)](https://ieeexplore.ieee.org/document/8366992/) / [Columbia ICSL](https://icsl.ee.columbia.edu/paws/) | 歩行者向け音響ウェアラブルの先行。軌道推定不可の自認→本研究の観測可能性実証と接続 | survey_novelty_update |
| [難聴者向けサイレン認識ウェアラブル (Sensors 2023)](https://www.mdpi.com/1424-8220/23/17/7454) / [PMC版](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10490602/) | DHH向けサイレン検出（方向なし）→ 方向つき・多クラスが本研究の差分 | survey_novelty_update |
| [6DoF SELD (arXiv:2403.01670)](https://arxiv.org/abs/2403.01670) | 歩行者視点SELDの先行。自己雑音の議論を引用する義務 | survey_novelty_update |

## 2. 先行研究・周辺（土俵の位置づけ用）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [WASN屋外SELD (arXiv:2403.20130)](https://arxiv.org/abs/2403.20130) / [IEEE](https://ieeexplore.ieee.org/document/11192195/) | 屋外SELDの近縁（センサ網、単体ウェアラブルでない） | survey_novelty |
| [サイレン検出+DOA (arXiv:2506.23437)](https://arxiv.org/html/2506.23437) | 単クラスのサイレンDOA先行 | survey_novelty |
| [ウェアラブルSELD状況認識 (arXiv:2509.14650)](https://arxiv.org/html/2509.14650) | ウェアラブルSELDの潮流 | survey_novelty_update |
| [SELDレビュー (npj Acoustics 2025)](https://www.nature.com/articles/s44384-025-00036-3) | SELD分野の総説 | survey_novelty |
| [Personal Sound Zones 物理ablation (arXiv:2603.02508)](https://arxiv.org/html/2603.02508) | 「物理要素のablation」という方法論の近縁例 | survey_novelty |
| [DCASE 2025 Stereo SELD](https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content) / [DCASE 2026](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) | コミュニティの課題設定の現在地 | survey_novelty |
| [audiblelight (生成ツール)](https://github.com/audiblelight/audiblelight) | 合成SELDデータ生成の同時代ツール | survey_novelty |

## 3. 当事者（DHH）調査・通知設計の人間側根拠

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [Findlater et al. CHI 2019](https://dl.acm.org/doi/10.1145/3290605.3300276) | DHH N=201: 通知してほしい音の優先度（緊急音>人>家電）→ クラス選定 | sound_class_research |
| Bragg et al.（書誌確定は8月） | DHHの音認識ニーズ | sound_class_research |
| [Arrive Alive: Hearing and Road Safety](https://www.arrivealive.mobi/hearing-and-road-safety) | 聴覚と道路安全の一般論 | sound_class_research |
| [歩行者の回避行動研究 (PubMed 20945247)](https://pubmed.ncbi.nlm.nih.gov/20945247/) | 回避に要する時間 ≈2秒 → リードタイム最低線2.0s | notification_design |
| AASHTO（知覚反応時間 2.5秒） | リードタイム合格線2.5sの根拠 | notification_design |
| FCW（前方衝突警報 約2.6秒） | 車載警報の時間設計との整合 | notification_design |
| SoundWatch（UW News 2020） | スマートウォッチ通知の先行・誤通知の許容論 | notification_design |

## 4. 音のクラス仕様・法規（音量較正=案A''の土台）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| 道路運送車両の保安基準 第49条（[解説](https://medium.com/learn-share-nihon/japan-introduces-the-comfort-siren-for-ambulances-can-you-hear-the-difference-2ee87019546d)） | サイレン 90〜120dB@20m | sound_class_research |
| [パトカーのサイレン=最高870Hz・周期8秒(4秒併用) (MOBY)](https://car-moby.jp/article/entertainment/general-entertainment/siren-sound-comparison/) / [パトライトFAQ](https://www.patlite.co.jp/support/faq/detail00342.html) | wailサイレンの周波数・周期（v9.1修正の根拠）。下限は非公開→435Hz仮置き・8月実測 | source_audit_2026-07-17 |
| [Vehicle horn (Wikipedia/ECE R28)](https://en.wikipedia.org/wiki/Vehicle_horn) | クラクション 2音同時(~400/500Hz)・93-112dB@7m | sound_class_research |
| [Back-up beeper (Wikipedia)](https://en.wikipedia.org/wiki/Back-up_beeper) / [OSHA解釈](https://www.osha.gov/laws-regs/standardinterpretations/1993-07-12-2) | バック警告音 ~1000Hz断続・87-112dB | sound_class_research |
| [Edworthy et al. 2023 (自転車ベル)](https://journals.sagepub.com/doi/10.1177/21695067231192480) | ベルの警告音としての有効性・80-95dB@1m | sound_class_research |
| [AVAS: UN R138 整合資料 (OICA)](https://wiki.unece.org/download/attachments/136446230/TFVS-04-12%20(OICA) | EVの接近通報音 50-56dB@2m（限界節で言及） | sound_class_research |

## 5. 踏切・自転車ベルの構造（2026-07-17 精査で追加）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [鉄道総研 人間科学ニュース200号](https://www.rtri.or.jp/rd/news/human/human_201511.html) | 踏切電子音=700/750Hz**同時の和音**・毎分130回・電鐘波形が下敷き → make_crossing_v2 | v9_values_research訂正 |
| [踏切警報機 (Wikipedia)](https://ja.wikipedia.org/wiki/%E8%B8%8F%E5%88%87%E8%AD%A6%E5%A0%B1%E6%A9%9F) | 断続変調・故障モード（正常=断続の裏付け）・方式3種 | v9_values_research |
| [鉄道用語解説 (isok.jp)](https://isok.jp/rail/term/term_hu/cralam.htm) / [デイリー新潮記事](https://www.dailyshincho.jp/article/2021/04010601/?all=1) | 踏切音の規格値の傍証 | v9_values_research |
| [警音器(自転車) Wikipedia / JIS D 9451](https://ja.wikipedia.org/wiki/%E8%AD%A6%E9%9F%B3%E5%99%A8_(%E8%87%AA%E8%BB%A2%E8%BB%8A)) | ベルの規格方式（引き打ち・単打・スプリング）→ ベル2種化の根拠 | source_audit |
| [cb-asahi ベル解説](https://www.cb-asahi.co.jp/blog/products/4734/) / [BICYCLE POST 構造](https://bicycle-post.jp/pwk0000709-post/) | 引き打ちベルの機構（歯車ローラー連打）→ 連打レート設計 | source_audit |
| [自転車産業振興協会 ベルの構造材質と音響 (1)](https://jbpi.or.jp/giken_post/tech/23154/) / [(2)](https://jbpi.or.jp/giken_post/tech/23157/) | ベルの音響特性の技術資料 | source_audit |

## 6. 車の走行音・タイヤ・静音車

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [タイヤ騒音 multi-coincidence peak ~1kHz](https://www.researchgate.net/publication/228572128_The_multi-coincidence_peak_around_1000_Hz_in_tyreroad_noise_spectra) | タイヤ帯1kHz中心の根拠（make_car_v9） | v9_values_research |
| [タイヤ/道路騒音研究 (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0301479724028913) | スペクトル形状の速度不変・レベル∝30log10(v) | v9_values_research |
| [TRID 30-50km/h走行音](https://trid.trb.org/View/482440) / [速度と騒音 (Camea)](https://www.cameatechnology.com/articles/noisy-speeding-impact-of-speed-on-noise-level/) | 車 60-67dB@10m の実測レンジ | sound_class_research |
| [NHTSA 静音車規則](https://www.transportation.gov/briefing-room/nhtsa-sets-%E2%80%9Cquiet-car%E2%80%9D-safety-standard-protect-pedestrians) / [Detection of Quiet Vehicles by Blind Pedestrians](https://www.researchgate.net/publication/273549678_Detection_of_Quiet_Vehicles_by_Blind_Pedestrians) | 人間の車検出限界カーブ（機械カーブとの比較土俵） | survey_novelty_update |

## 7. 環境・暗騒音

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [環境省 騒音に係る環境基準](https://www.env.go.jp/kijun/oto1-1.html) | 暗騒音軸 40-65dB(A)（住居昼55/夜45・沿道60）のアンカー | v9_values_research |
| [東京都環境局 騒音基準](https://www.kankyo.metro.tokyo.lg.jp/noise/noise_vibration/environmentstandards/noise) | 同上（地域類型の傍証） | v9_values_research |

## 8. 手法・ツール・規格（物理・信号処理）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [DynamicSound (arXiv:2601.15433)](https://arxiv.org/abs/2601.15433) | 物理シミュレータ本体（放射時刻解・等価性検証の対象） | PROGRESS/geometry.py |
| [PSELDNets 事前学習モデル (HF)](https://huggingface.co/datasets/Jinbo-HU/PSELDNets/resolve/main/model/mACCDOA-HTSAT-0.567.ckpt) | HTS-AT + mACCDOA の学習器・事前学習ckpt | PROGRESS |
| ISO 9613-1 | 大気吸収係数（fastsimのFIR設計） | fastsim.py |
| IEC 61672 | A特性（calibration.py の重み） | calibration.py |
| DAFX02 Karjalainen "Bell-Like Sounds" / CCRMA Risset's bell | ベル加算合成（部分音・warble） | alert_sounds.py |
| ResearchGate "Physically informed car engine sound synthesis" | エンジン音合成の手法根拠 | engine.py |
| soundcy.com / bosshorn.com（ホーン音色解説） | ホーンの2音・リード的音色 | alert_sounds.py |

## 9. シナリオ設計・音源改修の根拠（2026-07-18追記）

| 文献 | 使いどころ | 詳細 |
| --- | --- | --- |
| [警察庁 自転車追い越し規定 (2026/4施行)](https://www.npa.go.jp/bureau/traffic/bicycle/202603.pdf) / [解説(1m目安)](https://agoora.co.jp/jiko/knowledge/car-passing-bicycle2026.html) | S2背後ベルの側方間隔0.8-1.5mのアンカー | scenario_design |
| 道路交通法 第33条（踏切の一時停止） | S1で車が減速接近する想定の根拠 | scenario_design |
| [DOVA-SYNDROME 踏切生録音 (SE#769)](https://dova-s.jp/se/detail/769) | 本人の試聴比較→make_crossing_v2をゲート方式から打撃・余韻方式へ改修した判断材料 | source_audit |
