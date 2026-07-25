# 音クラスの音量と構成の調査（2026-07-16）

本人の問い「各クラスのdBは？」「音クラスはこれで十分か？」への調査結果。
以後、設計値はすべて出典付きで決める方針（feedback-research-before-deciding）の第1弾。

## A. 各クラスの音源音量（法規・実測）

| クラス | 音量 | 測定距離 | 根拠 |
| --- | --- | --- | --- |
| サイレン（日本の救急車） | **90〜120 dB** | 前方20m | 道路運送車両法の保安基準 |
| クラクション | **93〜112 dB(A)**（欧州ECE R28、乗用車）/ 米は93〜118dB@2m | 7m | UNECE / FMVSS |
| バック警告音 | **87〜112 dB**（純音1kHz型は97〜112dB。スマート型は環境+5〜10dB） | 約1.2m | OSHA / 業界標準 |
| 自転車ベル | **80〜95 dB**（典型85〜90） | 1m | ベル計測研究（Edworthy 2023ほか） |
| 車の走行音（エンジン車） | **60〜67 dB(A)**（軽車両。速度+10km/hで約+1dB） | 10m | 欧州pass-by実測 |
| 静音車＋AVAS | 最低 **50（10km/h）/56（20km/h）dB(A)**、上限75 | 2m | UN R138 |
| 住宅街の暗騒音 | 45〜55 dB(A)（一般値。**8月に実測で置換**） | — | 仮値 |

### 設計への大きな帰結（提案の変更）

**法規の音量幅そのものが巨大なジッタになっている**：サイレンは30dB幅、クラクション19dB、
ベル15dB。そこで音量ショートカット対策（旧・案A=一律±12dBジッタ）を、
**「法規に基づく現実の音量レンジをクラスごとに設定し、その幅内でジッタする」**方式に
置き換えることを提案する（案A'）。

- 現実性と統制の両立: クラスの音量差は現実どおり残るが、幅内ジッタ＋距離変動（3〜15mで
  約14dB）＋SNR変動により「クリップ音量だけでクラスを当てる」精度は大きく下がる
  → v9検証でD1と同じAUC測定を行い、ショートカットが実用水準でないことを数字で確認する
- 実装: シミュ内の基準（デジタル振幅⇔dB SPLの対応）を1つ決め、各クラスのゲインを
  「基準距離で法規レンジに一致」するよう較正。8月実測は暗騒音と実車の検証に充てる

## B. 音クラスは十分か（DHH当事者研究より）

**調査結果**: DHH当事者の音認識ニーズ調査（Findlater et al. CHI2019, N=201 / Bragg et al.,
N=87）では、優先度は「**緊急音（火災報知・サイレン・接近車両）＞人の存在
（呼びかけ・足音）＞家電**」。安全関連音への言及は36.3%で最多カテゴリ。
また中等度難聴の歩行者は**音の方向の特定が困難**なため事故リスクが高い（Arrive Alive）。

**現行5クラス（サイレン・クラクション・バック音・ベル・車走行音）の評価**:
当事者調査の最優先層（サイレン・接近車両・警告音）を**すでにカバーしている**。

**追加候補の検討**:

| 候補 | 当事者価値 | 音響的実現性 | 判定案 |
| --- | --- | --- | --- |
| **踏切警報音** | 高（日本の歩行環境で致命的に重要） | 高（トーン交互、合成容易、既存4種と帯域住み分け） | **v9への追加を推奨（第6クラス）** |
| オートバイ | 中〜高（速い・接近危険） | 中（車と同族の広帯域+高回転） | 車クラスの音源ファミリー拡張（f0・音色の幅を広げる）で吸収する案 |
| 人の呼びかけ・叫び声 | 高（調査で上位） | 低（音声認識の領域、別研究規模） | 将来課題として明記 |
| 電動キックボード | 中（増加中） | 低（ほぼ無音=EVと同じ物理限界） | 検出限界の議論に含める |
| 無ベルの自転車 | 中 | 低（ほぼ無音） | 同上（物理限界） |
| 防災無線・屋外放送 | 中 | 中 | 将来課題（方向性が異なる=場所固定） |

## 出典

- サイレン規制: 道路運送車両法（[解説記事](https://medium.com/learn-share-nihon/japan-introduces-the-comfort-siren-for-ambulances-can-you-hear-the-difference-2ee87019546d)）
- クラクション: [Vehicle horn (Wikipedia/ECE R28)](https://en.wikipedia.org/wiki/Vehicle_horn)、UNECE GRB
- バック警告音: [Back-up beeper (Wikipedia)](https://en.wikipedia.org/wiki/Back-up_beeper)、[OSHA解釈](https://www.osha.gov/laws-regs/standardinterpretations/1993-07-12-2)
- 自転車ベル: [Edworthy et al. 2023](https://journals.sagepub.com/doi/10.1177/21695067231192480)、ベル計測記事
- 走行音: [TRID 30-50km/h研究](https://trid.trb.org/View/482440)、[速度と騒音](https://www.cameatechnology.com/articles/noisy-speeding-impact-of-speed-on-noise-level/)
- AVAS: [UN R138とR51の整合資料(OICA)](https://wiki.unece.org/download/attachments/136446230/TFVS-04-12%20(OICA)%20UN%20R138%20and%20UN%20R51%20Matching.pdf?api=v2)
- DHH調査: [Findlater et al. CHI2019](https://dl.acm.org/doi/10.1145/3290605.3300276)、Bragg et al.、[Arrive Alive](https://www.arrivealive.mobi/hearing-and-road-safety)、[英国のろう者道路安全研究](https://pubmed.ncbi.nlm.nih.gov/20945247/)
