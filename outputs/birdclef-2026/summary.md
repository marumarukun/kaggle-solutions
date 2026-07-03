# BirdCLEF+ 2026 上位解法 事前読解ガイド（1〜18位）

> 各解法を精読する前に、コンペ全体の構図・共通アイデア・解法間の差分を短時間で把握するための日本語ガイドです。個々のWrite-upの置き換えではなく、精読の「地図」として使ってください。

- **対象順位**: Private最終順位 1〜18位
- **開催期間**: 2026年3月11日（開始） 〜 2026年6月3日 23:59 UTC（最終提出締切）。チームマージ締切は5月27日。
- **参加規模**: 4,000チーム超（公式リーダーボードのチーム数メタデータは4,094）
- **取得日**: 2026年7月3日
- **取得方法**: Kaggle CLI（コンペpage/leaderboard/discussion/topic-messages）のみを使用。Web検索・外部翻訳APIは不使用。翻訳は実行エージェントが直接実施。
- **目的**: フィールド全体像・共通する勝ち筋・順位帯ごとの差を、精読前に俯瞰する。

---

## 1. 結論サマリー（何が勝敗を分けたか）

1. **Perch v2（Google/Cornellのバイオ音響基盤モデル）が中心的役割**。上位のほぼ全チームが、Perchを「アンサンブル1枚として直接使う」か「知識蒸留（KD）の教師にする」形で利用した。Perch単体スコアは高くないが、**LSS（ラベル付きサウンドスケープ）でカバーされない稀少種でPerchが効く**（1位の分析）ため、アンサンブルに不可欠だった。
2. **focal→soundscapeのドメインギャップ克服が最大の主題**。学習データはXeno-canto中心のクリーンな単一種focal録音、評価はブラジル・パンタナールの受動録音（多種重畳・ノイズ）。この差を埋める**擬似ラベル（Noisy Student自己学習）が単一で最大の効き**を示した（17位BUETは1ラウンドの擬似ラベルで**+3.0ポイント絶対**と報告）。
3. **多様性の確保とランク（順位）ベースのブレンド**がほぼ標準化。ブランチ間でキャリブレーション（確率スケール）が食い違うため、確率平均ではなく**クラスごとにパーセンタイル順位へ変換してから平均**する手法が上位で頻出。
4. **メトリクスのノイズが極端に大きい**（Private LB ±0.005、ローカルCVはシード間で±0.02程度）。信頼できるCVは事実上存在せず、**等重みアンサンブル・複数シードでLB過学習を避けたチームがPrivateで生き残った**。Public上位の一部が大きく順位を落とすShakeが発生。
5. **テクスチャ系タクソン（昆虫Insecta・両生類Amphibia）とイベント系タクソン（鳥Aves）の扱い分け**が効いた。昆虫/カエルは「連続する持続音（テクスチャ）」で明確なオンセットがないため、**長い窓・タクソン別スムージング・専門モデル・site/hour/month事前分布**が有効。
6. **昨年（2025）のXeno-canto事前学習バックボーン**（特に2025年2位のEfficientNetV2-S）の流用が大きなブースト（およそ+0.007〜0.01）。自前のスクラッチ事前学習はほぼ全チームで劣化した。
7. **AIコーディングエージェント（Claude Code / Codex）の実装活用が常態化**。1位・2位・6位・8位などが明示的に謝辞。「アイデアは人間、実装はエージェント」という分担が繰り返し語られた。
8. **後処理の定番セット**：site/hour事前分布の融合、時間方向スムージング、file単位のtop-kスケーリング、genus/class（分類階層）スムージング、sonotypeミラーリング。

---

## 2. コンペ固有の用語集

この節を先に押さえると、以降の解法説明が読みやすくなります（一般的なMLの用語集ではなく、本コンペ特有の語に限定）。

- **Pantanal（パンタナール）**: ブラジルの巨大湿地。評価対象の録音が集められたデプロイ先ドメイン。650種超の鳥と多数の動物が生息。
- **focal録音 / train_audio**: Xeno-canto・iNaturalistの利用者が投稿した、対象種が主役の比較的クリーンな録音。32kHz oggへ変換済み。長さは様々（5分超も）で、**多くはクリップ全体に1ラベルのみ**（弱ラベル）。学習の主データ。
- **soundscape（サウンドスケープ）/ PAM録音**: パンタナールに設置された受動音響レコーダー（Passive Acoustic Monitoring）による録音。多種重畳・遠近・背景雑音があり、**テストドメインそのもの**。テストは約600本・各60秒。
- **LSS（labeled soundscapes）/ train_soundscapes_labels.csv**: 5秒窓ごとにラベル付けされたサウンドスケープ。**数十ファイルと極めて少数**（解法により59〜66ファイルと言及）。テストドメインの唯一の正解信号として貴重。
- **USS（unlabeled soundscapes）**: ラベルなしサウンドスケープ。擬似ラベル生成の対象プール。
- **Site 22 / S22**: LSS中の特定サイト。ラベルが特にノイジーで、未知サイトへの汎化チェックや検証で頻出（S22を除外/移動して評価する手法が多い）。
- **sonotype（ソノタイプ）**: 主に昆虫の「音型」ラベル。**孤立したfocal録音が存在せず**、窓単位では音響的に判別困難なグループがある。ミラーリング（グループ内で最大スコアを共有）の対象。
- **234クラス**: 本コンペの予測対象種（鳥・両生類・哺乳類・爬虫類・昆虫）。種コードで表現。
- **評価指標（macro ROC-AUC, 空クラススキップ）**: 5秒窓×クラスごとに在/不在の確率を予測し、**真陽性が1件も無いクラスを除外したマクロ平均ROC-AUC**（`birdclef-roc-auc`）。閾値不要・順位（ランキング）品質が本質。
- **テクスチャ系 vs イベント系タクソン**: 昆虫/両生類は持続的テクスチャ（連続音）、鳥/哺乳類/爬虫類は一過性イベント（単発音）。後処理カーネルや窓長を分けるのが定石。

---

## 3. 頻出テクノロジースタック

| 技術 | 使用した順位 | 概要 | 本コンペでの価値 |
|---|---|---|---|
| **Perch v2**（基盤モデル） | 1,2,3,4,5,6,7,8,10,13,14,16,17,18（実質全上位） | 5秒窓ごとに1536次元埋め込み＋234クラス生ロジットを出す事前学習済みバイオ音響モデル。凍結利用が基本 | LSS外の稀少種を含む広範な音響知識を供給。単体では中位でもアンサンブルの必須ピース |
| **Perch知識蒸留（KD）** | 1,3,4,5,6,7,8,18 ほか | Perchの埋め込み（1536次元）を教師に、SED CNNをcosine/MSEで一致させて学習 | 小さな競技データでは得られない表現を注入。単体+0.01〜0.02級の底上げ |
| **擬似ラベル / Noisy Student自己学習** | 1,2,3,4,5,6,7,8,9,10,13,14,16,17,18（ほぼ全員） | 学習済みモデルでUSSに軟ラベルを付与→焼き直し→反復 | focal→soundscapeのドメイン橋渡し。**単一で最大の効き**（BUETは+3.0点） |
| **2025年 Xeno-canto事前学習バックボーン** | 2,3,4,5,6,9,13,16 ほか | 昨年上位（特に2位のEfficientNetV2-S / vialactea・vladimirsydor）の重みを流用 | ドメイン整合の事前学習で+0.007〜0.01。自前スクラッチより一貫して優秀 |
| **SEDヘッド**（周波数プール+時間アテンション） | 上位のCNN系ほぼ全員 | frame単位とclip単位のロジットを同時出力。GeM/attentionプーリング | 弱ラベルfocalと5秒窓評価の橋渡し。frame-maxで短い鳴き声も拾う |
| **ランク/パーセンタイルブレンド** | 1,3,6,13,14,16,17,18 ほか | 確率平均でなくクラス別に順位変換してから平均 | 異なるキャリブレーションのブランチを公平に統合。ROC-AUCが順位指標なので相性◎ |
| **OpenVINO / ONNXランタイム** | 2,3,4,6,8,10,13,14,17,18 ほか | CPU向けにモデルをIR/ONNX化し推論高速化 | GPU無効・CPU90分制約下で複数モデルをアンサンブルするための実装土台 |
| **ProtoSSM（Mamba系列モデル）** | 5,7,8,13,14,16,17,18 | 60秒＝12窓の系列をMambaでモデリングしPerch埋め込みに時間文脈を付与。prototypeヘッドで稀少種対応 | 窓単位では見えない「60秒全体のテクスチャ持続」を捉える。公開notebook（hideyukizushi）由来 |
| **site/hour/month事前分布** | 1,3,4,8,10,14,16,17,18 ほか | LSSから種×サイト×時刻の出現頻度表を作りロジットへ融合 | 「この沼のこの時刻にこの種がいるか」の強い先験。テクスチャ系で特に有効 |
| **AIコーディングエージェント** | 1,2,6,8,10 ほか明示 | Claude Code / Codexで実装・解析・ログ要約を高速化 | 実験サイクルの加速。ただし「新規アイデアは人間側」との評価が共通 |

### 3.1 Perch v2 と Perch蒸留
**何か**: Google/Cornellが公開するバイオ音響の基盤モデル。5秒窓ごとに1536次元埋め込みと234クラス相当の生ロジットを出す。**本コンペのタクソノミーで学習されたものではない**ため、生ロジットは「ノイジーな出発点」。
**なぜ効いたか**: 競技データは小さくドメインが偏る一方、Perchは膨大なバイオ音響で事前学習済み。特にLSSに現れない稀少種で、Perchが実質的な同定を担う（1位が明言）。
**どこで効いたか**: (a) 凍結Perchを線形ヘッドで直接アンサンブルに投入、(b) Perch埋め込みを教師にSED CNNを蒸留、の2用途。多くのチームが両方を併用。

### 3.2 擬似ラベル（Noisy Student自己学習）
**何か**: 手元最良モデルでUSSに軟ラベルを付け、焼き直し学習を繰り返す半教師あり手法。
**なぜ効いたか**: 学習=focal（クリーン単一種）、評価=soundscape（多種・雑音）というドメイン差を直接埋める。soundscape風のサンプルにモデルを晒すことが本質。
**注意点**: 同一バックボーン系統での反復自己学習は「自分の盲点を再学習」して劣化しうる（17位が明確に報告）。1位も蒸留後は素朴なPLでは悪化し、ラベル和のキャップ・LSSとPLの非重複注入といった工夫が必要だった。

### 3.3 ランク（順位）ブレンド
**何か**: 各モデルの各クラス予測を全行にわたる百分位順位へ変換し、その順位を平均する。
**なぜ効いたか**: SEDのシグモイド出力とPerch派生スコアは確率スケールが噛み合わない。評価指標がクラス内の順序（AUC）である以上、**順位さえ保てれば十分**で、キャリブレーション不整合の悪影響を消せる。

### 3.4 CPU 90分制約と推論エンジニアリング
GPU無効・ネット無効・CPU90分という制約のため、**OpenVINO/ONNX化・melスペクトログラムの共有計算・単一グラフ化・訓練済み成果物のキャッシュ読込**が上位の共通技術。6位は全モデルを単一OpenVINOグラフにしてmelを1回だけ計算し7〜8分節約、8位・14位は学習済みProtoSSM成果物を推論時に再学習せずロードした。

---

## 4. コンペ概要

- **タスク**: 60秒のサウンドスケープを5秒×12窓に分割し、各窓について234種（鳥・両生類・哺乳類・爬虫類・昆虫）の在/不在確率を予測。
- **入力**: train_audio（focal録音、主+副ラベル）、train_soundscapes（5秒窓ラベル付き、少数）、USS（ラベルなし）、site/hour/monthメタデータ。全音源32kHz。
- **出力**: 各`row_id`（=ファイル×5秒窓）×234クラスの確率。
- **指標**: 真陽性ゼロのクラスを除外したマクロ平均ROC-AUC。
- **実行制約**: Code Competition。CPUノートブック≤90分、GPU不可、ネット不可、外部公開データ・事前学習モデルは可。
- **主な難所**: (1) focal↔soundscapeのドメインギャップ、(2) LSSが極少でCVが信頼できない・メトリクスがノイジー、(3) テクスチャ系（昆虫/両生類）と稀少種、sonotypeの音響的判別困難、(4) CPU90分に多様なアンサンブルを詰める推論最適化。

---

## 5. 取得状況（1〜18位）

| 最終順位 | チーム | Privateスコア | 解法ディスカッション | 状態 |
|---:|---|---:|---|---|
| 1 | Nikita Babych | 0.96574 | [1st Place Solution: Noisy Student Meets Distillation](https://www.kaggle.com/competitions/birdclef-2026/discussion/704752) | found |
| 2 | tennogh | 0.96013 | [2nd Place: Diverse Ensemble with Pseudo-Labeling and a Taxon Specialist](https://www.kaggle.com/competitions/birdclef-2026/discussion/704399) | found |
| 3 | kapenon | 0.95992 | [3rd Place Solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704420) | found |
| 4 | BirdCLEF+ 2026 Team🤗🤗🤗 | 0.95902 | [4th Place Solution: BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026/discussion/704309) | found |
| 5 | Jiacheng Ma | 0.95824 | [5th Place Solution: Diversity and Bug - Both Are All You Need](https://www.kaggle.com/competitions/birdclef-2026/discussion/704602) | found |
| 6 | Sinan Calisir | 0.95762 | [6th Place Solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704949) | found |
| 7 | 空飛ぶ宝石 | 0.95715 | [BirdCLEF 2026 7th solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704292) | found |
| 8 | kazumax | 0.95675 | [8th Place Solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704770) | found |
| 9 | Yannan Chen | 0.95661 | [Private 9th \| Public 2nd solution - Fight against randomness](https://www.kaggle.com/competitions/birdclef-2026/discussion/704887) | found |
| 10 | coolz | 0.95627 | [10th solution, simple model as always.](https://www.kaggle.com/competitions/birdclef-2026/discussion/704271) | found |
| 11 | Oh Captain! My Captain! | 0.95599 | [11th Place Solution (and the 3rd Place that Got Away) [Without Perch]](https://www.kaggle.com/competitions/birdclef-2026/discussion/704264) | found |
| 12 | Bobbing Redstart | 0.95581 | [12th place solution - MixMax Consistency Regularization](https://www.kaggle.com/competitions/birdclef-2026/discussion/704404) | found |
| 13 | riesentots | 0.95550 | [13th place solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704276) | found |
| 14 | Dieter | 0.95531 | [14th place solo gold solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704864) | found |
| 15 | YK | 0.95516 | — | **not_found**（Kaggle CLIで公開された全discussionを探索したが、当該チームの一次解法投稿は見つからず。存在しないと断定するものではない） |
| 16 | goonew | 0.95482 | [16th Place Solution](https://www.kaggle.com/competitions/birdclef-2026/discussion/704689) | found |
| 17 | BUET_Perceptron | 0.95475 | [Not Just Birds: Cross-Mutation Distillation meets Selective State Spaces](https://www.kaggle.com/competitions/birdclef-2026/discussion/708839) | found |
| 18 | Win or lose? | 0.95463 | [BirdCLEF+2026 18th Place Solution: Ensemble is all we need](https://www.kaggle.com/competitions/birdclef-2026/discussion/704287) | found |

---

## 6. 解法マップ（比較表）

| 順位 | 主なモデル / バックボーン | Perchの使い方 | 擬似ラベル | 窓長 | ブレンド | 後処理の核 | 特徴的な工夫 |
|---:|---|---|---|---|---|---|---|
| 1 | RegNetY/eca_nfnet/EffNetV2-S 等 多様SED+MLP+専門家 | KD教師（cosine）+native線形をアンサンブル | 反復Noisy Student（2反復が最適） | 5s | ランク融合（FT0.8/Perch0.2） | site事前分布・タクソン別平滑・genus/class平滑・delta TTA | LSSインジェクタ（ラベル和0.5に正規化）、genus専門モデル、S22のLSS外種マスク |
| 2 | EffNetV2-S/b0/NFNet + Perch + Insecta専門家 | 公開Perch枠+蒸留SED枠を採用（自KDは多様性のため不採用） | 5ラウンド反復PL | 主に5s | 概ね等重み | sonotypeミラーリング・時間連続性 | soft AUC+0.25 BCE損失、LB過学習を避け等重みでPrivate安定 |
| 3 | seresnext26t（PerchKD）+ EffNetV2-S（2025-2nd FT） | KD教師 + 昨年FT | 単一ラウンド | 5s | 加重（0.4/0.2/0.4） | site/hour/site×hour事前分布・file信頼度スケール・adaptive delta平滑 | サンプリング戦略に注力（稀少種上げ・site重み・1/√site_count） |
| 4 | Perch + 2×SED + 2×Perch蒸留SED（5モデル） | native + KD教師 | Perch由来PLのみ有効 | 5〜10s | ランクゲート合成 | genus平滑・time-of-day事前分布・sonotypeミラー・fat-tail平滑 | CPU 4vCPUの動的スケジューラで5モデルを90分に収める |
| 5 | hgnetv2-b0/EffNetV2-S/b3 + ProtoSSM + 蒸留SED | 全CNNをPerch蒸留 | 1反復（自己蒸留も併用） | 5〜10s | 0.6·CNN + 0.4·(proto/sed) | dual-window TTA・[0.1,0.8,0.1]平滑・file peak scale | 「バグ」＝dB領域での乗算FilterAugが+0.01の汎化。多様性最大化 |
| 6 | ECA-NFNet/EffNetV2/SwinV2 + 公開Perch/ProtoSSM | 公開Perch枠を等重みランク合成 | 3ラウンド（backbone毎独立） | 5s | ランク0.5/0.5 | 時間平滑のみ | 全モデルを単一OpenVINOグラフ化しmel共有計算で高速化。soft PL閾値0.55 |
| 7 | EffNetV2-S(Perch)/mambaout_tiny(BirdSet)/eca_nfnet(BirdSet) + 自作ProtoSSM | KD教師（Perch v2 + AudioProtoPNet/BirdSet、backbone非detach） | 1ラウンド | 10s | ランクアンサンブル | 分類名不一致修正・データ手動再ラベル | 4k+のマルチラベルfocalを人手で5秒粒度に再ラベル。名寄せ3種救済 |
| 8 | EffNetV2-b3/regnety_016/maxxvitv2 SED + NN/木スタッカー | オンラインKD（cosine, weight10） | 単一ラウンドSED PL | 5s | スタッカーで統合 | 時間平滑[.1.2.4.2.1]・sonotypeミラー | 2段目スタッカー（GRU/LSTM+LGBM/XGB）で60秒文脈・site/hour事前分布・時間近傍を学習 |
| 9 | EffNetV2-S（2025 XC事前学習）中心、eca_nfnetは多様性 | 自身は不使用（Perch蒸留は事後に「強かった」と反省） | 2ラウンドNoisy Student | 10s/2分割 | 擬似ラベルアンサンブル+再重み | taxon-aware平滑・file-post | class-interactionモジュール（clean LSSのみ勾配）、10s+CIの相乗。ランダム性との戦いを主題化 |
| 10 | Perch v2バックボーン+SED Transformer（×4） | Perch v2を凍結バックボーンに（他backboneは不発） | 10〜12件のみ付加（効果不明と明言） | 20s | 実質単一モデル | site/hour/site×hour事前分布・昼夜/月事前分布・per-taxon平滑 | Perch以外では動かない構成。RoPE付きSED Transformer。シンプル志向 |
| 11 | EffNet B0/B1/V2B0系（**Perch不使用**） | 使わず（GPU制約でPerch FT断念） | Stage式（focal→+soundscape→+XC） | 20s | ステージ別アンサンブル | — | 480 mel・hop1001で「20秒=20フレーム」設計。CE損失+nocallクラス |
| 12 | EffNetV2 b0/b3/s（自前XC事前学習） | 未使用（Perch論文の着想のみ流用） | オンライン擬似ラベル | 5s | 多様な増強レジームを混合 | logits sharpening・slidingTTA・平滑・file平滑・粗事前分布 | MixMax整合正則化（ラベル最大でmixup）、クラス別ASL（CASL）、手動データ洗浄 |
| 13 | 3×CNN + 3×SED + 公開Perch/ProtoSSM | 公開Perch/ProtoSSM枠を0.5/0.5合成 | 公開PL（Ali Memetoglu）を流用 | 5〜10s | 等重み平均 | 公開Perch側の後処理に一任 | 公開資産を丁寧に統合。自KDは擬似ラベル併用時に不発 |
| 14 | 20s EffNetV2-B3 SED×2 + Perch/ProtoSSM | 埋め込み/系列信号として使用しAvesは低重み | group別blend4教師で軟PL | 20s | ヒストグラムマッチ+タクソン別重み | temperature/file/rank scale・delta平滑 | sonotypeのみpower sharpening、Aves/非Avesで重み非対称（0.1/0.5） |
| 16 | 公開Perch枠 + SED枝1(EffNetV2-S) + SED枝2(eca_nfnet, soft AUC) | 公開Perch/ProtoSSM枠 + mapped-only蒸留 | 4段（pretrain→SS→train_audio軟ラベル→USS PL） | 10s | ランク空間で貪欲追加 | 階層タクソノミー平滑 | source-balanced Focal BCE、hard negative重み、背景ベッド増強（距離/残響/EQ） |
| 17 | ECA-NFNet/EffNetV2-S + 自作ProtoSSM(Mamba) | 埋め込み+site/hour事前分布→ProtoSSM系列 | Noisy Student→**cross-mutation**（異系統教師） | 5〜20s | ランク0.6/0.4 | sonotypeミラー | 同系統自己学習の劣化を発見し、異バックボーン間で擬似ラベルを交換。250+のスカラをPublicで調整した点を自己批判 |
| 18 | 256/128mel SED + Perch/ProtoSSM（3枝が最良） | native + 直交射影MSE蒸留 | 蒸留単一モデルのPerch枝MixUpのみ | 5s | グローバルper-classランク融合 | event/texture別平滑・file top-1 | 「アンサンブルこそ武器」。3モデルがPrivate最良、4モデル目のNFNetはPrivateで悪化 |

（15位YKは一次解法投稿が未取得のため空欄）

---

## 7. 個別解法サマリー（順位順）

### 1位 — Nikita Babych（Private 0.96574）
**一言**: BirdCLEF連覇。Perch蒸留を土台に、昨年のNoisy Student自己学習を「注入の正規化」で復活させた多様アンサンブル。
- **中核**: 「backbone蒸留（Perch v2埋め込みをcosineで一致）→蒸留を切って低LR full fine-tuning」の2フェーズを全モデルに適用。同一backboneでもヘッド/ラベル空間ごとに**毎回蒸留し直す**ことで微小な多様性を得る。
- **自己学習の復活**: 蒸留+LSSが強すぎて素朴なPLは有害化。**PLラベル和をfocal未満にキャップ**し、**LSSとPLを同一バッチの別サンプルに非重複注入**、さらにS22でLSS外種をマスク。これで2反復まで効いた（0.935→0.946→0.950）。
- **LSSインジェクタ**: LSSが極少で過学習するため、**注入LSSのラベル和を0.5に正規化**して信号を弱め、focal種への集中を維持。
- **アンサンブル**: SED-CNN群 + MLPヘッド + Amphibia/Insecta専門家 + genus（属）単位モデル + Perch v2 native線形。属モデルは重畳環境で判別不能な稀少種に属レベル信号を広げ+0.001〜0.002。
- **融合**: FT群とPerch nativeを**クラス列ごとにランク変換して0.8/0.2でブレンド**。Perchの寄与は主にLSS外種から。
- **検証**: S22分割（未知サイト汎化）とgreedy分割（種カバレッジ最大化）の2本立て。LB探りより検証で決めたパラメータの方が結局最良だった。
- **価値**: 蒸留+自己学習で相関が上がるモデル群に、設計多様性で非線形性を足してPublic首位・Private首位を保持。

### 2位 — tennogh（Private 0.96013、ソロ金）
**一言**: 公開Perch＋改良蒸留SED＋自作CNN＋昆虫専門家を、LB過学習を避け等重みで束ねた多様アンサンブル。
- **戦略**: 序盤から強い公開Perchに自CNNをアンサンブルすると効くと見抜き、**アンサンブル余地を残す**方向で自パイプラインを開発。Perch蒸留は相関を上げるため**あえて不採用**。
- **PL反復**: 焦点PL→サウンドスケープPLを5ラウンド。途中でsoundscapeを混入から**置換（50→60%）**へ変更、5s窓へ統一。XC事前学習backbone採用で0.93台へ。
- **損失**: 最良はsoft AUC + 0.25 BCE。増強はwaveform mixup+周波数/時間マスクのみ。
- **昆虫専門家**: 1位のレシピ流用。昆虫のみ+0.002、5倍アップサンプルで+0.004（アンサンブルの50%）。
- **結果**: 最終0.959 Public / 0.960 Private。**等重み**と反LB過学習が、個々のモデルがLB↔PBで乱高下しても全体を安定させた。
- **補足Q&A**: 教師のキャリブレーションが違うモデルは同一アンサンブルに入れず、gate機構（`ensemble *= 0.5 + 0.5*rank_normalize(gate_model)`）でPerch予測を条件付き併用。蒸留を切ったのはsoftAUC・事前学習backboneとstop_gradientの相性が悪かったため。90分制約はONNX+OpenVINO・同一mel3モデルのI/O共有・OOM回避で対応。

### 3位 — kapenon（Private 0.95992）
**一言**: 「Perch KD（seresnext26t）」と「2025年2位FT（EfficientNetV2-S）」という**特徴もアーキも異なる2系統**を軸にした3モデルアンサンブル。
- **多様性設計**: 特徴（mel256 vs mel128正規化）・backbone・学習レシピを意図的にずらす。特徴もアーキも違うペアが最も伸びた。
- **サンプリング重視**: macro ROC-AUC×極端な不均衡・ドメイン差・site偏りは全てサンプリング問題と捉え、稀少種上げ・focal/soundscapeバッチ比・site重みを丁寧に調整。USSは`1/√site_count`でサンプル。
- **PL**: 反復せず単一ラウンド。教師は3モデルアンサンブル。
- **後処理**: site/hour/site×hour事前分布をN/(N+K)で全体平均へ縮約し`logit(pred)+0.2·logit(prior)`で融合。file信頼度スケール・rank-awareスケール・信頼度依存のadaptive delta平滑。
- **価値**: 「自前アイデアは公開情報に勝てなかった」と率直に述べつつ、公開資産と過去上位解法の丁寧な統合で3位。

### 4位 — BirdCLEF+ 2026 Team🤗🤗🤗（Private 0.95902、チーム）
**一言**: Perch v2 + 自作SED2枚 + Perch蒸留SED2枚の**5モデル**を、Perchでゲートするランク合成でまとめた。
- **構成**: perch(native, 重み0.40) / SEDブレンド(v2s + 2025-2nd FT) / PDENSブレンド(v2s蒸留 + se26蒸留)。単体LBは蒸留SEDが最強（〜0.942）。
- **PLの知見**: SEDアンサンブル出力の擬似ラベルは効かず、**Perch由来PLのみ**が意味ある向上。自前XCスクラッチ事前学習・AnuraSet外部データは劣化。
- **ブレンド**: perchを種ランクのprototypeとしてSED信頼度をゲートし、弱いSEDが稀少種で偽陽性を出すのを防ぐ。
- **推論**: Kaggle 4vCPUを2vCPU×2レーンに割る動的スケジューラで5モデルを90分に収める（OpenVINO/ONNX混在）。
- **後処理**: genus平滑・time-of-day事前分布・sonotypeミラー・fat-tail時間平滑。
- **補足Q&A**: 信頼できるCV戦略は結局見つからず、全データ学習＋10%を無意味な検証に使う運用。表中「Perch単体〜0.949」は**Perchを含む公開モデル**の誤記で単体Perchではない。負例はマルチラベルの出力補集合を指す。バッチmixup・BCE正負重み1:19。

### 5位 — Jiacheng Ma（Private 0.95824、初金）
**一言**: 多様性最大化のCNNアンサンブル。最後に見つけた「dB領域の乗算FilterAug」というバグが汎化を生んだ。
- **構成**: hgnetv2-b0 + EfficientNetV2-S(5s/10s) + EfficientNet-b3（全てPerch蒸留ベース、SEDヘッド）+ ProtoSSM + 蒸留SED。`Final = 0.6·CNN群 + 0.4·(0.6·proto + 0.4·sed)`。
- **効き分解**: 単一モデルの伸びは蒸留50%・各種増強30%・アーキ20%。Baseline0.88→+増強→+Perch蒸留→+事前学習→+自己蒸留→+PL1反復→+TTA→+後処理で積み上げ。
- **「バグ」**: 通常のFilterAugはdB領域の加減算だが、著者は誤って**dB領域で線形ゲインを乗算**（物理的には無意味なスケーリング）。これが+0.01の汎化を生んだ。
- **多様性志向**: melパラメータを絞り単一推論を高速化→より多くモデルを積んで分散低減。**同じ後処理を全モデルに掛けると逆効果**で、位置ごとに異なる後処理を適用。
- **価値**: 頑健でないデータ処理を自覚しつつ、多様性と運でPrivateに良好フィット。

### 6位 — Sinan Calisir（Private 0.95762）
**一言**: 「多様性・事前学習・より多いデータでのPL・賢い後処理」という定石を各段で徹底し、単一OpenVINOグラフで高速アンサンブル。
- **モデル**: ECA-NFNet / EfficientNetV2 / SwinV2（正方入力のためmelを256×256にリサイズ）。2025年2位コードを起点。
- **PL**: backbone毎に独立で3ラウンド（EffNetは2）。**軟ラベル**（種pがp≥0.55で採用、0.05未満はゼロ）。閾値0.55はS22で精度≥0.8となる最小値として決定。PLなし0.900〜0.915→最終ラベルで単体0.945〜0.950。
- **高速化**: 全モデルが同一mel入力なので、**波形入力→log-mel1回計算→各サブモデル分岐→内部加重平均**を単一OpenVINOグラフに。7〜8分節約。
- **最終融合**: 公開Perch/ProtoSSM枝を等重みランクブレンド（`0.5·rank(a)+0.5·rank(b)`）。
- **効かなかったもの**: 10/20秒入力、時間平滑以外の後処理、他backbone、hardラベル、自PLレシピのPerch適用。
- **補足**: Claude Codeは「アイデアは人間、実装はエージェント」。メモリが過去の廃案と新実験を比較し始める副作用も報告。

### 7位 — 空飛ぶ宝石 / Joseph Zhou（Private 0.95715、ソロ金・GM昇格）
**一言**: モデリングは標準的だが、**4k+件のマルチラベルfocalを人手で5秒粒度に再ラベル**するデータ品質勝負。
- **データ**: 学名不一致で少数化していた3種（例: Antiurus↔Hydropsalis maculicaudus）をXC名寄せで救済。長クリップに1ラベルしかない4k+件（全体の約10%）を、Audacityと識別アプリで**共起種を見ながら手動で10s粒度に再ラベル**。誤ラベルも複数発見。
- **モデル**: timm backbone + SEDヘッド + 10s。全モデルPerch v2 + AudioProtoPNet(BirdSet)の2教師で蒸留（**backboneをdetachしない**）。最終はSED4枚+自作ProtoSSM系1枚のランクアンサンブル。
- **melが鍵**: `n_mels=128, hop=320, fmin=50, fmax=16000`で単体0.94（PLなし）。以前は0.92が限界だった。
- **PL**: 1ラウンド。各epochで擬似ラベル部を重み付きサンプリングし約1万件混入。夜行性種は夜行性種と高確率でmixup。
- **価値**: 地道なラベル洗浄と2教師蒸留、Perchの種名対応（例: 1595929=Lysapsus↔Pseudis limellum）の修正で+0.005級を積み上げ金・GM。

### 8位 — kazumax（Private 0.95675）
**一言**: 5秒Perch蒸留SEDの上に、**60秒文脈・site/hour事前分布・時間近傍を学ぶ2段スタッカー**を重ねた設計。
- **1段目SED**: EfficientNet-v2-b3 / regnety_016 / maxxvitv2_nano。192×320 dB-mel、周波数GeM、オンラインPerch蒸留（cosine, weight10）。損失は**soft cross entropy + 0.03·負ロジットペナルティ**（クラス間競合＋偽陽性抑制）。
- **2段目スタッカー**: NN（GRU/LSTM）と木（LightGBM/XGBoost）。入力はbase logits・SED埋め込み・prior logits/probs・site/hourメタ・前後窓ロジット。**residual（補正予測）とdirect（直接予測）を非sonotype/sonotypeで使い分け**。木は`class_id`を特徴に単一GBMで全クラス対応、hard negativeサンプリング。
- **事前分布**: global/hour/site/site-hourの頻度をglobalへ縮約。
- **段階効き**: Baseline0.925(Priv)→+Perch蒸留+0.0157→+PL+0.008→+スタッカー+0.008。
- **価値**: 「より強い5秒モデル」でなく**2段目のサウンドスケープ構造モデリング**が勝敗を分けたと結論。実装はほぼCodex。

### 9位 — Yannan Chen（Private 0.95661 / Public 2位、ソロ金）
**一言**: 「ランダム性との戦い」を主題に、複数シード検証で本物の改善だけを残した。class-interactionモジュールが独自。
- **backbone**: 2025年XC事前学習EfficientNetV2-S（Vladimir Sydorskyy）。自前事前学習の代替backboneは全滅、EffNetが王。
- **MixUpがドメイン橋渡しの要**: 生波形上でfocal同士をmix（focalのみ相手を混入、稀少なLSSは左側では**混ぜずアンカーに保持**）。Beta(0.4)のU字で「1つ大音量+1つ微音」を作りサウンドスケープを模擬。
- **class-interactionモジュール**: clip-wiseロジットへの小さな残差クラス×クラス補正（zero-init）。**clean・未mixのLSSサンプルからのみ勾配**を受け、パンタナールの実際の種共起先験を学ぶ。5s単独では効かず、**10s/2分割と組むとPublic 0.939→0.952**の相乗。
- **外部データ**: 234種をXC/iNatから各200件上限で追加しPublic 0.958（PLなし）。
- **Noisy Student**: 軟ラベルをpower1.3〜1.4で鋭化して2ラウンドで単体〜0.96。ただしPrivateへの寄与は不明瞭。
- **教訓**: 事後に**素のEffNet2枚（exp_C+exp_D）のアンサンブルが、自分の複雑な本提出よりPrivateで高かった（0.958）**。site/hour事前分布は途中から静かに悪化していた。
- **補足Q&A**: CIモジュールのablationは実施したがprior後処理の不具合の影響で解釈困難。事前学習backboneは少なくともPublic +0.01。単一foldは提出せず常に4fold。メトリクスがノイジーで再現困難。

### 10位 — coolz（Private 0.95627）
**一言**: 「いつも通りシンプル」。Perch v2をバックボーンにしたSED Transformer単一モデル系。
- **構成**: Perch v2の凍結mel frontend + Perch v2 CNN backbone(1536次元) → 線形+RoPE → SED Transformer×4層(dim512, RMSNorm) → dual-head SED（att_logits/max_logits）。**Perch v2以外のbackboneでは0.94止まりで機能しない**と明言。
- **学習**: cos+warmup、BCE、10epoch。サンプル戦略80〜500（>500は低重み、<80は高重み、soundscape×2.5）。
- **後処理が主戦場**: site/hour/site×hour事前分布、昼夜補正、Amphibiaの月事前分布（乾季/雨季）、file信頼度スケール、per-taxon時間平滑。ホストが今年出したsite/month/hourメタを活用。
- **PL**: 10〜12件付加のみで効果は不明（PLなしでも1か月前にPublic0.951）。
- **価値**: 単一backboneゆえ多様性は薄いが、Transformer/Mamba差し替えでも同等性能。「単一で十分」。
- **補足Q&A**: コードはGitHub公開（github.com/610265158/birdclef26、DeepSeekで整理したため散らかっていると注記）。

### 11位 — Oh Captain! My Captain! / Salman Ahmed（Private 0.95599、ソロ金、**Perch不使用**）
**一言**: GPUが壊れ小型GPU1枚のみ。Perch無し・大きめmelのEfficientNet系で3位相当の未提出解を持っていた。
- **制約**: 大会2週前に自作GPUリグが故障。PerchはGPUに載らず断念。
- **mel設計**: **480 mel / hop1001 / n_fft4096**。20秒で20フレーム（1秒1確率）を狙った設計。melは480超で頭打ち。
- **学習/推論**: 全モデルCrossEntropyLoss学習・Sigmoid推論、**nocallを追加クラス**に。
- **ステージ**: Stage0(train_audioのみ)→Stage1→Stage2(+soundscapes)→Stage3(+XC追加)。**Stage1（未提出）はPrivate 0.959で3位相当だった**。最終提出はStage混成でPublic0.953/Private0.955。
- **価値**: 限られた計算資源とPerch無しでも金。提出選択の難しさ（より良いPrivate解を選べなかった）を体現。
- **補足Q&A**: 損失はFocalBCE/BCEも試したが、head毎にno-callクラスを持たせる設計上CrossEntropyが自然だった。増強はmixupのみ。XC追加データはむしろ悪化したが多様性のため保持。

### 12位 — Bobbing Redstart / Antoine Masq（Private 0.95581、ソロトップ2級）
**一言**: Perch埋め込みは蒸留せず、Perch論文着想の**MixMax整合正則化**とクラス別非対称損失で戦った。
- **データ洗浄**: 1秒ノイズだけの破損録音369件をURL再取得で復元、ハッシュ重複86件除去、Bos Taurus等で背景種混入区間を手動除去、silent soundscapeを負例として反復採掘。名寄せでXC追加（71/92件）。自前XC事前学習backboneも使用（+0.01）。
- **損失（CASL）**: クラス別γのAsymmetric Loss。ノイジーになりやすいクラスを軟らかく減衰。昆虫sonotypeのfocal損失も低減。
- **MixMax整合正則化**: ラベルを線形補間でなく**最大**でmixup（種が居れば混合物にも居るはず）。個別録音と混合物・異なる増強間で予測の整合を要求し、ノイズラベルへの自己教師信号に。CASLとMixMaxは単独では弱いが**相乗**（CASL+事前学習+MixMaxでPrivate 0.940）。
- **後処理**: logits sharpening(T=0.8)・2.5s sliding TTA(max)・両生類/昆虫の平滑・file平滑・粗事前分布。sliding TTAだけでPrivate +0.0099。
- **最終**: EfficientNetV2 b0/b3/s×増強レジーム違いのアンサンブルでPublic0.960/Private0.956。

### 13位 — riesentots（Private 0.95550、チーム）
**一言**: 公開資産の丁寧な統合。3CNN+3SEDに公開Perch/ProtoSSMを0.5/0.5で足した。
- **構成**: hgnetv2b0 / effnetb3ns / effnetv2s のCNN3枚（tawaraのhgnet notebook）+ effnetv2s×2(10s含む)/eca-nfnet のSED3枚（tuckerのSED notebook、head差し替え・**Perch蒸留を無効化**・増強変更）。全てOpenVINO化。
- **PL**: 初回反復をせず、**Ali Memetoglu公開の擬似ラベル**を流用（これだけでbase hgnetを0.881→0.934）。2回目反復・Perchロジット併用は不発。
- **損失**: CNN/SEDともFocal BCE（CE/ASL/AUCは劣化）。2025年2位の増強がeffnetv2sで有効。
- **アンサンブル**: 6モデル等重み平均 + 公開Perch。自CNN/SED側はPrivate 0.953と公開Perchより強かった。
- **効かなかったもの**: 背景雑音、自前XC事前学習、Perch蒸留、raw signalモデル。

### 14位 — Dieter（Private 0.95531、ソロ金）
**一言**: 2つの強力な自作SED枝 + Perch/ProtoSSM系列枝を、**ヒストグラムマッチとタクソン別重み**で統合。
- **3枝**: SED main（20s EfficientNetV2-B3、234クラス、group別teacher"blend4"の軟PL、sonotype専用power sharpening）/ SED small（BirdNET-v3風neck、広hop、多様性用）/ ProtoSSM・Perch枝（Perch v2ロジット+1536埋め込み、MLPプローブ、site/hour事前分布、系列補正）。
- **group-wise teacher**: 鳥はBirdNET V3専門家55%、Perch ProtoSSM 35%…と**タクソン群ごとに最良ソースを配合**。
- **最終ブレンド3点**: (1) 平均前に各クラス分布をmainへ**ヒストグラムマッチ**（順位保持・スケール統一）、(2) **Aves 0.9SED/0.1Proto、非Aves 0.5/0.5の非対称重み**、(3) sonotypeミラー。
- **知見**: 20s文脈は5s独立より一貫して良い。自作SEDは価値あり（公開Perch/ProtoSSMだけでは相関が高すぎる）。Perchは埋め込み/系列信号として使い、Avesで下げれば「罠」ではない。CVの絶対値は信用しない。
- **価値**: 上位と同型だがラベル作業と多様性が薄く14位。トップとの差は「モデル多様性とラベル品質」と自己分析。

### 16位 — goonew（Private 0.95482）
**一言**: 公開Perch/ProtoSSM枝 + 独立設計の2つのSED枝を、ランク空間で貪欲に足した3枝ランクアンサンブル。
- **Perch枝**: hideyukizushi公開notebook（LB0.926）ベース。**mapped-only蒸留**（Perchが表現できる205/234クラスのみ蒸留、残りは損失0）とpairwise-rankingヒンジ損失（macro AUC狙い）を追加し0.932へ。
- **SED枝1**: EfficientNetV2-S、10s/5s hop。4段（2021-2025事前学習→SS+train_audio→train_audioへオンライン軟ラベル→USS PL）。**source-balanced Focal BCE**（train_audioとsoundscapeの損失を別々に平均して等貢献、実質soundscape3倍）、**hard negative重み**（file内正例だが当該窓不在=3.0、同(site,hour)出現=2.0）、**背景ベッド増強**（距離減衰・残響・EQ転写でPAMドメインを模擬）。
- **SED枝2**: eca_nfnet_l0、soft AUC損失（2025年4位由来）、mel設定違い。多様性が結果的に増加。
- **選抜**: LB過学習抑制のため各追加モデルを複数シードアンサンブルにし、検証LSSのSpearman相関で優先度を管理して貪欲追加。
- **段階**: 0.926→0.932→0.943→0.951→0.95586(Public)/0.95482(Private)。sonotype専用モデルは失敗。

### 17位 — BUET_Perceptron（Private 0.95475、BUET 2人チーム）
**一言**: 「同系統の自己学習が劣化する」問題を発見し、**異バックボーン間で擬似ラベルを交換するcross-mutation蒸留**＋Perch読みのMambaを足した。
- **最重要の教訓**: Stage2で**1ラウンドの擬似ラベル追加が+3.0点**（同一backboneで最大の効き）。「他の凝った工夫を全部足しても、この1ステップの半分に満たない。ドメインギャップをまず埋めよ」と強調。
- **Stage4の失敗→cross-mutation**: 同系統で自己学習を反復すると4backbone中3つが劣化（自分の盲点を再学習）。そこで**各系統に別系統の擬似ラベルを与える**cross-mutationへ切替。個々の伸びは小さいが「悪い理由で一致」から「良い理由で不一致」なアンサンブルになった。
- **擬似ラベルの扱い**: power-sharpening（EffNetV2-S≈1.54、ECA-NFNet≈1.67）、confidence重みサンプリング、**生波形50/50 mixup**（半分クリーン前景・半分実サウンドスケープ背景）。
- **ProtoSSM枝**: Perch埋め込み(1536)＋site/hour事前分布（テクスチャ重み1.10 > イベント0.45）を、3層Mamba+4head cross-attention+prototype読み出し（1.8M params）で60秒系列処理。稀少種にprototypeが効く。
- **融合**: キャリブレーションが全く合わないため**ランクブレンド**（`0.60·SED順位 + 0.40·Perch順位`）。最後にsonotype4群をミラー。
- **正直な自己批判**: 唯一の検証信号がPublic LBで、**250+の自由度（20スカラ＋234クラス閾値）をPublic1分割にフィット**。V2S cross-mutationはPublic+1.1/Private-0.3とその危険の実例。全数値をPublic/Privateペアで公開。
- **重要度順**: ドメイン適応 > ノイジーPLへの頑健性 > アンサンブル多様性 > 凝ったアーキ。

### 18位 — Win or lose?（Private 0.95463、チーム）
**一言**: 「アンサンブルこそ全て」。256/128mel SED + Perch/ProtoSSMの3枝をグローバルper-classランク融合。
- **構成（Private最良=3モデル）**: 256mel SED(EfficientNet-B0、Perch直交射影MSE蒸留) / 128mel SED(EfficientNetV2-S系) / Perch+ProtoSSM。4モデル目のNFNetはPublic多様性は足すがPrivateで悪化。
- **SEDヘッド**: GeM周波数プール+512bottleneck+時間アテンション。`loss = 0.5·BCE(clip) + 0.5·BCE(max_frame)`。
- **Perch/ProtoSSM枝**: site/hour/site-hour事前分布（event/textureで平滑を分ける）→ ProtoSSM(width256, 3層, state16, prototype+familyヘッド) + PCA圧縮埋め込みのクラス別MLPプローブ。
- **PL**: 蒸留単一モデルでUSSに軟ラベル（`score**1.6`鋭化、5シャードで実行）。**Perch単一モデルのMixUpにのみ使用**し全枝には広げず。
- **ランク融合**: 各モデル×クラスをグローバル百分位順位化→等重み平均→最後にfile top-1スケール。キャリブレーション差を吸収するのが狙い。
- **アブレーション**: event/texture別カーネルの時間平滑が有効（Private+0.002/Public-0.0005）。「4モデル>3モデル」ではない＝サイズ≠品質。

---

## 8. 解法横断の比較

### 共通パターン（ソース根拠あり）
- **Perch中心 + 擬似ラベル + ランクブレンド**が上位の共通骨格。11位（Perch不使用）が唯一の例外で、それでもStage式PLで金圏。
- **2025年上位の資産流用**（XC事前学習backbone、蒸留SED公開notebook、ProtoSSM公開notebook、擬似ラベル公開）が広範に効いた。多くが「自前の新規性は公開情報に勝てなかった」と述懐。
- **ドメイン適応（focal→soundscape）が最優先**。擬似ラベル・生波形mixup・背景ベッド/距離増強・LSS注入など手段は違うが目的は同一。
- **キャリブレーション非依存のランク/パーセンタイル融合**が異種ブランチ統合の定番。
- **テクスチャ/イベントの扱い分け**（窓長、タクソン別平滑カーネル、専門家、事前分布重み）。
- **CPU90分の推論最適化**（OpenVINO/ONNX、mel共有計算、成果物キャッシュ）。

### 意味のある差分
- **多様性の作り方**: 「蒸留で相関を上げてから設計多様性で崩す」（1位）vs「蒸留を避けて多様性を温存」（2位・13位）。3位は特徴/アーキの意図的分離、17位はcross-mutationで多様性を能動生成。
- **2段目モデリング**: 8位はGRU/LSTM+GBMの明示的スタッカー、5/7/14/16/17/18位はProtoSSM系列枝で60秒文脈を導入。1位はスタッカー無しでSED多様性＋genus専門家。
- **窓長**: 5s（1・2・3・4・12・18位）〜10s（7・9・16位）〜20s（10・11・14・17位一部）と幅がある。テクスチャ系には長窓が有利という指摘が複数（9・14・17位）。
- **ラベル品質への投資**: 7位（4k+件の手動再ラベル）・12位（破損復元/重複除去/silent採掘）が突出。14位は「トップとの差はラベル品質と多様性」と自己分析。

### 順位帯の傾向（synthesis／推論を含む）
- **1〜2位**は「自己学習の作法（注入正規化・非重複注入）」と「LB過学習の回避」で0.960前後へ抜けた。個々の技術より**運用規律**の差が大きい。
- **中位帯（4〜10位）**は公開Perch/蒸留SED/ProtoSSMの統合＋擬似ラベル＋site/hour後処理という定石の完成度で並び、0.956〜0.957に密集。
- **下位帯（11〜18位）**は「公開0.950級を等重み/ランクで束ねる」戦略が中心で、差別化要素（Perch不使用の11位、手動ラベルの12位、cross-mutationの17位）を1つ持つ形。**多くが金は運の要素が大きいと明言**しており、Privateのノイズ（±0.005）を踏まえると順位差の相当部分は偶然と読める。

> 注: 15位（YK）の一次解法投稿はKaggle CLIで公開された範囲では取得できませんでした。存在しないことを意味しません。上記の順位帯傾向は取得できた17解法に基づく整理です。
