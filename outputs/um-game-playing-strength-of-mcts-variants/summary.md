# UM - Game-Playing Strength of MCTS Variants：上位13解法の事前読解ガイド

- 対象順位：最終1～13位
- 開始：2024-09-05
- 終了：2024-12-02 23:59 UTC
- 取得日：2026-07-04（JST）
- 目的：個々の解法を精読する前に、上位解法の全体像、共通項、差分を理解する
- 取得方法：Kaggle CLIでCompetitionページ、公式最終Leaderboard、全Discussion一覧、選定した解法本文とコメントを取得。一般Web検索は不使用

## 結論

1. 勝敗を最も広く決めたのは、モデル名よりも**対戦順序の対称性を利用したagent flip**だった。`agent1/agent2` を交換し、targetの符号と `AdvantageP1` を変換することで、多くの上位解法が大きく改善した。
2. ただしflipの最適な使い方は一様でない。学習データを直接倍増する方法、元データ用・拡張データ用モデルを分ける方法、推論時TTAだけに使う方法があり、CV改善がLBへ移らない例も多い。
3. 1位を分けた独自性は、既存tabular特徴量だけでなく、Ludii上の初期局面を15秒searchして**ゲームの本当の均衡度に近い追加特徴量**を作った点にある。これは約0.012 LBの初期改善をもたらした。
4. `AdvantageP1` は最重要だがnoisyで、binning、補数変換、targetからの差分化、interaction、ensembleへの直接投入など、各チームが異なる形で再構成した。
5. CatBoostが最も普遍的な強モデルだった一方、LightGBM Dart、XGBoost `lossguide`、TabM、MLP/DeepTablesも多様性源として価値を持った。単体最強より**誤差の異なるモデルの結合**が重要だった。
6. `GameRulesetName` 単位のGroupKFoldは標準だったが、未知game、fold間variance、Public/Privateの分布関係によりCV–LB相関は不安定だった。信頼できるCVを作れた1位と、Public LBへ明示的にfitして成功した2・4・6位が併存する。
7. 予測が0へ縮むbiasを補うため、係数倍、clip、isotonic regression、重み合計を1より大きくするensembleが広く効いた。これは単なる小技ではなく、targetと予測の分布差への補正だった。
8. 追加データやtext特徴量は「多ければ良い」わけではない。1位の高品質な追加rulesetは改善したが、11位の30k～40k行生成はCVだけ改善し、TF-IDFやBERT系も多くのチームで不安定だった。

## コンペ固有用語

- **MCTS**：Monte-Carlo Tree Search。selection、playoutなどを繰り返してゲーム局面の行動を選ぶ探索法。本コンペでは構成要素の異なるMCTS agent同士の強さを予測する。
- **Ludii**：多数のboard gameを記述・実行できるsystem。`LudRules` は完全な形式言語ルール、`EnglishRules` は自然言語説明。
- **GameRulesetName**：Ludii上のgame名とruleset名の組。CVでは同一rulesetをfold間で分離しないgroup keyとして頻用された。
- **agent1 / agent2**：順序付きの2つのMCTS設定。文字列は `MCTS-<SELECTION>-<EXPLORATION_CONST>-<PLAYOUT>-<SCORE_BOUNDS>`。
- **utility_agent1**：予測target。`(agent1の勝数-agent1の敗数)/試合数` で、範囲は-1～1。
- **AdvantageP1**：random playから推定した先手側の有利さ。targetと強く相関するが、対象MCTS agentの非random playに対する完全な有利度ではなくnoiseもある。
- **Balance**：gameの均衡性に関する特徴量。6位はflip時にこれも反転することで元・拡張データ間の誤差差を縮めた。
- **agent flip / inversion**：agent1とagent2を交換し、`utility_agent1` の符号、通常は `AdvantageP1` の補数なども変換するコンペ固有の対称性augmentation。
- **TTA**：Test-Time Augmentation。元行とflip行を両方予測し、flip側の符号を戻して平均する。
- **Public / Private LB**：testの公開部分・非公開部分によるLeaderboard。最終順位はPrivate。Public 35% / Private 65%で、両者に一部同じrulesetが含まれたことが終了後に説明された。
- **Evaluation API**：hidden testを100行ずつ渡す推論API。各batchは10分以内、Notebook全体はCPU/GPUとも9時間以内、Internetなし。

## 頻出技術スタック

| Technology | 使用順位 | 概要 | このコンペでの価値 |
|---|---|---|---|
| Agent flip / TTA | 2, 3, 5, 6, 7, 8, 9, 10, 11, 13 | 対戦順序を交換するaugmentation | 対称性を明示し、学習量と推論多様性を増加 |
| CatBoost | 1～12の大半 | categoryに強いGBDT | agent部品と多数のtabular特徴量を安定処理 |
| LightGBM / Dart | 1, 2, 4, 6, 7, 9, 12 | 高速GBDT、Dartはdropout boosting | CatBoostと異なる誤差、多数seed/foldの高速生成 |
| XGBoost | 2, 11, 13 | GBDT | 13位では `lossguide` と強いsubsamplingで主役 |
| Group系CV | 1～13 | ruleset/gameでgroup分割 | 同じgameの暗記による楽観評価を抑制 |
| OOF stacking | 3, 6, 11, 12 | fold外予測を次段特徴量へ | 単体予測を圧縮しつつ多様な誤差を統合 |
| Scale / clip / isotonic | 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13 | 予測分布の後処理 | 0へ縮む予測をtarget分布へ整合 |
| Rule text features | 1, 2, 3, 5, 11, 12 | TF-IDF、SVD、DeBERTa等 | ruleset構造を抽出する一方、未知gameへのoverfit riskあり |
| Neural networks | 1, 2, 4, 6, 7, 9, 12 | TabM、MLP、DeepTables、DeBERTa | GBDTと異なる表現・誤差をensembleへ提供 |

### Agent flip

2人zero-sumゲームの順序対称性を利用する。単純な行複製ではなく、`AdvantageP1`、target、場合によって `Balance` も整合させる必要がある。効果は大きいが、9位のように学習augmentationが不安定でTTAだけを残す例もあり、元データだけで検証することが重要だった。

### GBDTとtabular DL

CatBoostはcategoryと非線形関係を扱いやすく、上位の事実上の標準だった。LightGBM Dartは1位・9位で強い多様性源、13位はXGBoostの `grow_policy='lossguide'`、深い木、column subsamplingを有効化した。1位のTabMはtree由来特徴量なしでLightGBM級となり、piecewise-linear embeddingが効いた。

### Group CVとstacking

未知rulesetへの汎化を測るため `GameRulesetName` GroupKFoldが中心だったが、Game単位や試合数層化も試された。3位は第1段階OOFを第2段階へ、6位はleakを避けるnested CV、11位はfoldごとに異なるモデルsubsetをOptuna選択した。

### 分布補正

RMSEモデルの予測は0へ縮みやすい。係数1.12～1.3、ensemble重み合計約1.14～1.17、`[-0.98,0.98]` clipなどが現れた。1位は固定scaleよりOOFでfitしたisotonic regressionの方がCV–LB相関に優れた。

## コンペ概要

入力は1,000超の2人・逐次・zero-sum・完全情報board gameについて、game/ruleset特徴量と2つのMCTS agent設定を並べたtabularデータである。出力はagent1のutility、評価はRMSE。hidden testは約60,000行で、100行batchのEvaluation APIを通して推論する。

難所は、未知gameへの汎化、800列超のうちconstant・noisy・相関列が多いこと、gameごとの観測数差、random simulation由来特徴量のnoise、CVとLeaderboardの弱い相関、そして限られた推論時間内での追加計算だった。

## 取得状況

| 最終順位 | Team | Private score | Solution discussion | Status |
|---:|---|---:|---|---|
| 1 | James Day | 0.41780 | [1st Place Solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549801) | found |
| 2 | Richard_U | 0.41996 | [2nd place solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549718) | found |
| 3 | senkin13 | 0.42080 | [two stage flip augmentation stacking](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549588) | found |
| 4 | Manuel Campos | 0.42192 | [4th Place Solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549603) | found |
| 5 | dümensemble | 0.42224 | [5th Place Solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549585) | found |
| 6 | Vadim Timakin | 0.42283 | [The 6th place solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549582) | found |
| 7 | gezi | 0.42363 | [Ensemble of Tree + NN](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549617) | found |
| 8 | Kohei | 0.42374 | [a 10-day challenge](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549616) | found |
| 9 | No Overfitting, Just Skills | 0.42451 | [Various Augmentations + Modeling tricks](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549624) | found |
| 10 | DO THE HARLEM SHAKE | 0.42458 | [10th place solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549605) | found |
| 11 | Kansai-kaggler | 0.42490 | [11th place solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549708) | found |
| 12 | DeadKey | 0.42564 | [12 Place Solution](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/550400) | found |
| 13 | G&G | 0.42591 | [XGBoost ensemble](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549781) | found |

## 解法マップ

| Rank | 主モデル | 主要な独自要素 | CV / ensemble | 後処理 |
|---:|---|---|---|---|
| 1 | CatBoost, LGBM Dart, TabM | 15秒tree search、追加ruleset | 10-fold×2 seed | isotonic regression |
| 2 | 4 GBDT + 3 NN | ruleset clustering、flip | 5-fold、6 seed set | OLS、負重み、clip |
| 3 | LGBM + CatBoost | 2段階flip OOF | StratifiedGroupKFold | ×1.12 |
| 4 | 公開11解法 | cascade merging | Public LB逐次fit | ×1.25、clip |
| 5 | CatBoost/LGBM/DeepTables | 2系統pipeline、flip TTA | Group 5/6/10-fold | weighted blend |
| 6 | CatBoost中心 + LGBM/DNN | Balance反転、OpenFE | nested 5×5 | distribution match |
| 7 | LGB + NN | 上位20×20交差特徴量 | Game GKF | `a*x+b` |
| 8 | CatBoost | 10日、polars高速FE | 10-fold | ×1.2 |
| 9 | CatBoost + 2 LGBM Dart | 多様なaugmentation検証 | 8-fold GKF | TTA、weight/clip最適化 |
| 10 | CatBoost | AdvantageP1 10-bin | 10-fold | ×1.25 |
| 11 | CatBoost系stacking | 3人の異質な特徴量 | fold別Optuna stacking | Nelder–Mead |
| 12 | DeBERTa + LGBM + 公開model | LudRulesからAdvantage改善 | OOF、agent-pair層化 | ensemble |
| 13 | XGBoost 30本 | Advantage interaction、強subsample | 3×10-fold | linear weights + median |

## 個別解法

### 1位 James Day

ゲーム開始局面の短いMCTS searchから均衡度と探索速度を測り、提供特徴量の限界を直接補った唯一性の高い解法。GAVELとLLMで484 ruleset・14,365行を追加し、noiseをsample weightで制御。CatBoost/LGBM Dart/TabMへisotonic regressionを重ね、CVを慎重に再検証したTrust CV ensembleがPrivate 0.4178となった。Q&Aでは、searchがtest labelを近似する禁止simulationでなく短時間の追加特徴量生成であるとホストも確認した。

### 2位 Richard_U

複雑なFEよりflipと大規模model diversityへ集中。rulesetをTF-IDF+KMeansでまとめ、4 GBDTと3 NN、6 seed setをOLSで結合した。負重みを含むPublic LB fit版がCV版を上回りPrivate 0.41996。Public/Privateに一部共通rulesetがあった構造を結果的に活用した。

### 3位 senkin13

flipで第1段階を学習し、そのOOFを第2段階の特徴量へする簡潔なstacking。LGBM/CatBoostを3 seed平均し、Leaderboard probingで予測を1.12倍。Privateのshake-upが小さいことを利用してLBを強く信頼した。

### 4位 Manuel Campos

11の公開Notebook予測をcascade状に順次加減し、各段階をPublic LBへ手動fit。最後に1.25倍とclipを適用しPrivate 4位。CV型stackingは20位相当で、モデル開発よりpublic knowledgeの選択・多様性・残差補正が価値を生んだ例。

### 5位 dümensemble

Anil側のCatBoost+flip TTAと、Sercan側のCatBoost/LGBM/DeepTables・TF-IDFを重み付きblend。単純なagent対称性と、tree/NN/textの異質性を組み合わせた。

### 6位 Vadim Timakin

flip時に `Balance` も反転して元・拡張OOFのgapを縮め、nested CVのCatBoost OOFをLGBM/DNNへ渡した。OpenFE上位特徴量も使用。ただしPrivate最良はCV最良ensembleでなくPublic分布へscale+clipしたCatBoostだった。

### 7位 gezi

LGBM上位20数値特徴量の全積・商がtree性能を大きく改善。LGBとleaf特徴量付きNNをblendしPrivate 0.423。Q&Aでは、候補特徴量の選び方が重要でCatBoost importanceでは悪化したこと、agent categoryごとのtarget meanがLBだけ0.001改善したことを説明。

### 8位 Kohei

10日で公開baselineをpolars化し前処理を410秒から7秒へ短縮、その速度を多数のgame-rule FEへ投資。単一CatBoost 10-fold平均、flip、1.2倍でPrivate 0.423。複雑な推移律augmentationは最終的に効かなかった。

### 9位 No Overfitting, Just Skills

flip/self-play/transitivityを広く検証し、学習augmentationは不安定として元データ学習+flip TTAへ収束。CatBoostと2つのLGBM Dart、変形 `AdvantageP1` を結合。Q&AではTTAを2視点予測の平均として具体式で説明した。

### 10位 DO THE HARLEM SHAKE

公開CatBoostへflip、`AdvantageP1` 10-bin、10 foldsを追加し、予測を1.25倍。Public 0.427から0.417へ段階的に改善した。NNやpseudo labelより、noiseの強い最重要特徴量の安定化が効いた。

### 11位 Kansai-kaggler

3人が少数精選CatBoost、追加生成+multi-target、rule section TF-IDFなど異質なモデルを作り、foldごとにOptunaでsubsetを選ぶstacking。CV 0.3912まで改善したがPrivate 0.424で、CV varianceの難しさも示した。

### 12位 DeadKey

`LudRules` を読むDeBERTaでrandom MCTS対戦の結果を推定し、LightGBMで補正して `AdvantageP1` を改善。公開MCTS Starter/DeepTablesとensemble。Q&AではOOF予測、agent label encoding、agent-pair層化が最も効いたと補足した。

### 13位 G&G

XGBoostを `enable_categorical=True`、`lossguide`、深い木、強いcolumn subsamplingで成立させた。3組×10-foldの線形結合のmedianを採用。重み合計は約1.16で、予測scale補正も暗黙に実現した。

## 横断比較

**source-backedな共通点**として、flip、group CV、CatBoost、予測scale補正が多数解法に現れる。`AdvantageP1` はほぼ全員が重要視し、単純利用より変換・binning・interaction・再推定が上位差を作った。一方でrule text、追加生成、NNはチームごとに成否が分かれた。

**差分**は情報をどこから増やしたかにある。1位はゲームengineを短時間動かすdomain feature、3・6位はOOF stacking、7・13位は数値interaction、11・12位はrule表現、4位は公開予測そのものを情報源にした。

**順位傾向としての推論**は、上位ほど単体モデルの交換よりも、対称性・分布・検証のどれかを明示的に扱っていることだ。ただし2・4・6位のLB fit成功と1位のTrust CV成功が同時に存在するため、「CVかLBか」という一般則ではなく、test split構造と予測biasを診断して選ぶ必要がある。
