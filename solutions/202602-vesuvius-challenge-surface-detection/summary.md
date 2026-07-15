# Vesuvius Challenge - Surface Detection：上位解法を読む前のガイド

- 対象：最終1〜12位
- 開催期間：2025-11-13〜2026-02-27 23:59 UTC
- 取得日：2026-07-07
- 目的：各write-upを精読する前に、上位解法の共通構造・勝敗を分けた要因・差分を把握する
- 取得方法：Kaggle CLIでcompetition pages、official final leaderboard、全Discussion listing、解法本文・commentを取得。技術的な質問に著者または確認できたteammateが答えたQ&Aのみ採用

## 重要な結論

1. 勝負の中心は「強い3D segmentation model」だけでなく、穴・tunnel・merge・splitをmetricに沿って直すpost-processingだった。1位はpost-processingだけでPrivate .596→.627、3位は約.02〜.03改善した。
2. nnU-Net/ResEnc系が12解法中ほぼ全体を占めた。少ない設計労力で強い3D baseline、augmentation、sliding-window、cascadeを得られたためである。
3. 長時間訓練と大きいcontextが有効だった。上位は1000〜8000 epochs、128³〜288³ patch、またはtrainより大きいinference patchを使った。ただし大patchがSurface Diceを上げてもTopoを落とす例もあり、単純な「大きいほどよい」ではない。
4. 小さな局所誤りがTopoScoreを大きく壊すため、closing、hysteresis、median/Gaussian/anisotropic diffusion、Euler/Betti検出、PCA/height-map補間が大きな価値を持った。
5. Public LBとPrivate/CVの相関が弱く、threshold、fusion、post-processing選択で逆転が多発した。1位はprobability fusionへoverfitし、9位は高CV版をPublic feedbackで捨てた。
6. topology改善には二つの流派があった。予測後に幾何・homologyを直接修復する方法と、SDF・cascade・diffeomorphic refinementなどmodelに形状修正を学習させる方法である。
7. unlabeled/外部データは決定打ではなかった。収束補助やdiversityには使えたが、pseudo-label noiseにより不採用とした上位解法も多い。

## コンペ固有用語

- **Recto / verso**：パピルスを構成する2層。rectoはscroll中心（umbilicus）側を向き水平繊維を持つ。目標は主にrecto surfaceだが、unwrap可能な近似surfaceなら両層を含んでもよい。
- **Surface mask**：CT volume内の薄いパピルス面を表す3D binary mask。通常の物体segmentationより「薄い連続面」のtopologyが重要。
- **SurfaceDice@τ**：予測とGTのsurface点が物理距離τ=2.0以内にある割合。重み0.35。
- **VOI_score**：connected componentのsplit/mergeを測るVariation of Informationをscore化。`1/(1+0.3×VOI_total)`、重み0.35。
- **TopoScore**：Betti matchingによるtopological F1。k=0 component、k=1 tunnel/handle、k=2 cavityを重み.34/.33/.33で評価。重み0.30。
- **Betti-0/1/2**：それぞれconnected component、loop/tunnel、enclosed cavityの数・特徴。
- **Label 2 / ignore region**：評価対象外voxel。境界の扱いにより見かけのloopやcavityが変化し、post-processing検証を難しくした。
- **Sheet merge / split**：隣接する巻き層の誤接続／同一sheetの断裂。Surface Diceが高くてもVOI・Topoを大きく落とす。

## 頻出技術スタック

| Technology | 使用順位 | 概要 | このコンペでの価値 |
|---|---|---|---|
| nnU-Net / ResEnc UNet | 1–4, 6–12 | 3D segmentationを自動構成するframework | 強いbaseline、long training、multi-scale/cascadeを短期間で実現 |
| Model/TTA ensemble | 1–3, 5–8, 10–12 | fold・patch・architecture・flip/rotationの融合 | sheetの欠損を減らし、context差を補完 |
| Morphology / hysteresis / filtering | 1–4, 6–9, 12 | closing、fill、diffusion、Gaussian、threshold | 小穴・noise・断裂を低costで修復 |
| Euler / Betti / persistent homology | 1, 2, 4–6 | loop/cavityの検出とbirth/death座標推定 | metricが直接罰するtopology errorを狙い撃ち |
| Projection / PCA / interpolation | 1, 2, 4, 6 | sheetを2D平面・height mapへ写して穴埋め | 薄い面というdomain geometryを利用 |
| Cascade / learned refinement | 3, 10, 11 | previous predictionを次stageで修正 | heuristicで難しいshape・merge・hole修正を学習 |

### nnU-Net

dataset spacingやpatchを扱う3D medical segmentation frameworkで、本競技では薄いsurfaceにも非常に強かった。上位はdefaultをそのまま使うだけでなく、MedialSurfaceRecall/clDice、ResEnc、lowres→cascade、長期epoch、train/test label処理整合へ拡張した。

### Topology-aware post-processing

metricの30%がTopoScoreで、VOIもsplit/mergeへ反応するため、数voxelのbridge/hole修正がmodel変更以上の改善を生んだ。特に1位のlookup-table hole plug、2位のEuler-guided local interpolation、5位のpersistent-homology tunnel fillingが代表的である。

### SDFとlearned refinement

5位はbinary probabilityでなくsurfaceまでのsigned distanceを回帰し、境界・skeleton・threshold tradeoffを滑らかに扱った。10位は複数段UNetとdiffeomorphic fieldで予測形状そのものを変形した。どちらも単純morphology以外の方向を示す。

## Competition概要

炭化して開けないHerculaneum scrollの3D CT chunkからパピルスsurfaceをsegmentし、virtual unwrappingへ渡す課題。sample sizeは可変で、`.tif` volume maskを元画像と同じshape/data typeで提出する。Scoreは`0.30×TopoScore + 0.35×SurfaceDice@2.0 + 0.35×VOI_score`のtest-volume平均。Notebook-onlyでCPU/GPUとも9時間、internet無効。主な難しさは薄く絡み合うsurface、damaged/frayed layer、隣接sheet merge、holeによるsplit、ignore region、Public subsetの小ささだった。

## 取得状況

| 最終順位 | Team | Private score | Solution discussion | Status |
|---:|---|---:|---|---|
| 1 | Vesuvius Team | 0.62702 | [1st Place Solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679238) | 取得 |
| 2 | risk of overfitting | 0.62250 | [A postprocessing win](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679278) | 取得 |
| 3 | W & A | 0.62090 | [3rd place solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679236) | 取得 |
| 4 | Starry | 0.61986 | [4-th Place Solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679222) | 取得 |
| 5 | Dieter | 0.61941 | [5th Place Solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679360) | 取得 |
| 6 | #hui | 0.61847 | [6-th place solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/680280) | 取得 |
| 7 | DECEM | 0.61812 | [7th Place Solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679251) | 取得 |
| 8 | lingyundev | 0.61697 | [8th-place-solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679248) | 取得 |
| 9 | 阿對對對對隊 | 0.61641 | [SUPER-CONSERVATIVE](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679241) | 取得 |
| 10 | Vibes & Scrolls Trade-off | 0.61490 | [10th place solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679227) | 取得 |
| 11 | Aindriú | 0.61365 | [11th place](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679237) | 取得 |
| 12 | E. Honda. | 0.61307 | [12th place solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/679259) | 取得 |

## Solution map

| Rank | Main model | Representation / loss | Ensemble | 主なpost-processing |
|---:|---|---|---|---|
| 1 | nnU-Net | binary probability | 4 models | component除去、height map、1-voxel plug、closing |
| 2 | nnUNet ResEnc | Dice/CE系 | 128³+160³ | Euler検出、multi-threshold local interpolation |
| 3 | nnU-Net Fullres→Cascade | clDice | checkpoints+rot TTA | inverse EDT、Hessian ridge、Z consistency |
| 4 | huge 7-stage nnUNet | binary | single | PCA/RBF、Betti matching、cavity操作 |
| 5 | SEResNeXt152 AttUNet | SDF L1+mass | architectures/checkpoints | persistent-homology tunnel fill |
| 6 | nnUNet MedialSurfaceRecall | binary | 5 folds+all | watershed、Betti、RBF、scroll別処理 |
| 7 | nnUNet ResEncM lowres | CE+Dice+Skeleton Recall | processed/raw labels | Frangi、CED、anisotropic closing |
| 8 | nnUNet ResEncUNetL | probability | patch 224/256/288 | anisotropic diffusion+hysteresis |
| 9 | nnUNet MedialSurfaceRecall | binary | single fold TTA | line norm、iterative gap/diagonal、Gaussian |
| 10 | ResEnc-L+Primus | multi-stage masks/SDF shift | learned stack | diffeomorphic warp、後日median |
| 11 | lowres→cascade | binary | 2-model geo-mean | public 5-step morphology |
| 12 | nnUNet lowres+fullres | binary probability | 4 models | hysteresis、opening/closing |

## 個別解法

### 1位 Vesuvius Team

長期nnU-Netのmulti-patch ensembleに、穴サイズ別の幾何修復を積み重ねた王道の完成形。128 patchを4000 epochs訓練し、192/256 fine-tuneやscratch 192をweight融合。20K未満除去、height-map interpolation、2³ lookup tableによる1-voxel hole plug、binary closingを順に適用しPrivate .596→.627。touching sheetは未解決で、Publicを信じたprobability fusionと小thresholdはPrivate上は最適でなかった。

### 2位 risk of overfitting

modelよりlocal interpolationの設計で迫った。128³/160³ nnUNetを40/60で融合し、Euler numberで問題patchだけ見つけ、高threshold maskへ切り替えながら2D投影補間。global版はPrivate .631だったがlabel-2 riskとPublic/CVで選べず、local版が最終提出になった。

### 3位 W & A

train 128・inference 192の非対称patchとFullres→Cascadeを8000+8000 epochsまで訓練。±15° rotation TTA、inverse EDT、Hessian ridge、Z-axis consistency pruningで約.02〜.03改善。全データ訓練を選びpseudo-labelは不採用。

### 4位 Starry

巨大nnUNetにPCA projectionとBetti matchingを重ねた。large holeをPCA平面でfillし、20³ local blockのBetti-1 errorを修復。Betti-2を人工操作するmetric-specific処理はPublic +.007に対しPrivate +.001で、効き方のdataset依存を示した。

### 5位 Dieter

上位で最も異質な、custom SEResNeXt152 Attention UNetによるSDF回帰。full 320³ inferenceでsliding-window artifactを避け、birth/death座標へbridge guard付きball fillを13回反復。SDF thresholdでsurface/merge tradeoffを直接制御し、binary probability modelよりtopologyが良かった。

### 6位 #hui

6-model nnUNet ensembleに多数のtopology operationを組み合わせた。Betti輪郭、Dijkstraでloopをedgeへ開く処理、RBF、cavity生成まで使い、scroll densityとignore regionに応じ処理強度を変更。merged sheetは残る最大の難所だった。

### 7位 DECEM

test pipelineと同じ処理をtrain labelへ適用したことが最大の発見。raw-label modelとensembleし、Frangi sheetness→CED→anisotropic closing。Q&Aのcontrolled comparisonではSGD .590、AdamW .594、Muon+AdamW .595だった。

### 8位 lingyundev

224/256/288 patchのscale diversityを単純平均し、probability mapへPerona–Malik系diffusionを適用。z方向だけのhysteresis connectivityで薄いlayerを守りつつ横方向のover-expansionを防いだ。

### 9位 阿對對對對隊

小さな局所ルールを反復してTopoを積み上げた。skeleton+EDT line normalization、gap sandwich、diagonal kernelを5回、Gaussian filter、2×2 bridgeを適用しCV .6096→.6337。Public feedbackで捨てたline completion版は競技後CV .6391だった。

### 10位 Vibes & Scrolls Trade-off

3 initial modelsをResEnc-Lでstackし、さらにrefinement、最後にSVFを予測するdiffeomorphic networkでshape/thicknessを調整。learned geometry correctionはSurface Diceを改善し、競技後にmedian filterを足すとPrivate .614→.624まで上がった。

### 11位 Aindriú

2 lowres modelのgeometric mean→3D Cascade Fullresという簡潔な構成。cascadeが実質的なlearned post-processingを担い、簡単なthresholdだけで20〜30位圏。post-process radiusの小変更は視覚差なしでも3位相当になり得た。

### 12位 E. Honda.

lowres/fullres各2 foldの4-model ensemble。hysteresis、opening、anisotropic closingで単体CV .571→.606、Topo .246→.342。1000→4000 epochs、TTA、multi-resolutionを順に積みPrivate .552→.613へ改善した。

## 横断比較

source-backedな共通点は、nnU-Net優勢、長期訓練、TTA/ensemble、topology post-processingの大きな寄与、Public/Private不一致である。一方、どの修復が最良かは予測errorの性質で変わった。1位の大穴height mapは見た目ほどscore寄与がなく、10位の強いmodelには単純medianが後から大幅に効いた。

ここからの統合的な解釈として、順位差は「modelの平均精度」より、各modelが残すerror distributionと修復器の相性で決まった可能性が高い。上位ほど、全voxelを一様に平滑化するのでなく、component、Euler/Betti、threshold、projectionを使って危険箇所だけを選別し、Surface Diceを保ちながらTopo/VOIを上げている。5位・10位はこの選別をSDFやlearned deformationへ内在化した別ルートであり、今後のsurface segmentationへ特に一般化しやすい示唆を持つ。
