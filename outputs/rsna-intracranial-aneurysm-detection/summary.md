# RSNA Intracranial Aneurysm Detection — 上位解法の事前読解ガイド

- **対象最終順位:** 1〜11位（Kaggle CLIで2026-07-07に取得したfinal leaderboard基準）
- **開催期間:** 2025-07-28〜2025-10-14 23:59 UTC
- **取得日:** 2026-07-07
- **目的:** 個別write-upを精読する前に、上位解法の全体像、共通点、差分を理解する。
- **取得方法:** Kaggle CLIでcompetition pages、final leaderboard、全Discussion list、選定Discussion本文と全commentを取得した。

> **順位表について:** 取得時点のKaggle CLI leaderboard先頭には、competition終了時の順位と整合しないscore 1.00000の`perfect_submission.parquet`が含まれていた。生の取得結果は`.work/leaderboard-cli-current.json`へ保存し、本資料と成果物ではこの異常行を除外した順位を使用する。

## 主要な結論

1. **勝負を決めたのは、全volumeをそのまま分類することより「aneurysmがあり得る領域へ計算を集中する設計」だった。** 3D vessel ROI、brain/skull crop、固定CoW cropなど実装は異なるが、ほぼ全解法がbackgroundを減らした。
2. **localization supervisionがclassificationを強くした。** vessel segmentation、aneurysm center、bounding box、heatmap/blob、voxel-wise lossを使い、series labelだけでは得にくい局所featureを学ばせた。
3. **3Dだけが正解ではない。** 上位にはnnU-Net/3D ResNet系だけでなく、2.5D CNN + sequence modelも多い。2.5Dは高resolution、計算効率、pretrained 2D backboneを活かせた。
4. **modality、orientation、spacing、multi-frame DICOMへの耐性がmodel精度と同程度に重要だった。** DICOM変換失敗、missing tag、thick slice、T2/T1-postの見え方、左右labelの扱いが各pipelineの成否を左右した。
5. **極端なclass imbalanceには「loss weightだけ」よりsamplingと補助taskが効いた。** hard-negative sampling、vessel領域からのnegative選択、rare-class oversampling、sphere/heatmap regressionが繰り返し使われた。
6. **evaluation APIの12時間制限がensemble設計を直接制約した。** fold数やTTAを減らしたteamが複数あり、coarse-to-fine crop、patch間引き、dual GPU処理はaccuracyだけでなく完走率を上げる技術だった。
7. **CV–LB gapは大きく、単一のvalidation設計を絶対視できなかった。** modality stratificationやmultilabel stratificationが使われた一方、Public/Private shift、abnormal data、runtime failureが最終順位へ影響した。

## コンペ固有用語集

- **SeriesInstanceUID:** 1つのscan seriesを識別するID。train.csvの1行およびtest APIの予測単位。
- **SOPInstanceUID:** series内の個別DICOM imageを識別するID。
- **CTA / MRA / MRI T1post / MRI T2:** 本competitionに混在する撮像modality。intensity特性、slice spacing、orientationが異なる。
- **CoW（Circle of Willis）:** 脳底部の動脈輪。13のaneurysm locationとvessel segmentationを理解する中心的anatomy。
- **13 locations:** left/right ICA、MCA、ACA、PCom、ACom、Basilar Tip、Other Posterior Circulationからなるlocation target。
- **Aneurysm Present（AP）:** series内のどこかにaneurysmがあるかを示す14番目のtarget。
- **train_localizers.csv:** aneurysmの`SOPInstanceUID`、xy center coordinate、locationを与えるlocalization label。
- **segmentations/:** training subsetに提供された13-class vessel NIfTI mask。
- **weighted multilabel AUC:** 14 labelのAUCを計算し、APへweight 13、各locationへweight 1を与える。実質的にAP AUCと13-location macro AUCの平均。
- **2.5D:** 隣接sliceをRGB channelなどへstackし、2D backboneへdepth contextを与える表現。
- **MIP:** Maximum Intensity Projection。volumeを特定axisへ投影し、vesselなど高intensity構造を2Dで強調する。

## 頻出技術スタック

| Technology | 使用順位 | 概要 | このcompetitionで生んだ価値 |
|---|---|---|---|
| ROI cropping / coarse-to-fine | 1〜9, 11 | vessel、brain、skull、固定anatomical regionでinputを絞る | tiny aneurysmのsignal-to-noise改善、high resolution化、runtime短縮 |
| 2.5D CNN | 4〜6, 8〜9, 11 | 隣接sliceをchannelとして2D modelへ入力 | 2D pretrained backboneとdepth contextを低costで両立 |
| nnU-Net / 3D U-Net | 1〜3, 6〜7, 9, 11 | 自動構成型3D segmentationまたはROI model | vessel/aneurysm localization、orientation correction、crop生成 |
| Localization auxiliary task | 1〜5, 7, 9, 11 | mask、sphere、heatmap、box、voxel-wise head | series labelだけより局所aneurysm featureを学びやすい |
| Hard-negative / targeted sampling | 4〜6, 8, 11 | false positiveやvessel sliceを重点sample | 1:250級のslice imbalanceに対処 |
| Sequence model | 6, 8, 11 | frame featureをTransformer/RNNでseries predictionへ統合 | slice間の位置・depth関係を保持 |
| TTA / fold ensemble | 1〜7, 9, 11 | flip、multi-fold、異種model平均 | modality shiftとmodel varianceを低減。ただしruntimeとのtrade-off |

### ROI cropping

ROIは共通しているが作り方は多様だった。1位はcoarse vessel segmentation、2位はtri-axial 2D nnU-Net、3位はsagittal/coronal YOLO、4位はDINOv3 coordinate regression、5位はbrain detector、7位は固定中央上部crop、11位はTotalSegmentatorで作ったbrain maskを使う。共通目的はaneurysmが数pixel〜小領域しか占めないvolumeからirrelevant backgroundを除くことだった。

### 2.5Dとsequence aggregation

2.5Dは隣接sliceをRGBへ入れて局所depth変化を捉える。6位はframe CNN後にsequence model、8位はConvNeXt feature後にTransformer、9位はYOLO detection、11位はCoW/aneurysm feature抽出へ使った。3D volume全体より軽く、512×512などの高いin-plane resolutionを維持しやすい。

### Segmentation / heatmap supervision

segmentationの最終Diceを競うのではなく、classifierのstructural guideやsampling maskとして使う解法が多い。1位Q&Aではvessel Dice約0.70でもsmall branchのrecallを優先したと説明され、7位はsemantic segmentation形式をEDT blob regressionへ転用した。

### Samplingとimbalance対策

negative sliceが圧倒的に多いため、全sliceを均等に扱うとaneurysmを学習しにくい。5位はlocalizer周辺のpositiveと全negativeを使い、6位はvessel segmentationで良いnegativeを選び、8位はfalse-positive OOF位置を再sampleし、11位は3 roundのhard-negative samplingを行った。rare locationにはoversampling、class weight、focal系lossも使われた。

## Competition overview

複数modalityのbrain DICOM seriesから、13 anatomical locationごとのaneurysm有無と`Aneurysm Present`を予測するmulti-label classification taskである。training dataにはseries label、全aneurysmのcenter localizer、subsetの13-class vessel segmentationがある。testは約2500 seriesをevaluation APIが1 seriesずつ渡し、GPU notebookは12時間以内、internetなしで完走する必要がある。

metricはweighted multilabel ROC-AUCで、AP AUCがfinal scoreの半分、13-location macro AUCが残り半分を占める。主な難所はtiny lesion、強いclass imbalance、撮像modality/施設/protocol差、DICOM irregularity、左右anatomy、large 3D volumeと12時間制限である。

## 取得状況

| Final rank | Team | Private score | Solution discussion | Status |
|---:|---|---:|---|---|
| 1 | tomoon33 | 0.87912 | [1st Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611846) | found |
| 2 | BraveCoWCoW | 0.86732 | [2nd Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611867) | found |
| 3 | BTYND | 0.85065 | [3rd place solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611856) | found |
| 4 | Harshit Sheoran | 0.84750 | [4th Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611893) | found |
| 5 | more CV challenge pls | 0.84746 | [5th place solution with code](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611849) | found |
| 6 | Ian, Theo & Bartley | 0.84007 | [6th Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611925) | found |
| 7 | MIC-DKFZ | 0.83173 | [7th place solution - 3D nnU-Net + blob regression](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/612039) | found |
| 8 | Konni | 0.82471 | [8th Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/613534) | found |
| 9 | Vibes and Genius Trade-Off | 0.82300 | [9th place solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/611908) | found |
| 10 | Dieter | 0.82206 | — | CLIで取得した全Discussion内では対応する一次解説を確認できず |
| 11 | ←→AB | 0.82201 | [11th Place Solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/612186) | found |

## Solution map

| Rank | Main representation/model | ROI/localization | Validation | Ensemble/post-process |
|---:|---|---|---|---|
| 1 | pretrained nnU-Net backbone + location-aware Transformer | 3-stage vessel segmentation、masked pooling | multilabel-stratified 5-fold | 4 folds + LR flip TTA |
| 2 | 3D nnU-Net multi-task | tri-axial 2D nnU-Net ROI | modality-stratified 5-fold | 2 folds + 8x TTA |
| 3 | 3D ResNet-18 voxel-wise classifier | sagittal/coronal YOLO vessel box | fold ensemble | 11 models、Top-N mean |
| 4 | CoaT/MaxViT 2.5D classifier | DINOv3 crop coordinate regression | OOF AUC | max over slices、pseudo-distillation |
| 5 | ViT/EVA + MIT-B4 FPN 2.5D | brain YOLO + aneurysm YOLO | OOF AUC | 6-model weighted average |
| 6 | CoAtNet/MaxViT 2D + sequence model | skull crop、vessel maskでsampling | CV 0.895 | 3-model sequence ensemble |
| 7 | 3D nnU-Net EDT blob regression | fixed superior ROI | modality-stratified 5-fold | single model、patch max |
| 8 | ConvNeXt DINOv3 + Transformer | hard-negative spot sampling | OOF feature pipeline | 4 models、dynamic frame thinning |
| 9 | YOLO 2.5D + EfficientV2s/3D CenterNet + GBDT | boxes、heatmap、vessel aux mask | multilabel-stratified 5-fold | 2-fold base models + 3 GBDT |
| 11 | CoAtNet/EfficientNet 2.5D + RNN | TotalSegmentator brain ROI、keypoint | 5-fold OOF | 11 aneurysm models + 45 RNNs |

## 個別解法サマリー

### 1位 — tomoon33

**一言:** vessel anatomyを明示的なmapとしてclassifierへ渡す、最も構造化されたcoarse-to-fine 3D pipeline。

low-resolution nnU-NetでROIを見つけ、balanced/recall-focusedの2 fine segmentationでvessel maskを作る。segmentation-pretrained nnU-Net classifierはaneurysm sphere auxiliary loss、location-wise masked pooling、Location-Aware Transformerを使う。full-resolution ablation score 0.916で、backbone pretrainingを外すと0.794まで低下した。Q&Aでは10 aneurysmがROI外だったがcomputeとのtrade-offで許容し、coarse-to-fineは大きなCTAでむしろ高速かつmemory-stableだったと説明した。

### 2位 — BraveCoWCoW

**一言:** tri-axial 2D ROI extractionと3D nnU-Net multi-task learningを、DICOM例外処理まで含めて汎用化した解法。

各axis 3 sliceからvascular ROIを推定し、全dataを224³へ統一。vessel/aneurysm segmentation、classification、modality head、cross-attention poolingを共同学習した。annotation refinementと8x TTAが大きく、single-fold baselineのPrivate 0.81268からensemble 0.86727へ改善した。Q&Aでは各axis 1 sliceでも多くの場合十分だが、3 sliceはoutlier耐性のためと説明した。

### 3位 — BTYND

**一言:** 2-view YOLOで3D vessel ROIを作り、high-resolution feature map上で直接14-class predictionする3D ResNet解法。

sagittal/coronal中央sliceからYOLOでboxを検出し、90³/120³ mm ROIをcrop。volume-level headではなく各feature mapへclassification headを付け、Top-N meanでcase predictionへ集約した。feature mapを4³から25³へ高resolution化したことが大きい。missing DICOM tag対策としてEfficientNet V2 Sでvoxel spacingを回帰し、Privateを0.84から0.85へ改善した。

### 4位 — Harshit Sheoran

**一言:** 14日間で構築した、DINOv3 crop + CoaT-Lite-Medium 2.5D classifierという非常に簡潔なpipeline。

48 slice volumeからDINOv3 ViTがcrop coordinateを回帰し、95%超のaneurysmをROI内へ保持。545k slice中positive約2.2kというimbalanceをweighted samplingなしで学習し、rotation、2.5D、soft pseudo-distillation、MaxViT ensembleでCV 0.805から0.896へ伸ばした。Q&AではROI cropがbaselineを0.7から0.8へ押し上げたと述べた。

### 5位 — more CV challenge pls

**一言:** 手作業box、brain crop、external pseudo-label、multi-task segmentationを積み上げた2.5D ensemble。

localizer前後±10 sliceへaneurysm boxをannotateし、MRI T2をdark appearanceの別classとしてYOLO学習。brain detector cropで0.03〜0.05改善し、ViT/EVA classifierとMIT-B4 FPNをcleaned RSNA + external TOF-MRAへ拡張した。33 negative caseをpositiveへrelabelし、最良single componentはExp5のOOF 0.8629だった。

### 6位 — Ian, Theo & Bartley

**一言:** vessel segmentationを最終予測ではなくnegative samplingへ使い、2D CNNとsequence modelを極限まで磨いた2.5D解法。

skull crop後、vessel maskで良いnegative frameを選び、CoAtNet/MaxViTへrelative-position-aware poolingを適用。隣接frame、spacing-aware sampling、左右label swap、手修正localizer maskが各約+0.01 CV。frame maxだけで0.87、sequence modelで0.88、3-model ensembleで0.895 CVへ達した。

### 7位 — MIC-DKFZ

**一言:** 3D segmentation frameworkを14-channel EDT blob regressionへ最小限の変更で転用した単一model解法。

固定`[200,160,160] mm` ROIをmedian spacingへresampleし、ResEnc nnU-NetをTopK 20% BCEで学習。13 anatomical blobとそのmaxであるAP channelをpatch/space方向へmax aggregationする。single model・TTAなしでPublic/Private 0.83/0.83、約8時間。Q&AではEDT radius 65がfold 0で0.896と最良だった。

### 8位 — Konni

**一言:** segmentationもresamplingも使わず、aneurysm spotをoutlierとして学ぶ2-stage 2D/Transformer解法。

ConvNeXt Base DINOv3でslice spotを分類し、OOF false positiveをhard-negativeとして再sample。feature sequenceをTransformerへ渡す。4 model、2種類のslice stepをensembleし、長いseriesは192 frame以下になるよう動的に間引いた。

### 9位 — Vibes and Genius Trade-Off

**一言:** YOLO 2.5D、2D encoder + 3D CenterNet、GBDT meta-classifierを重ねた異種stacking解法。

YOLO11m/EfficientNetV2-Sが13 location boxを予測し、Flayerはslice-wise EfficientNet featureを浅い3D headでheatmap/offsetへ変換。LightGBM/XGBoost/CatBoostは全location・全model予測とmetadataをまとめる。parallel stackingは5-fold CV 0.858、最終2-fold Private 0.8230。Q&Aではpure 3Dを避けた主因をcomputeとtraining timeと説明した。

### 11位 — ←→AB（投稿者: RihanPiggy）

**一言:** brain crop、2つの2.5D feature extractor、45個のRNNをつないだdepth-sequence ensemble。

TotalSegmentatorで作った4386 brain maskから3D U-Netを学習してCoW ROIをcropし、2.5D CoW segmentation modelと2.5D aneurysm keypoint modelのfeatureを抽出した。3 roundのhard-negative sampling、Focal loss、OOF self-distillation、EMAを使い、depth方向のfeature sequenceをLSTM、GRU、BiLSTM、BERTで統合した。11 aneurysm modelと45 RNNを2 T4で並列化して12時間以内に完走し、選択submissionはCV 0.8909、Public 0.87689、Private 0.82201だった。Q&AではPublic/Private gapの候補としてsite間domain shiftによるbrain segmentator failureとDICOM tag問題を挙げた。

## Cross-solution comparison

### Source-backed common patterns

- 10解法すべてが、明示的crop、spot sampling、またはpatch処理でfull volume問題を小さくした。
- 1〜7位、9位、11位はlocalization情報をtrainingまたはROI作成へ使用した。
- 2.5D系は4〜6位、8〜9位、11位、3D segmentation/classification系は1〜3位、7位、9位、11位に分かれた。
- 左右flipではleft/right targetもswapする設計が複数解法で使われた。
- submission timeoutやruntime instabilityによりfold/TTA/model数を減らした記述が複数ある。

### Synthesis / inference

- **順位上位ほど3D anatomyを強くmodel化する傾向**はあるが、4〜6位の僅差はwell-tuned 2.5Dが3Dへ十分競争的であることを示す。
- **segmentation品質そのものよりdownstreamでの使い方が重要**と考えられる。maskをclassifier input、ROI、negative sampling、auxiliary targetのどれに使うかで価値が変わった。
- **最も再利用性が高い基本形**は「cheap ROI locator → high-resolution local predictor → series aggregation」であり、locatorとpredictorは2D/3Dのどちらでもよい。
- **再現実験ではまずdata pipelineとsamplingを固定すべき**である。model名だけを合わせても、orientation、spacing、crop、negative ratioが異なると結果は大きく変わる。
