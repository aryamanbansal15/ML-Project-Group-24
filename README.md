# Listening Past the Noise
### Audio Classification Under Noisy Labels

### INDIVIDUAL CONTRIBUTION

AASTHA NAGARKOTI (2024013) : SUBMISSION1, notebook-3.ipynb,SUBMISSION5:Notebook 4 Updated — Spectral Residual Blend, Notebook 6 — Fisher Vector Audio Classification, NOTEBOOK 7

## Project Overview

This project is based on Problem Statement 4 of the Machine Learning course project, *Listening Past the Noise*.

The objective is to classify short audio recordings into one of 41 everyday sound categories. The main challenge is that most training labels are unchecked and may contain errors, while only a small subset of labels is verified.

This repository currently contains the implementation and results of **Submission 1**.

## Dataset

- Training samples: 5,604
- Test samples: 1,592
- Categories: 41
- Verified training samples: 620
- Evaluation metric: Mean Average Precision at 3 (MAP@3)

## Methodology

The notebook implements the following pipeline:

1. Audio data exploration and preprocessing.
2. Handcrafted audio feature extraction using Librosa.
3. Feature engineering and feature selection.
4. Experiments with noisy-label handling.
5. Comparison of classical machine learning models.
6. Ensemble learning using weighted probability averaging.
7. Generation and validation of top-three test predictions.

## Feature Engineering

The selected feature set consists of **89 audio features**:

- 75 original features, including MFCC statistics, chroma statistics, spectral centroid, bandwidth, rolloff, zero-crossing rate, RMS energy, and duration.
- 14 spectral contrast features.

Experiments with additional MFCC delta and delta-delta features did not improve the selected configuration.

## Models Evaluated

The following models were explored:

- RBF Support Vector Machine
- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- CatBoost

## Selected Ensemble — Submission 1

The first submission uses a weighted ensemble of four classical machine learning models:

| Model | Weight |
|---|---:|
| RBF Support Vector Machine | 40% |
| Random Forest | 30% |
| Extra Trees | 20% |
| Logistic Regression | 10% |

The models were trained using the 620 verified training samples and the selected 89-feature representation.

## Submission 1 — Results

| Evaluation | MAP@3 |
|---|---:|
| Three-seed cross-validation mean | 0.720878 |
| Kaggle Submission 1 score | **0.69479** |

**Submission status:** Successfully submitted and evaluated on Kaggle.

The submission generated predictions for all 1,592 test recordings, with up to three distinct category predictions per recording.

## Repository Structure

```text
listening-past-the-noise/
│
├── notebook.ipynb
├── SUBMISSIONS/
└── README.md
```

- `notebook.ipynb`: Implementation, feature extraction, experiments, model comparison, evaluation, and the first submission pipeline.
- `SUBMISSIONS/`: Competition submission files.

## Tools and Technologies

- Python
- NumPy and Pandas
- Librosa
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib and Seaborn
- Jupyter Notebook
- Kaggle

## Next Steps

Further experiments will focus on:
- Hyperparameter tuning of the selected models.
- Optimising ensemble weights.
- Exploring additional handcrafted audio features.
- Improving noisy-label handling.
- Evaluating improvements through cross-validation before subsequent submissions.

## Acknowledgements

Machine Learning Course Project — Problem Statement 4: *Listening Past the Noise*.

## SUBMISSION_4
## P4 — Listening Past the Noise

**Notebook:** `notebook-3.ipynb`  
**Task:** Rank the three most likely sound categories for each recording.  
**Result:** Model 3 achieved **0.73488 public MAP@3** on Kaggle, up from **0.69479** for the original submission.

### Dataset and objective

The dataset contains **5,604 training recordings**, **1,592 test recordings**, and **41 sound categories**. Each recording belongs to one category. The main difficulty is label quality. Only **620 training labels** are verified (`checked = 1`). The remaining **4,984** are unverified and may contain errors.

The notebook investigates two questions. Can the unverified recordings improve a model trained on the smaller verified set? And do more detailed audio features help the classifier rank the correct category higher? Three models are evaluated on the same validation folds to track the contribution of each change.

### Audio feature extraction

Each WAV file is converted into a fixed-length feature vector so recordings of different durations can be passed to the same classifier. The extractor computes a short-time Fourier transform using a **2,048-sample window** and a **512-sample hop**, then reuses the spectrum to calculate several features.

| Feature set | Dimensions | What it captures |
|---|---:|---|
| Baseline | 89 | MFCCs, chroma, spectral centroid, bandwidth, roll-off, spectral contrast, zero-crossing rate, RMS energy, and recording duration |
| Extended | 319 | All baseline features, plus log-mel band statistics, MFCC changes over time, spectral flatness, onset strength and frequency, energy percentiles, and the fraction of active frames |

The extended feature set treats quiet and active portions of a recording separately. For example, it calculates the mean, standard deviation, and 90th percentile of log-mel energy over active frames. This helps preserve information about a short sound that occupies only part of a clip. Feature extraction completed for **all 7,196 recordings**. The notebook caches the arrays with their recording IDs so later runs can reuse them and verify row alignment.

### Models and handling of unverified labels

| Model | Training data | Features and classifier |
|---|---|---|
| **Model 1 — Verified baseline** | 620 verified recordings | 89 features. Weighted ensemble of an RBF SVM, logistic regression, random forest, and Extra Trees |
| **Model 2 — Filtered labels** | Verified recordings plus selected unverified recordings | Keeps the 89-feature ensemble and tests whether including lower-weight examples improves classification |
| **Model 3 — Extended features** | Verified recordings plus selected unverified recordings | Uses 319 features and combines two RBF SVMs, logistic regression, random forest, and Extra Trees |

The selected filtering rule keeps an unverified recording when a model trained on verified recordings ranks its supplied label first. Each retained example receives a training weight of **0.5**, compared with **1.0** for a verified example. During cross-validation, the model that makes this decision is trained only on the verified training portion of that fold. Agreement is used as a confidence signal and does not establish that an unchecked label is correct.

For the final Model 3 fit, **1,838 of the 4,984 unverified recordings** passed the filter. This increased the training set from **620 to 2,458 recordings**. The final ensemble assigned 25% to an RBF SVM with `C=3`, 15% to an RBF SVM with `C=10`, 10% to logistic regression, and 25% each to random forest and Extra Trees.

### Validation and results

The notebook uses **five-fold stratified cross-validation** with random seeds **42** and **2026**. Validation is restricted to verified recordings. The competition metric, **Mean Average Precision at 3 (MAP@3)**, awards 1 point when the correct category ranks first, ½ when it ranks second, ⅓ when it ranks third, and 0 when it falls outside the top three.

| Model | Validation MAP@3 · seed 42 | Validation MAP@3 · seed 2026 | Public Kaggle MAP@3 |
|---|---:|---:|---:|
| Model 1 | 0.7263 | 0.7102 | 0.69479 |
| Model 2 | 0.7422 | 0.7344 | — |
| **Model 3** | **0.7742** | **0.7653** | **0.73488** |

Adding filtered recordings improved validation MAP@3 on both seeds. The larger gain came in the next comparison, when Model 3 introduced the extended features and wider ensemble together. On seed 42, Model 3 ranked the correct category **first for 71.9%** of verified recordings and **somewhere in the top three for 84.2%**. The public leaderboard improvement over the original submission was **0.04009 MAP@3**.

### Error analysis and limitations

The verified set is unevenly distributed across categories. **34 categories have 10 verified recordings each**, while **7 categories have 40 each**. The notebook therefore reports ordinary validation MAP@3 alongside estimates that assign different weights to categories. Both seeds repartition the same 620 verified recordings, so they measure sensitivity to the split rather than performance on two independent datasets.

The error analysis breaks predictions down by category, confusion pair, and recording duration. On seed 42, `C34` and `C35` had the lowest per-category MAP@3, at **0.25** and **0.30**. Each result was based on only ten verified examples. The notebook also records recurring errors, including `C05` and `C26` being predicted as `C19` four times each.

### Running the notebook and locating the output

### SUBMISSION5:Notebook 4 Updated — Spectral Residual Blend

This notebook classifies audio recordings into **41 sound categories** and returns the three most likely categories for each clip. It extends the previous acoustic blend by adding **spectral residuals**, which describe how each short audio window differs from its nearest learned acoustic pattern. Two recordings can contain similar patterns at similar frequencies while still differing in their spectral detail.

The final model averages predictions from three components:

| Component | Weight | What it captures |
|---|---:|---|
| Recording-level ensemble | 45.5% | Whole-clip summaries of spectral content, energy, and timing |
| Short-window histogram SVM | 24.5% | How often learned acoustic patterns occur in each clip |
| Spectral residual SVM | 30.0% | How short windows differ from their assigned cluster centers |

The notebook fits **MiniBatch K-means** dictionaries on unchecked training audio using windows of approximately **0.1 and 0.4 seconds**. For the residual representation, it aggregates the differences between windows and their assigned cluster centers, then uses a linear-kernel SVM for classification. Verified training labels receive full weight. Unchecked labels receive lower weight, with more weight given when they agree with the recording-level model.

### Results

Both blends were evaluated on the same verified recordings using stratified five-fold cross-validation with three random seeds.

| Validation seed | Previous acoustic blend | Spectral residual blend |
|---|---:|---:|
| 42 | 0.79409 | **0.80349** |
| 2026 | 0.79785 | **0.80269** |
| 913 | 0.80242 | **0.80833** |
| **Mean MAP@3** | **0.79812** | **0.80484** |

The spectral residual blend improved validation MAP@3 on all three splits. Its submission scored **0.80731 public MAP@3** on Kaggle, up from the previous submitted score of **0.73488**.

### Output

With the P4 competition dataset attached on Kaggle, the notebook writes `/kaggle/working/submission.csv` with **1,592 rows**. Each row contains a test recording ID and three distinct category codes ordered by predicted likelihood.

The notebook runs on Kaggle with the P4 competition dataset attached. It reads `train.csv`, `test.csv`, `sample_submission.csv`, and the corresponding training and test WAV directories. Audio processing uses `librosa` and `soundfile`, model training uses `scikit-learn`, and `joblib` parallelizes feature extraction.

The selected predictions are written to **`/kaggle/working/submission.csv`**. Model-specific outputs are stored under `/kaggle/working/model_files/`. The final audit checks that the submission contains all **1,592 test IDs** exactly once and that each prediction contains distinct, valid category codes in ranked order.

## Notebook 6 — Fisher Vector Audio Classification

`notebook-6.ipynb` extends Notebook 4’s spectral residual ensemble with a **Fisher-vector SVM**. It ranks the three most likely sound categories for each recording across **41 classes** and achieved approximately **0.8495 public MAP@3** on Kaggle.

### Method

Notebook 4’s residual features measure how short audio windows differ from their nearest acoustic cluster. Notebook 6 uses a Gaussian mixture model so that each window can contribute to several acoustic patterns. The Fisher vector summarizes how the windows in a recording differ from those patterns in both **mean and variance**.

The scaler, PCA, and Gaussian mixture model are fitted on windows from unchecked training recordings. A linear-kernel SVM learns from the resulting recording-level Fisher vectors. The final prediction blends **70% of Notebook 4’s residual ensemble** with **30% of the Fisher-vector SVM**. At inference, the notebook estimates class proportions from unlabelled test predictions and uses them to adjust the final category rankings.

### Results

Both ensembles were evaluated on the same verified recordings using stratified five-fold cross-validation across three seeds.

| Seed | Notebook 4 MAP@3 | Notebook 6 MAP@3 |
|---|---:|---:|
| 42 | 0.80350 | **0.82823** |
| 2026 | 0.80269 | **0.83226** |
| 913 | 0.80833 | **0.83145** |
| **Mean** | **0.80484** | **0.83065** |

The Fisher-vector blend improved validation MAP@3 on every seed. On the public leaderboard, it improved from **0.80731** to approximately **0.8495**.

### Output

With the P4 competition dataset attached on Kaggle, the notebook writes `/kaggle/working/submission.csv` with ranked predictions for all **1,592 test recordings**.

### P4 Audio Classification — Submission Notebook 7
Notebook 7 tests two ways to improve the ranking of sound categories in short audio recordings. The reference is Notebook 6’s Fisher-vector ensemble, which scored 0.84950 public MAP@3. The first experiment adds information about when the sound changes within a clip. The second uses acoustically similar recordings with verified labels to refine the predictions.

The verified-neighbour blend was the submitted model. It scored 0.84528 public MAP@3, a decrease of 0.00422 from Notebook 6 despite improving cross-validation MAP@3 on two seeds.

Data and evaluation
The training set contains 5,604 recordings across 41 categories. Of these, 620 have verified labels and 4,984 have unchecked labels. Validation uses verified recordings, with each candidate evaluated on the same stratified five-fold splits as its reference. The submission contains predictions for 1,592 test recordings.

The competition metric is Mean Average Precision at 3 (MAP@3). Each recording has one correct category. A correct prediction earns 1 point at rank one, ½ at rank two, and ⅓ at rank three. This makes the order of the three submitted labels important even when the set of predictions stays the same.

Experiment 1 — Temporal Fisher vectors
Notebook 6 pools short acoustic windows across an entire recording. That pooling can give similar representations to two clips that contain the same sounds in a different order. Notebook 7 tests whether the difference between Fisher vectors from the first and second halves captures useful information about how a recording changes over time.

The candidate combined 70% of the Notebook 6 probabilities with 30% from the temporal Fisher SVM.

Validation seed	Notebook 6 MAP@3	Temporal blend MAP@3	Change
42	0.82823	0.81559	−0.01263
2026	0.83226	0.82016	−0.01210
913	0.83145	0.82500	−0.00645
The temporal blend scored lower on all three seeds. The whole-recording Fisher ensemble was therefore retained as the reference for the second experiment.

Experiment 2 — Verified acoustic neighbours
The second experiment asks whether similar recordings with trusted labels can help resolve uncertain predictions. For each query recording, the model combines similarities from the spectral-residual and Fisher representations, selects the 12 nearest verified training recordings, and gives closer neighbours more weight. During validation, the neighbour pool contains only verified recordings from the training folds.

The submitted blend used 85% Notebook 6 probabilities and 15% neighbour votes, followed by the existing class-prior adjustment. The paired validation results below were measured before that final adjustment.

Validation seed	Notebook 6 MAP@3	Neighbour blend MAP@3	Change	Folds improved
42	0.82823	0.84839	+0.02016	5 of 5
913	0.83145	0.83952	+0.00806	3 of 5

Submission and findings
Notebook 7’s verified-neighbour blend scored 0.84528 public MAP@3, compared with 0.84950 for Notebook 6. The blend improved cross-validation MAP@3 in 8 of 10 folds, but that improvement did not carry over to the public test split.

A likely explanation is that recordings can sound similar even when they belong to different categories. The neighbour model searched only the 620 verified clips across 41 categories. If a test recording had few close examples from its own category, recordings with a similar background or rhythm could give a competing label enough votes to move ahead. MAP@3 is sensitive to that change: moving the correct label from first to second cuts its contribution from 1 to 0.5.

The validation comparison also measured rankings before the final class-prior adjustment, while Kaggle scored the rankings after that adjustment. The public score alone cannot identify which predictions changed for the worse. Overall, Notebook 6’s original ranking held up better on the public split.
## 1. Baseline Exploration, Feature Selection & SVM Optimization (`svm-optimisation (2).ipynb`)

### Objective
Establish a leak-free, verified-only cross-validation benchmark, systematically compare classical ML families, engineer and ablate recording-level audio features, and test naive pseudo-labeling before building a first ensemble submission.

### Pipeline & Key Experiments
- **Data Audit**: Verified zero missing values, zero duplicate IDs, exact 1-to-1 CSV/WAV alignment across all 5,604 train and 1,592 test files, and uniform 22,050 Hz mono PCM-16 format.
- **Model Comparison (75 baseline features, 5-fold Stratified CV, Seed 42)**:
  - RBF SVM (`C=3.0`): `0.6661` (best single family)
  - Random Forest: `0.6473` | Logistic Regression: `0.6470` | Extra Trees: `0.6441`
- **Feature Ablation**: Adding 14 spectral contrast features (mean/std) boosted RBF SVM from `0.6707` (75 features) to `0.6909` (89 features), whereas adding MFCC delta (`0.6398`) and delta-2 (`0.6422`) degraded performance on the small verified-only set.
- **Hyperparameter Tuning (89 features)**:
  - Fine-grained RBF SVM grid search over `C`, `gamma`, and `class_weight` improved 3-seed mean MAP@3 from `0.6847` (`C=3, gamma="scale"`) to `0.6950` (`C=20, gamma=0.005618, class_weight="balanced"`), peaking at `0.7008` on Seed 42 and `0.7022` on Seed 123.
  - Gradient boosting baselines on verified-only 89 features scored `0.6497` (XGBoost) and `0.6511` (CatBoost).
- **Controlled Pseudo-Labeling Check**: Adding high-confidence (`>= 0.40`) single-model pseudo-labels from unchecked clips slightly hurt on 89 features (`0.6911` vs. `0.6944` verified-only), so unchecked labels were excluded from this submission.

### Final Submission Configuration & Validation
- **Features**: 89 verified recording-level features (75 baseline + 14 spectral contrast).
- **Ensemble**: Weighted probability average — **40% RBF SVM (`C=20, gamma=0.005618`) + 10% Logistic Regression + 30% Random Forest + 20% Extra Trees**.
- **3-Seed OOF MAP@3**: Seed 42 = `0.72634` | Seed 2026 = `0.71022` | Seed 31415 = `0.72608` | **Mean = `0.72088`**.

---

## 2. Acoustic Modulation & Fisher Vectors (`notebook3cb95cbbfb (2).ipynb`)

### Objective
Unlock the 4,984 unchecked training recordings safely via two-stage agreement filtering and move beyond clip-level averages by modeling short-window spectral events (BoAW, VLAD, Fisher vectors) and temporal modulation/inter-band correlation structure.

### Pipeline & Architecture
- **Recording-Level Features (`Frich`, 319 dims)**: Extends the 89 base+contrast features with active-frame 40-band log-mel statistics (mean, std, 90th percentile, max), active MFCC stats, MFCC delta std, spectral flatness, onset strength and onset rate, log-RMS percentiles, activity ratio, and log-duration.
- **Short-Window Representations (2 temporal scales: 8 & 32 frames, ~0.1s & ~0.4s)**:
  - **BoAW (192 dims)**: 96-cluster `MiniBatchKMeans` per scale with 3-NN inverse-distance soft voting and square-root normalization.
  - **VLAD Residual (precomputed linear kernel)**: 32-cluster intra-normalized, signed power-normalized, L2-normalized residuals.
  - **Fisher Vectors (precomputed linear kernel)**: `PCA(24, whiten=True)` + 32-component diagonal `GaussianMixture`, signed power-normalized and L2-normalized.
- **Acoustic Modulation & Band-Correlation (`modulation_v1`, 468 dims)**: 24-band Mel energy envelope mean/std, 276 upper-triangle inter-band correlations, and periodogram power integrated across 6 modulation frequency bands (`0.5–50 Hz`).
- **Two-Stage Noisy-Label Filtering**: A verified-only Stage-1 `Frich` ensemble scores the 4,984 unchecked clips; clips whose given label ranks #1 (`max_rank=1`) receive weight `0.5` (non-matching clips get `0.08` in BoAW/VLAD/Fisher SVMs and are dropped in `Frich` and `modulation`).
- **Blending & Post-Processing**: Fixed blend (`80%` reference ensemble of `Frich` + BoAW SVM + VLAD SVM + Fisher SVM + `20%` Modulation RBF-SVM), followed by temperature sharpening and Saerens EM prior adjustment (`power=0.5`).

### Validation Results (Paired 2-Seed 5-Fold CV)
- **Reference Ensemble (without modulation)**: Seed 42 = `0.82823` | Seed 913 = `0.83145` (**Mean = `0.82984`**).
- **With Acoustic Modulation (+20% weight)**:
  - Seed 42: raw gain `+0.01022` (**`0.83845`**), adjusted gain `+0.00887`.
  - Seed 913: raw gain `+0.00887` (**`0.84032`**), adjusted gain `+0.00995`.
  - **Mean Raw OOF MAP@3 = `0.83938`** (passed the `>= +0.005` adjusted-gain gate on both seeds).

---

## 3. Fast 4-Core Parallel Multi-Kernel, CQT-Fisher, Dual-Teacher EM & Graph Diffusion (`notebooked58be53fd.ipynb`)

### Objective
Maximize MAP@3 within strict classical-ML rules by introducing orthogonal pitch/harmonic representations (CQT), cross-view multi-kernel learning, dual-teacher EM noisy-label cleaning, broader model diversity (gradient boosting, shrinkage LDA, direct softmax), exact class-mass balancing, and cross-fitted stacking with 4-core CPU parallelization.

### Pipeline & Key Upgrades
- **Expanded Multi-View Audio Features**:
  - `Frich_v2_fast` (**515 dims**): Adds fast 40-band Mel-domain HPSS (global and 4-band harmonic/percussive ratios & centroids), MFCC delta/delta-2 quantiles, log-mel temporal deltas, 3-segment temporal energy envelope, crest factor, clipping fraction, spectral flux, and Tonnetz.
  - **Activity-Masked 3-Scale Mel Patches (`8, 24, 64` frames)**: Filters silent windows (`< max - 45 dB`) to mitigate the 8.16s unchecked vs. 5.49s test duration mismatch; encodes 128-cluster BoAW (`384` dims), 32-cluster PCA(48)-VLAD (`4,608` dims), and 48-component GMM Fisher vectors over PCA(32) (`9,216` dims).
  - **Orthogonal Constant-Q Transform (`CQT`, 60 bins / 5 octaves)**: CQT Fisher vectors (`40` GMMs × `28` PCA dims) + `180` clip-level CQT stats.
  - `Modulation_v2` (**540 dims**): Silence-trimmed envelope correlations and periodograms plus per-band skewness, excess kurtosis, and slow/fast modulation ratios.
  - **Positive-Definite Fused Multi-Kernel (`7196 × 7196`)**: Weighted unit-diagonal composite kernel combining Fisher (`0.26`), `Frich_v2` RBF (`0.18`), VLAD (`0.16`), CQT Fisher (`0.14`), `Modulation_v2` RBF (`0.14`), and BoAW RBF (`0.12`).
- **Dual-Teacher EM Noisy-Label Cleaning & Exact Class Balancing**:
  - Combines a 5-view verified-only Stage-1 teacher (OOF MAP@3: `0.78763` Seed 42, `0.79382` Seed 913) with a leak-free 5-fold OOF unchecked-set linear teacher (`0.696` top-1 agreement) via `em_noise` for continuous reliability weights (~4,860–4,898 kept, ~36–54 high-confidence relabeled per fold).
  - Applies exact effective-weight class balancing (`alpha=1.0`) so all 41 classes carry equal training mass.
- **11-Model Diverse Ensemble (10-Fold OOF Plain MAP@3)**:

| Model | Seed 42 | Seed 913 | Mean MAP@3 | Learned Stack Weight |
| :--- | :---: | :---: | :---: | :---: |
| **`fused_mkl`** (6-view Composite Kernel SVM) | `0.83306` | `0.83172` | **`0.83239`** | `0.2191` |
| **`lgbm_dense`** (LightGBM on 1,619 dense features) | `0.81075` | `0.80538` | **`0.80806`** | `0.2296` |
| **`frich_ens`** (SVM3 + SVM10 + LR + RF + ET) | `0.80511` | `0.80484` | **`0.80497`** | `0.1020` |
| **`lda_dense`** (Shrinkage LDA, uniform prior) | `0.79973` | `0.80081` | **`0.80027`** | `0.0899` |
| **`fv_logreg`** (L2 Softmax on PCA-512 joint FV+VLAD+CQT) | `0.79543` | `0.79731` | **`0.79637`** | `0.2595` |
| **`fisher_svm`** (3-scale Mel Fisher linear SVM) | `0.78306` | `0.78602` | **`0.78454`** | `0.0857` |
| **`vlad_svm`** (3-scale Mel PCA-VLAD linear SVM) | `0.75108` | `0.75349` | `0.75228` | `0.0000` |
| **`xgb_dense`** (XGBoost `hist` on PCA-256 dense) | `0.74409` | `0.74866` | `0.74637` | `0.0000` |
| **`mod_svm`** (Active-trimmed `Modulation_v2` RBF SVM) | `0.74382` | `0.73548` | `0.73965` | `0.0141` |
| **`boaw_svm`** (3-scale Mel BoAW RBF SVM) | `0.64785` | `0.65699` | `0.65242` | `0.0000` |
| **`cqt_svm`** (CQT Fisher linear SVM) | `0.62903` | `0.62930` | `0.62917` | `0.0000` |

- **Cross-Fitted Stacking, Post-Processing & Bagging**:
  - Cross-fitted per-model temperature calibration + L2-regularized NLL optimization (SLSQP) over simplex weights and arithmetic/geometric mixing (`lambda = 0.850`, ensemble temp `1.145`).
  - Acoustic k-NN graph label propagation (`alpha` grid `0.0–0.20`) and EM prior power (`0.0–1.0`) were both validated strictly on CV; CV automatically selected `alpha = 0.0` and `power = 0.0` (consistent with exact uniform class balancing).
  - Final test predictions blend **50% 10-fold CV-bagged probabilities + 50% full-data retrained probabilities**, predicting all **41/41** unique top-1 classes cleanly (min 10, max 106 per class).

### Final Validation Results
- **Plain OOF MAP@3**: Seed 42 = **`0.84570`** | Seed 913 = **`0.84032`** (**Mean = `0.84301`**)
- **Target-Weighted (`w_cal`) MAP@3**: Seed 42 = **`0.89134`** | Seed 913 = **`0.87939`**
- **Uniform-Weighted (`u_cal`) MAP@3**: Seed 42 = **`0.86004`** | Seed 913 = **`0.85436`**
- **Combined 3-Metric CV Score**: **`0.86186`**


### Submission 3 — 77-Feature Tuned 4-Model Ensemble (`SUBMISSIONS/Submission 3.ipynb`) — Pranav Sethi (2024425)

#### Objective
Improve upon the 89-feature verified-only baseline (`0.72088` CV / `0.69479` LB) through systematic MFCC dimensionality ablation, SVM hyperparameter optimization, classical model benchmarking, and controlled pseudo-labeling checks.

#### Pipeline & Experiments Tested
- **Features (77 dims):** 14 MFCCs (mean/std, 28), 12-bin Chroma STFT (mean/std, 24), spectral centroid/bandwidth/rolloff/ZCR/RMS (mean/std) + duration (11), and 7-band spectral contrast (mean/std, 14).
- **Ablations Run (3-seed 5-fold CV, seeds `42, 2026, 31415`):**
  - MFCC sweep (`13, 14, 20, 30, 40` coefficients) and feature additions (spectral flatness, 32 mel bands) showed `n_mfcc=14` (`77` dims) reduced variance over `n_mfcc=20` (`89` dims).
  - Model benchmarks on verified-only clips: tuned RBF-SVM (`C=5.0, gamma=0.01, balanced`) = `0.6963`, CatBoost = `0.6841`, Cosine k-NN (`k=11`) = `0.6188`, LightGBM = `0.5898`.
  - Single-model agreement and confidence filtering on unchecked clips (`threshold = 0.40–0.80`) did not beat verified-only (`0.73029` vs `0.73154`), so unchecked clips were excluded here.
- **Final Ensemble (620 verified recordings):** `40%` RBF-SVM (`C=5.0, gamma=0.01, balanced`) + `30%` Random Forest (`500` trees) + `20%` Extra Trees (`500` trees) + `10%` Logistic Regression (`C=1.0, balanced`).

#### Validation & Leaderboard Results
- **3-Seed OOF MAP@3:** Seed 42 = `0.73307` | Seed 2026 = `0.72016` | Seed 31415 = `0.74140` | **Mean = `0.73154` ± `0.00874`** (`+0.01066` over `0.72088` baseline).
- **Kaggle Public LB:** **`0.69585`** (`+0.00106` over `0.69479` baseline).
- **Takeaway:** Verified-only clip-level averages plateaued near `0.70` LB due to having only 10 verified samples for 34 of the 41 classes, motivating short-window local descriptors and semi-supervised unchecked utilization.

---

### Submission 8 — Full-Data Retraining + TPP + Dual Kernel (VLAD + PCA Fisher) (`SUBMISSIONS/submission 8.ipynb`) — Pranav Sethi (2024425)

#### Objective
Unlock the 4,984 unchecked recordings and combine clip-level rich features with multi-scale local temporal and spectral residual kernels while training final models on 100% of verified data.

#### Changes from Submission 3 & Pipeline
- **Expanded features (`319` dims):** Upgraded from 77 clip averages to 319 active-frame rich spectral descriptors (log-mel percentiles, MFCC deltas, flatness, onset rate, activity ratio).
- **Multi-scale short-window views (`w = 8, 32` frames, `12 ms` hop, fit on unchecked audio):**
  - **Temporal Pyramid Pooling (TPP, `768` dims):** Global + Start/Mid/End 3-segment histograms over 96-cluster `MiniBatchKMeans`.
  - **VLAD Residual Kernel:** 32-cluster intra-normalized, signed power-normalized linear Gram matrix.
  - **PCA-Whitened Fisher Kernel:** `PCA(24, whiten=True)` + 32-component diagonal `GaussianMixture` linear Gram matrix.
- **Full-620 Retraining & Soft Unchecked Weighting:** Stage-1 `Rich` ensemble trained on all 620 verified clips selected top-1 agreeing unchecked clips (`weight = 0.50`, floor `0.08` for non-agreeing clips in VLAD/Fisher SVMs), avoiding 4-fold subsampling starvation on 10-sample classes.
- **Blending & Calibration:** Compounding probability blend (`Rich + TPP + VLAD + PCA-Fisher`) followed by temperature sharpening (`T = 1.25`) and test-set EM prior adjustment (`pi^0.5`).

#### Leaderboard Results
- **Kaggle Public LB:** **`0.84353`** (**`+0.14768`** over Submission 3).
- **Takeaway:** Leveraging unchecked recordings alongside compact PCA-Fisher + VLAD kernels and full-620 retraining drove a major jump in generalization.

---

### Submission 9 — 4-Stream Meta-Fusion with Graph Label Propagation (`SUBMISSIONS/submission 9.ipynb`) — Pranav Sethi (2024425)

#### Objective
Test whether graph-based Label Propagation (`k=15`), a higher-capacity 15,360-dim GMM-16 Fisher kernel, and SLSQP OOF meta-weight optimization could improve upon Submission 8.

#### Changes from Submission 8 & Pipeline
- **4-Stream Architecture (3-seed 5-fold CV, fold-averaged predictions):**
  - Stream 1 (`48%`): 319-dim Rich 5-model ensemble (3-seed OOF `0.7769`).
  - Stream 2 (`13%`): 768-dim TPP patch RBF-SVM (3-seed OOF `0.6761`).
  - Stream 3 (`31%`): 15,360-dim 16-component un-PCA'd GMM Fisher Global linear SVM over all 4,984 unchecked clips.
  - Stream 4 (`8%`): `LabelPropagation(kernel='knn', n_neighbors=15)` filtered Fisher linear SVM (3-seed OOF `0.7833`).
- **Meta-Optimization & Calibration:** SLSQP weight optimization directly on OOF MAP@3 + `T = 2.0` + EM prior (`pi^0.5`).

#### Leaderboard Results & Error Analysis
- **Kaggle Public LB:** **`0.81047`** (`-0.03306` vs. Submission 8).
- **Root-Cause Findings (Negative Result):**
  1. Fitting `LabelPropagation` across all 620 verified labels before CV leaked validation labels into `keep_b`.
  2. Dropping VLAD and expanding Fisher vectors to 15,360 un-PCA'd dimensions increased variance on 10-sample classes.
  3. 5-fold CV averaging left rare classes with only 8 verified clips per model instead of 10.
  4. SLSQP directly on discrete MAP@3 over 620 rows overfit the validation set.

---

### Submission 10 — 3-Stream CV-Ensemble (`SUBMISSIONS/submission 10.ipynb`) — Pranav Sethi (2024425)

#### Objective
Isolate the impact of removing Label Propagation leakage and SLSQP weight overfitting while keeping the 3-stream fold-averaged 15,360-dim Fisher setup.

#### Changes from Submission 9 & Pipeline
- Removed the leaky Graph Label Propagation stream (`Stream 4`) and SLSQP weight tuning.
- Combined 3 fold-averaged streams across 3 seeds (`42, 2026, 913`) with clean fixed weights: `55%` 319-dim Rich + `10%` 768-dim TPP Patch SVM + `35%` 15,360-dim GMM-16 Fisher linear SVM, followed by `T = 2.0` and EM prior adjustment.

#### Leaderboard Results & Takeaways
- **Kaggle Public LB:** **`0.81540`** (`+0.00493` over Submission 9, `-0.02813` vs. Submission 8).
- **Takeaway:** Removing Label Propagation leakage immediately improved LB generalization. Comparing Submissions 8 (`0.84353`), 9 (`0.81047`), and 10 (`0.81540`) confirmed that compact PCA-whitened Fisher + VLAD kernels and full-620 retraining are essential for rare classes.

---

### Submission 13 — Step 1: 7-Stream HPSS + 8-View MKL + Debiased-Likelihood EM (`Step1_HPSS_MKL_DebiasedEM.ipynb`) — Pranav Sethi (2024425)

#### Objective
Address the core acoustic and statistical bottlenecks identified across Submissions 3–10 and the team's `0.85161` benchmark: harmonic/percussive spectral superposition, lack of cross-view kernel interaction, distorted implicit class priors breaking naive EM, and missing verified-manifold anchoring.

#### Changes from Submissions 8–10 (and 0.85161 Benchmark) & Pipeline
- **Five-View Single-Pass Audio Extraction:**
  - `Rich-319d` + **`HPSS-178d`** (active `>-20 dB` + top-20% energy harmonic/percussive MFCCs, centroids, flatness, energy ratios; `+1.05%` plain / `+1.30%` unif standalone gain over `Frich + Mod_v1`).
  - `Mod_v1-468d` (10 ms hop 24-band modulation & cross-band correlation).
  - Standard 40-mel patches (`w = 8, 32`) -> BoAW (`192d`), VLAD (`7,680d`), PCA-24 GMM-32 Fisher (`3,072d`).
  - **HPSS-Dual 24+24-mel patches (`w = 8, 32`)** -> BoAW (`192d`), VLAD (`9,216d`), and **2-seed bagged PCA-32 GMM-40 Fisher (`5,120d`)** (`+4.72%` plain / `+6.17%` unif standalone over standard Fisher).
- **Prior-Debiased 6-View Stage-1 Teacher:** Verified-only teacher (`0.8115` OOF plain MAP@3 vs. `0.7546` for `Frich`-only) debiased by fold-internal uniform-weighted OOF marginals before top-1 agreement (`~48%` kept at `0.50`, floor `0.08`).
- **7-Stream Classical Ensemble (Full-620 + 4,984 Unchecked Retraining):**
  - `mkl` (`28%`): 8-View Joint Multi-Kernel SVM (`HPSS-FV`, `Std-FV`, `HPSS-VLAD`, `Std-VLAD`, `BoAW`, `Rich`, `HPSS-178`, `Mod_v1`).
  - `jlr` (`18%`): L2 Multinomial Logistic Regression on concatenated unit-norm `[FV_H1, FV_H2, FV_S, VLAD_H, VLAD_S]` (globally convex, zero Platt noise).
  - `hpss_k` (`15%`) & `std_k` (`10%`): Dedicated HPSS and Standard Log-Mel kernel SVMs.
  - `tab` (`12%`): 497-dim `[Rich + HPSS]` 5-model balanced ensemble (`SVM3 + SVM10 + LR + RF + ET`).
  - `mod` (`10%`): 50% keep-only + 50% all-unchecked modulation SVM.
  - `knn` (`7%`): Class-balanced Verified `16-NN` in 8-View MKL kernel space (anchors predictions to the verified recording distribution, which matches test at adversarial AUC `0.5077`).
- **Exact Debiased-Likelihood EM Calibration:** Divides out the empirical uniform-weighted 3-seed OOF class marginal `pi_src` before temperature scaling (`T = 1.90`) and damped Saerens EM (`power = 0.50`), fixing naive EM's `-1.1%` to `-1.5%` distortion.

#### Validation Results (3-Seed 5-Fold Stratified CV, Seeds `42, 913, 2026`)
| Metric | Seed 42 | Seed 913 | Seed 2026 | **3-Seed Mean** |
| :--- | :---: | :---: | :---: | :---: |
| Plain OOF MAP@3 | `0.8766` | `0.8712` | `0.8688` | **`0.87222`** |
| Uniform-Weighted OOF MAP@3 | `0.8927` | `0.8870` | `0.8846` | **`0.88808`** |
| Train-Total Weighted OOF MAP@3 | — | — | — | **`0.90152`** |
| `v10` / `v40` OOF MAP@3 | — | — | — | **`0.89771` / `0.84127`** |

- **Simulated Test-Prior Shift (`DebiasEM 0.50`):** Uniform = `0.88464`, Train-Total = `0.89798`, Skewed (`C05=10%`) = `0.88340` (`+4.95%` to `+6.26%` over `Highest_Score.ipynb`'s naive EM).

---

### Submission 14 — Step 2B: 9-Stream HPSS-Modulation + Shrunk LDA & Decoupled Train/Inference Pipeline (`SUBMISSIONS/notebook-score0-883.ipynb`, `Step2B_Train_SaveArtifacts.ipynb` + `Step2B_Inference_Only.ipynb`) — Pranav Sethi (2024425)

#### Objective
Test Step 2 candidate expansions (`Mod_HPSS-512d`, CatBoost, LightGBM, and Shrunk LDA) against Submission 13 on the exact same 3-seed CV splits, promote only components that beat Submission 13 across CV and simulated test-prior shift, and split the winning pipeline into offline artifact training + fast inference-only notebooks for private-test evaluation.

#### Changes from Submission 13 & Pipeline
- **Added 2 Validated Streams (Zero Extra HPSS Extraction Cost):**
  - **`mod_hpss` (`9%`, `Mod_HPSS-512d`):** Harmonic + percussive modulation spectrum and cross-band correlation SVM, computed directly from the `S1_h, S1_p` spectrograms already produced for HPSS patches (`+3.95%` plain / `+4.02%` unif standalone over `Mod_v1`).
  - **`lda` (`4%`):** Shrunk LDA (`solver='lsqr', shrinkage='auto'`) on 497-dim `[Rich + HPSS]` (~1 s/fold generative Gaussian diversity).
- **Controlled Pruning (Step 2 Gate):** Tested and explicitly rejected CatBoost (`-0.0019` unif) and LightGBM (`-0.0021` unif), which added +35–40 min CPU time while regressing `unif`, `w_tot`, and simulated test-prior shift.
- **Final 9-Stream Weights:** `mkl 25%`, `jlr 16%`, `hpss_k 14%`, `std_k 9%`, `tab 11%`, `mod 5%`, `mod_hpss 9%`, `lda 4%`, `knn 7%`.
- **Decoupled Train/Inference Split (Private-Test Ready):**
  - `Step2B_Train_SaveArtifacts.ipynb` trains exclusively on the 5,604 training recordings (training-only medians & BoAW gammas) and exports `797.6 MB` to `/kaggle/working/step2b_artifacts/` (`meta_and_encoders.joblib`: `0.3 MB`, `models.joblib`: `107.6 MB`, `train_refs.npz`: `689.7 MB`).
  - `Step2B_Inference_Only.ipynb` loads those artifacts, extracts features only for test audio with identical extraction code, drops any hardcoded `1592` test-length check, and generates `/kaggle/working/submission.csv` in `~10 min` on Kaggle 4-core CPU (down from `~84 min` full end-to-end training), with `99.81%` top-1 parity (`1,589 / 1,592`) and `41/41` classes represented.

#### Validation & Leaderboard Results (3-Seed 5-Fold Stratified CV on Kaggle)
| Metric | Seed 42 | Seed 913 | Seed 2026 | **3-Seed Mean** |
| :--- | :---: | :---: | :---: | :---: |
| Plain OOF MAP@3 | `0.87876` | `0.87177` | `0.87473` | **`0.87509`** |
| Uniform-Weighted OOF MAP@3 | `0.89502` | `0.88994` | `0.89014` | **`0.89170`** |
| Train-Total Weighted OOF MAP@3 | `0.90689` | `0.90377` | `0.90106` | **`0.90390`** |
| `v10` (34 classes, 10 verified) | `0.90490` | `0.90098` | `0.89951` | **`0.90180`** |
| `v40` (7 classes, 40 verified) | `0.84702` | `0.83631` | `0.84464` | **`0.84266`** |
| **Kaggle Public Leaderboard** | — | — | — | **`0.883`** |

- **Gain:** Beat Submission 13 across every CV metric and every simulated test-prior shift regime (`Uniform 0.88692`, `Train-Total 0.89898`, `Skewed 0.88760`), reaching **`0.883`** on Kaggle Public LB (`+0.0395` over Submission 8, `+0.0676` over Submission 10, and `+0.0314` over the team's previous `0.85161` benchmark).
