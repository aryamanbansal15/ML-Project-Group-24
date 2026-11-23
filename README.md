# Listening Past the Noise
### Audio Classification Under Noisy Labels

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
