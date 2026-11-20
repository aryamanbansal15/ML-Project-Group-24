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
