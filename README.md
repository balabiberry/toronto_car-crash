# Fatal Collision Risk Prediction

Machine learning project for predicting whether a traffic collision is fatal using Toronto KSI collision data.  
The project focuses on building a leakage-safe, accident-level classification pipeline for a noisy and highly imbalanced real-world tabular dataset.

## Project Overview

Fatal collision prediction is a difficult classification problem because fatal accidents are rare, many features are categorical, and the available data can contain noisy or incomplete signals. Instead of optimizing only for overall accuracy, this project prioritizes identifying fatal collisions through recall, precision-recall analysis, F1-score tuning, and confusion matrix evaluation.

The final pipeline compares traditional machine learning models and a neural network under the same preprocessing, feature engineering, and evaluation setup.

## Problem

The goal is to classify each accident as:

- `Fatal`
- `Non-Fatal`

The dataset is transformed from participant-level records into accident-level records so that each collision is represented once. This prevents duplicated participants from leaking information across train and test splits.

## Key Challenges

- Severe class imbalance: fatal collisions are much rarer than non-fatal collisions.
- Noisy categorical data with many low-frequency labels.
- Multiple participant rows can belong to the same accident.
- Risk of data leakage if rows from the same accident appear in both train and test sets.
- Limited predictive signal from available tabular features.
- Trade-off between precision and recall for the fatal class.

## Approach

### 1. Leakage-Safe Data Splitting

The data is split by `ACCNUM`, not by individual rows.  
This ensures that all participants from the same accident stay either in the training set or the test set.

### 2. Feature Engineering

The pipeline creates accident-level features from raw collision records, including:

- Season extracted from date.
- Weekend indicator.
- Time-of-day bins such as night, morning rush, day, afternoon rush, and evening.
- Cyclical hour features using sine and cosine transformations.
- Age group mapping.
- Grouped pedestrian, cyclist, driver condition, and action categories.
- Participant count per accident.
- Interaction features such as:
  - senior pedestrian vulnerability,
  - rain and speed-related fatality risk,
  - truck speeding interaction,
  - blind-spot style crash indicators.

### 3. Preprocessing

The preprocessing stage includes:

- Missing value handling.
- Categorical cleanup.
- One-hot encoding.
- Removal of noisy unknown-category dummy columns.
- Train-test column alignment.
- Standard scaling for model compatibility.

### 4. Accident-Level Aggregation

Participant-level features are aggregated into one row per accident.

Depending on the feature type, values are aggregated using:

- `sum` for participant/action counts,
- `max` for accident-level indicators,
- participant count as an additional feature.

## Models Tested

The following models were benchmarked:

- Neural Network with TensorFlow/Keras
- Logistic Regression
- Random Forest
- Gradient Boosting

The models were evaluated with class weighting and threshold tuning to better handle the minority fatal class.

## Evaluation Strategy

Because the dataset is imbalanced, accuracy alone is not a meaningful metric.  
The project evaluates models using:

- Precision
- Recall
- F1-score
- Confusion matrix
- Precision-recall curve threshold tuning

The threshold for classification is selected using the best F1-score from the precision-recall curve.

## Results

| Model | Fatal Precision | Fatal Recall | Fatal F1 |
|---|---:|---:|---:|
| Neural Network | 0.22 | 0.57 | 0.32 |
| Logistic Regression | 0.28 | 0.48 | 0.35 |
| Random Forest | 0.32 | 0.38 | 0.35 |
| Gradient Boosting | 0.28 | 0.48 | 0.36 |

The neural network achieved the highest fatal-class recall at **57%**, while Gradient Boosting achieved the strongest fatal-class F1-score at **0.36**.

## Interpretation

The results show that fatal collision prediction is challenging with the available data.  
The project demonstrates that careful preprocessing, leakage-safe splitting, feature engineering, class weighting, and threshold tuning can improve minority-class detection, but the dataset still contains limited predictive signal.

This makes the project valuable as a real-world machine learning case study: the focus is not only on model performance, but also on responsible evaluation, data limitations, and the trade-offs involved in imbalanced classification.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib

## Project Structure

```text
.
├── notebook.ipynb
├── data/
│   └── TOTAL_KSI_3737821728629277523.csv
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository.

```bash
git clone <repo-url>
cd <repo-name>
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add the dataset to the project directory or update the CSV path in the notebook.

4. Run the notebook.

```bash
jupyter notebook
```

## Future Improvements

Potential next steps:

- Add geospatial features using latitude, longitude, intersections, and neighbourhood-level statistics.
- Test calibrated models and probability calibration.
- Use SHAP or permutation importance for interpretability.
- Compare additional imbalance-handling techniques such as SMOTE, undersampling, and focal loss.
- Add cross-validation at the accident level.
- Build a lightweight prediction dashboard.

## Resume Summary

Built a leakage-safe accident-level ML pipeline for fatal collision prediction on noisy, imbalanced Toronto KSI data; engineered temporal, demographic, categorical, and interaction features, benchmarked neural networks and tree-based models, and optimized PR/F1 thresholds to improve fatal-class recall.
