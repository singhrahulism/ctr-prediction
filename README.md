# CTR Prediction System (Ad-Tech)

## Overview

This project implements an **end-to-end Click-Through Rate (CTR) prediction pipeline** inspired by real-world ad-tech systems.  
The goal is to predict the probability that an ad impression will be clicked, using high-cardinality categorical features at scale.

The system is designed with :
- Artifact persistence
- Consistent preprocessing
- Batch inference on unseen data

---

## Dataset

- **Source**: Avazu CTR Prediction Dataset
- **Scale**:
  - ~40 million training impressions
  - Highly imbalanced binary target (`click`)
- **Feature characteristics**:
  - Mostly high-cardinality categorical features
  - Examples: `site_id`, `app_id`, `device_id`, `device_ip`
  - Temporal feature encoded as `hour` (YYMMDDHH)

### Key challenges
- One-hot encoding is infeasible due to cardinality
- Sparse data with weak individual signals
- Strong class imbalance

---

## Feature Engineering

### Decisions
- Dropped `id` (pure identifier, no predictive signal)
- Extracted `hour_of_day` from `hour`
- Treated almost all remaining fields as categorical

### Strategy
- **Feature Hashing** for categorical variables
  - Fixed-size sparse representation
  - Constant memory usage
  - Acceptable collision trade-off
- Numeric features appended directly

---

## Model

- **Algorithm**: Logistic Regression
- **Solver**: `saga`
- **Regularization**: L2 (default)
- **Imbalance handling**: `class_weight="balanced"`

### Why Logistic Regression?
- Scales well to large sparse feature spaces
- Produces calibrated probabilities
- Stable, interpretable baseline for CTR prediction

---

## Training Pipeline

Training is performed **offline**.

Steps:
1. Load sampled training data
2. Feature engineering
3. Feature hashing + preprocessing
4. Train logistic regression model
5. Evaluate using ROC-AUC
6. Save model and preprocessing metadata

### Validation Result

On a representative sampled dataset:

Validation ROC-AUC ≈ 0.725

This confirms the pipeline learns meaningful signal and generalizes better with more data.

---

## Inference Pipeline

Inference is **decoupled from training**.

- Loads saved model artifacts
- Applies identical feature engineering and hashing
- Outputs:
  - `click_probability`
  - Optional `predicted_click` using configurable threshold

This design supports both batch and low-latency serving scenarios.

---
