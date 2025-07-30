# Machine Learning Models

## Overview
Classification models for predicting student performance outcomes using educational data features. Three algorithms were implemented and compared for optimal performance.

## Model Performance Comparison

| Algorithm | Accuracy | Best Class Performance |
|-----------|----------|----------------------|
| **XGBoost** | **70.9%** | Pass: 0.80 F1-score |
| Random Forest | 68.5% | Pass: 0.79 F1-score |
| Logistic Regression | 63.6% | Pass: 0.75 F1-score |

## Key Findings

**Top Predictive Features (XGBoost):**
- `last_access` - Student's final course interaction
- `avg_submission_day` - Average assessment submission timing
- `avg_score` - Historical performance average
- `banked_count` - Assessment attempts stored
- `engagement_duration` - Total time spent in course

**Model Strengths:**
- **XGBoost**: Best overall accuracy, excellent pass prediction (80% F1)
- **Random Forest**: Strong feature importance analysis, good generalization
- **Logistic Regression**: Interpretable coefficients, fast training

## Technical Implementation
- **Target Classes**: Distinction, Fail, Pass, Withdrawn
- **Features**: 25 engineered features from student behavior and performance data
- **Evaluation**: Classification reports, confusion matrices, feature importance analysis
- **Framework**: Python scikit-learn, XGBoost

## Files
- `Logistic_regression_model.py` - Linear classification baseline
- `Random_Forest.py` - Ensemble method with feature importance
- `XGBoost.py` - Gradient boosting (best performing model)
- `requirements.txt` - Dependencies for model reproduction