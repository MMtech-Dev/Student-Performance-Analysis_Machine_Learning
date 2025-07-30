# Student Performance Analysis & Machine Learning Portfolio

## 🎯 Project Overview

This repository showcases two comprehensive data science projects demonstrating end-to-end analytical capabilities and machine learning expertise. These projects highlight practical skills in data exploration, visualization, feature engineering, and predictive modeling using real-world educational datasets.

## 📊 Projects

### 1. Student Analytics Project
**Comprehensive educational data analysis with professional visualizations**

- **Data Exploration**: Multi-dimensional analysis of student behavior, assessment patterns, and performance metrics
- **Key Insights**: Assessment distribution analysis, demographic performance patterns, submission timing trends
- **Visualizations**: Professional matplotlib charts showing engagement patterns, score distributions, and learning behaviors
- **Business Impact**: Identified key factors affecting student success and withdrawal patterns

### 2. Student Performance Classification
**Machine learning pipeline for predicting academic outcomes**

- **Problem**: Multi-class classification predicting student final results (Pass/Fail/Distinction/Withdrawn)
- **Approach**: Implemented and compared three algorithms using CRISP-DM methodology
- **Best Model**: XGBoost achieving **70.9% accuracy** with excellent Pass prediction (F1-score: 0.80)
- **Key Features**: Engagement metrics, submission patterns, historical performance, and behavioral indicators

## 🏆 Technical Achievements

**Model Performance Comparison:**
| Algorithm | Accuracy | Strengths |
|-----------|----------|-----------|
| **XGBoost** | **70.9%** | Best overall performance, superior feature importance |
| Random Forest | 68.5% | Strong generalization, robust feature selection |
| Logistic Regression | 63.6% | Interpretable baseline, fast training |

**Critical Success Factors Identified:**
- `last_access` - Final course interaction timing
- `avg_submission_day` - Assessment submission patterns  
- `avg_score` - Historical performance trends
- `engagement_duration` - Total learning time investment

## 🛠️ Technical Stack

**Core Technologies:**
- **Python** | **Pandas** | **NumPy** | **Matplotlib** | **Seaborn**
- **Scikit-learn** | **XGBoost** | **Jupyter Notebooks**

**Methodology:**
- **CRISP-DM Framework** for structured data science approach
- **Feature Engineering** from raw educational data
- **Model Evaluation** using classification metrics and confusion matrices
- **Data Visualization** for insights communication

## 📁 Repository Structure

```
├── README.md
├── Student_Analytics_Project/
│   ├── README.md
│   ├── Visualisation/
│   │   ├── README.md
│   │   └── [Professional charts & analysis]
│   └── Reports/
├── Student_Performance_Classification/
│   ├── README.md
│   ├── Models/
│   │   ├── README.md
│   │   ├── XGBoost.py
│   │   ├── Random_Forest.py
│   │   └── Logistic_regression_model.py
│   ├── Data_processing/
│   └── requirements.txt
```
## 📋 Data Requirements

**Note:** Datasets are not included in this repository due to data privacy and compliance requirements.

**To run the analysis:**
- Student performance datasets with features including engagement metrics, assessment scores, demographic information, and learning behaviors
- Data should include target variables for classification (Pass/Fail/Distinction/Withdrawn)

**Data Structure Expected:**
- Student interaction logs (clicks, access times, engagement duration)
- Assessment data (scores, submission timing, assessment types)
- Demographic information (age bands, socioeconomic indicators)
- Final outcome classifications

The code is designed to work with properly formatted CSV files following the structure shown in the visualizations.

## 🎓 Professional Context

These projects demonstrate practical application of data science skills developed through:
- **PDA in Data Science** (SQA Level 8) - Advanced analytics and machine learning
- **AWS Certified Cloud Practitioner** - Cloud computing fundamentals
- **IBM Python for Data Science, AI & Development** - Professional certification

## 🚀 Career Focus

Actively seeking opportunities in:
- **AI/ML Developer** roles
- **Data Science** positions  
- **Python Developer** roles with data focus
- **Data Analyst** positions