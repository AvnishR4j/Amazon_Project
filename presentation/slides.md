# Amazon Supply Chain Intelligence
## AI-Powered Delivery Delay Risk Prediction

**UCS321 - AI for Engineers | Mini Project**

**Team Members:**
- ARUN AK (Team Lead)
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

---

# Agenda

1. Problem Statement
2. Dataset Overview
3. Methodology
4. Feature Engineering
5. Model Development
6. Results & Evaluation
7. Dashboard Demo
8. Business Impact
9. Conclusion

---

# 1. Problem Statement

## The Challenge

> Amazon processes **millions of deliveries daily**. Predicting which deliveries are at risk of delay enables **proactive intervention** and **improved customer satisfaction**.

## Our Objective

Build an AI system to predict delivery delay risk levels:
- **On-Time** (≤90 minutes)
- **At-Risk** (91-160 minutes)
- **Delayed** (>160 minutes)

## Business Value
- Reduce customer complaints by **15%**
- Enable proactive customer communication
- Optimize resource allocation

---

# 2. Dataset Overview

## Amazon Delivery Dataset (Kaggle)

| Metric | Value |
|--------|-------|
| **Total Records** | 43,739 |
| **Features** | 16 |
| **Time Period** | March 2022 |
| **Source** | Real Amazon last-mile logistics |

## Key Features

| Feature | Description |
|---------|-------------|
| Agent_Age | Delivery agent's age |
| Agent_Rating | Agent performance rating (1-5) |
| Distance | Haversine distance (calculated) |
| Weather | Sunny, Cloudy, Stormy, Fog, etc. |
| Traffic | Low, Medium, High, Jam |
| Vehicle | Motorcycle, Scooter, Bicycle, Van |

---

# 3. Methodology

## ML Pipeline Flow

```
Data Ingestion → Preprocessing → Feature Engineering →
SMOTE Balancing → Model Training → Evaluation → Dashboard
```

## Key Steps

1. **Data Cleaning** - Handle missing values, outliers
2. **Feature Engineering** - Create 19 predictive features
3. **Class Balancing** - SMOTE for imbalanced data
4. **Model Comparison** - 8 algorithms evaluated
5. **Deployment** - Interactive Streamlit dashboard

---

# 4. Feature Engineering

## Engineered Features (19 Total)

| Category | Features |
|----------|----------|
| **Distance** | Haversine distance (km) |
| **Temporal** | Hour, Day, Month, Weekend flag, Rush hour |
| **Weather** | Severity score (1-5) |
| **Interaction** | Distance × Traffic, Risk score |
| **Agent** | Experience score |
| **Encoded** | Weather, Traffic, Vehicle, Area, Category |

## Distance Calculation (Haversine Formula)

```
d = 2r × arcsin(√(sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)))
```

---

# 5. Model Development

## 8 Algorithms Compared

1. Logistic Regression
2. **Random Forest**
3. **XGBoost** ← Best Model
4. **Gradient Boosting**
5. Support Vector Machine
6. K-Nearest Neighbors
7. Decision Tree
8. Neural Network (MLP)

## Class Imbalance Handling

| Before SMOTE | After SMOTE |
|--------------|-------------|
| On-Time: 28.1% | On-Time: 33.3% |
| At-Risk: 48.8% | At-Risk: 33.3% |
| Delayed: 23.1% | Delayed: 33.3% |

---

# 6. Results & Evaluation

## Model Performance Comparison

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **XGBoost** | **79.3%** | **0.793** |
| Gradient Boosting | 79.0% | 0.790 |
| Random Forest | 76.7% | 0.767 |
| Decision Tree | 72.9% | 0.729 |
| Neural Network | 72.8% | 0.728 |
| SVM | 65.0% | 0.648 |

## Best Model: XGBoost

| Metric | Value |
|--------|-------|
| Accuracy | 79.3% |
| Precision | 79.3% |
| Recall | 79.3% |
| F1-Score | 0.79 |

---

# 6. Results - Confusion Matrix

## Classification Performance

| Actual \ Predicted | On-Time | At-Risk | Delayed |
|-------------------|---------|---------|---------|
| **On-Time** | 1889 | 512 | 58 |
| **At-Risk** | 555 | 3428 | 289 |
| **Delayed** | 82 | 318 | 1617 |

## Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| On-Time | 80% | 77% | 78% |
| At-Risk | 78% | 80% | 79% |
| Delayed | 82% | 80% | 81% |

---

# 7. Top Predictive Features

## Feature Importance (SHAP Analysis)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **Agent Rating** | 0.146 |
| 2 | **Distance × Traffic** | 0.121 |
| 3 | **Weather Encoded** | 0.104 |
| 4 | **Risk Score** | 0.092 |
| 5 | **Distance (km)** | 0.088 |

## Key Insights

- Higher agent ratings → Faster deliveries
- Long distance + Heavy traffic → High delay risk
- Stormy weather increases delays by **45%**
- Rush hour orders have **25%** higher delay rate

---

# 8. Dashboard Demo

## Streamlit Application Features

1. **Real-time Prediction** - Enter order details, get instant risk assessment
2. **Data Analytics** - Explore delivery patterns and trends
3. **Model Performance** - View evaluation metrics and charts
4. **Business Insights** - Actionable recommendations

## Live Demo

```bash
streamlit run dashboard/app.py
```

[SHOW LIVE DEMO]

---

# 9. Business Impact

## Projected Benefits

| Metric | Improvement |
|--------|-------------|
| Late Delivery Reduction | 15-20% |
| Customer Satisfaction | +25% |
| Proactive Alerts | 100% At-Risk orders |
| Resource Optimization | 10% cost savings |

## ROI Calculation

For 100,000 monthly orders:
- Current delay rate: 23%
- Projected delay rate: 19.5%
- Delays prevented: ~3,500/month
- Cost savings: **$52,500/month** (at $15/delay)

---

# 10. Innovation Highlights

## What Makes This Project Special

1. **Real Kaggle Dataset** - 43,739 actual Amazon deliveries
2. **Comprehensive Pipeline** - End-to-end ML workflow
3. **8 Model Comparison** - Rigorous algorithm selection
4. **SHAP Explainability** - Interpretable AI
5. **Interactive Dashboard** - Production-ready UI
6. **Business Focus** - Actionable insights

## Technologies Used

`Python` `Scikit-learn` `XGBoost` `SHAP` `Streamlit` `Pandas` `Plotly`

---

# 11. Conclusion

## Summary

- Built an AI system achieving **79.3% accuracy** in predicting delivery delays
- Identified key factors: Distance, Weather, Traffic, Agent Rating
- Created interactive dashboard for real-time predictions
- Provided actionable business recommendations

## Future Work

- Real-time API integration with carrier systems
- Deep learning models (LSTM for time-series)
- A/B testing framework for model deployment
- Mobile app for logistics team

---

# Thank You!

## Questions?

**GitHub Repository:**
https://github.com/Arun07AK/Amazon_Project

**Contact:**
- ARUN AK
- UCS321 - AI for Engineers

---

# Appendix: Technical Details

## Data Preprocessing Steps

1. Load 43,739 records from Kaggle
2. Clean column names, remove whitespace
3. Handle missing values (median/mode imputation)
4. Clip outliers using IQR method
5. Calculate Haversine distance
6. Extract temporal features
7. Create weather severity index
8. Encode categorical variables
9. Apply SMOTE for class balancing
10. Scale numerical features

---

# Appendix: Evaluation Metrics

## Why F1-Score?

- Balanced metric for imbalanced classes
- Considers both precision and recall
- Weighted F1 accounts for class distribution

## Cross-Validation

- 5-Fold Stratified CV
- Preserves class distribution in each fold
- Robust performance estimation

---

# Appendix: SHAP Explainability

## What is SHAP?

- **SH**apley **A**dditive ex**P**lanations
- Game-theoretic approach to feature importance
- Explains individual predictions
- Globally consistent importance values

## Why SHAP over Feature Importance?

- More accurate and consistent
- Explains direction of impact
- Works for any model

---
