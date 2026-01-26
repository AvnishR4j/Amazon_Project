# Claude Code Context File

> **Instructions**: When returning to this project, tell Claude: "Read CLAUDE_CONTEXT.md and help me continue"

---

## Project Summary

**Name**: Amazon Supply Chain Intelligence
**Purpose**: Predict delivery delay risk (On-Time, At-Risk, Delayed)
**Course**: UCS321 - AI for Engineers (Mini Project, 60% of grade)
**Student**: ARUN AK
**GitHub**: https://github.com/Arun07AK/Amazon_Project

---

## Current Status: COMPLETE (Ready for Submission)

| Component | File | Status |
|-----------|------|--------|
| Data Preprocessing | `src/data_preprocessing.py` | Done |
| Model Training | `src/train_fast.py` | Done |
| Evaluation | `src/evaluation.py` | Done |
| SHAP Explainability | `src/explainability.py` | Done |
| Streamlit Dashboard | `dashboard/app.py` | Done |
| Presentation | `presentation/presentation.html` | Done |
| Documentation | `docs/UNDERSTANDING_THE_PROJECT.md` | Done |

---

## Key Technical Details

- **Dataset**: Kaggle Amazon Delivery Dataset (43,739 records)
- **Best Model**: XGBoost with 79.3% accuracy
- **Classes**: On-Time (≤90min), At-Risk (91-160min), Delayed (>160min)
- **Features**: 16 original → 19 engineered
- **Class Balancing**: SMOTE
- **Models Compared**: 8 (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN, Decision Tree, Neural Network)

---

## File Structure

```
Amazon_Project/
├── config/config.yaml           # Configuration
├── data/
│   ├── raw/amazon_delivery.csv  # Kaggle dataset
│   └── processed/               # Preprocessed train/test splits
├── dashboard/app.py             # Streamlit dashboard
├── models/                      # Trained models (.joblib)
├── notebooks/                   # Jupyter EDA notebook
├── presentation/                # HTML slides for viva
├── reports/figures/             # All visualizations
├── src/                         # Python modules
└── docs/                        # Documentation
```

---

## Quick Commands

```bash
# Run full pipeline
python src/data_preprocessing.py
python src/train_fast.py
python src/evaluation.py
python src/explainability.py

# Launch dashboard
streamlit run dashboard/app.py

# View presentation
open presentation/presentation.html

# Push to GitHub
git add -A && git commit -m "message" && git push
```

---

## Potential Improvements (If User Wants More)

1. **Improve accuracy to 85%+**: Hyperparameter tuning, feature selection, ensemble stacking
2. **Add more visualizations**: Geographic maps, time series analysis
3. **API endpoint**: Flask/FastAPI for model serving
4. **Docker**: Containerize the application
5. **CI/CD**: GitHub Actions for automated testing
6. **Deep Learning**: Try LSTM or neural network architectures
7. **Real-time data**: Streaming predictions with Kafka

---

## Known Issues

- Jupyter notebook (`notebooks/01_EDA_and_Preprocessing.ipynb`) has a NaN time parsing error - works as template but needs manual cell execution
- Team member names in README.md and slides.md are placeholders

---

## User Background

- Knows: C++, DSA, OOPs, HTML/CSS/JS
- Learning: Python, Machine Learning (first ML project)
- Created `docs/UNDERSTANDING_THE_PROJECT.md` with beginner-friendly explanations

---

## Last Updated
January 2026 - Project complete, expected marks 36-38/40 (90-95%)
