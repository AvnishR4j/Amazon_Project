# Amazon Supply Chain Intelligence - System Flow Diagram

## Complete ML Pipeline Architecture

```mermaid
flowchart TB
    subgraph INPUT["📥 DATA INGESTION"]
        A[("🗄️ Kaggle Dataset<br/>43,739 Records")] --> B["📋 Raw Data<br/>16 Features"]
    end

    subgraph PREPROCESS["🔧 DATA PREPROCESSING"]
        B --> C["🧹 Data Cleaning<br/>• Remove duplicates<br/>• Handle NaN strings<br/>• Standardize columns"]
        C --> D["🔄 Missing Value Handling<br/>• Median for numerical<br/>• Mode for categorical"]
        D --> E["📊 Outlier Treatment<br/>• IQR Method<br/>• Clip outliers"]
    end

    subgraph FEATURE["⚙️ FEATURE ENGINEERING"]
        E --> F["📍 Distance Calculation<br/>Haversine Formula"]
        E --> G["🕐 Temporal Features<br/>• Hour, Day, Month<br/>• Rush hour flag<br/>• Weekend flag"]
        E --> H["🌤️ Weather Encoding<br/>Severity Score"]
        E --> I["🚗 Traffic Interaction<br/>Distance × Traffic"]
        F & G & H & I --> J["🎯 19 Engineered Features"]
    end

    subgraph TARGET["🏷️ TARGET CREATION"]
        J --> K{"⏱️ Delivery Time"}
        K -->|"≤ 90 min"| L["✅ On-Time<br/>Class 0"]
        K -->|"91-160 min"| M["⚠️ At-Risk<br/>Class 1"]
        K -->|"> 160 min"| N["🚨 Delayed<br/>Class 2"]
    end

    subgraph BALANCE["⚖️ CLASS BALANCING"]
        L & M & N --> O["📊 Imbalanced Data<br/>28% / 49% / 23%"]
        O --> P["🔄 SMOTE<br/>Synthetic Oversampling"]
        P --> Q["✅ Balanced Data<br/>33% / 33% / 33%"]
    end

    subgraph SPLIT["✂️ DATA SPLITTING"]
        Q --> R["📚 Training Set<br/>80% (34,991)"]
        Q --> S["🧪 Test Set<br/>20% (8,748)"]
    end

    subgraph TRAIN["🤖 MODEL TRAINING"]
        R --> T["📈 8 ML Algorithms"]
        T --> T1["Logistic Regression"]
        T --> T2["Random Forest"]
        T --> T3["🏆 XGBoost"]
        T --> T4["Gradient Boosting"]
        T --> T5["SVM"]
        T --> T6["KNN"]
        T --> T7["Decision Tree"]
        T --> T8["Neural Network"]
    end

    subgraph EVAL["📊 EVALUATION"]
        T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 --> U["🔄 5-Fold Cross Validation"]
        U --> V["📈 Performance Metrics<br/>• Accuracy: 79.3%<br/>• F1-Score: 0.79<br/>• ROC-AUC"]
        S --> V
    end

    subgraph OUTPUT["📤 OUTPUT"]
        V --> W["💾 Best Model<br/>XGBoost"]
        W --> X["🖥️ Streamlit Dashboard"]
        X --> Y["🔮 Real-time Predictions"]
        X --> Z["📊 Analytics & Insights"]
    end

    style A fill:#ff9900,stroke:#232f3e,color:#000
    style T3 fill:#28a745,stroke:#155724,color:#fff
    style W fill:#28a745,stroke:#155724,color:#fff
    style L fill:#28a745,stroke:#155724,color:#fff
    style M fill:#ffc107,stroke:#856404,color:#000
    style N fill:#dc3545,stroke:#721c24,color:#fff
    style X fill:#ff4b4b,stroke:#232f3e,color:#fff
```

## Simplified Linear Flow

```mermaid
flowchart LR
    A["📥 Raw Data"] --> B["🔧 Preprocessing"]
    B --> C["⚙️ Feature Engineering"]
    C --> D["⚖️ SMOTE Balancing"]
    D --> E["🤖 Model Training"]
    E --> F["📊 Evaluation"]
    F --> G["🖥️ Dashboard"]

    style A fill:#ff9900
    style G fill:#28a745
```

## Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1. Ingestion | Kaggle CSV | Load & Validate | 43,739 × 16 DataFrame |
| 2. Cleaning | Raw DataFrame | Clean & Standardize | Clean DataFrame |
| 3. Missing Values | Clean Data | Impute median/mode | Complete Data |
| 4. Outliers | Complete Data | IQR Clipping | Normalized Data |
| 5. Feature Engineering | Normalized Data | Create 19 features | Feature Matrix |
| 6. Target Creation | Delivery Time | Threshold binning | 3-class labels |
| 7. Balancing | Imbalanced Data | SMOTE | Balanced Data |
| 8. Split | Balanced Data | 80/20 split | Train/Test sets |
| 9. Training | Training Data | 8 algorithms | Trained Models |
| 10. Evaluation | Test Data | Metrics calculation | Best Model |
| 11. Deployment | Best Model | Streamlit | Live Dashboard |
