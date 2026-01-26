"""
Amazon Supply Chain Intelligence
SHAP Model Explainability Module
================================

This module provides model interpretability using SHAP values.

Author: ARUN AK
Course: UCS321 - AI for Engineers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
FIGURES_PATH = "reports/figures/"

def main():
    print("\n" + "=" * 60)
    print("AMAZON SUPPLY CHAIN INTELLIGENCE")
    print("SHAP Explainability Analysis")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Load data and model
    print("\n▶ Loading data and model...")
    X_test = pd.read_csv(f"{PROCESSED_DATA_PATH}X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_PATH}y_test.csv").values.ravel()
    model = joblib.load(f"{MODEL_PATH}best_model.joblib")
    model_info = joblib.load(f"{MODEL_PATH}model_info.joblib")

    print(f"  ✓ Loaded {model_info.get('model_name', 'Model')}")
    print(f"  ✓ Test samples: {len(X_test)}")

    # Use a sample for SHAP (faster computation)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    print(f"\n▶ Computing SHAP values (using {sample_size} samples)...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("  ✓ SHAP values computed")

    # Feature names
    feature_names = X_test.columns.tolist()

    # =========================================================================
    # 1. SHAP Summary Plot (Bar)
    # =========================================================================
    print("\n▶ Generating SHAP summary plots...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, class_names=['On-Time', 'At-Risk', 'Delayed'])
    plt.title('SHAP Feature Importance (All Classes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}shap_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved shap_summary_bar.png")

    # =========================================================================
    # 2. SHAP Summary Plot (Dot) for Delayed class (most important)
    # =========================================================================
    class_names = ['On-Time', 'At-Risk', 'Delayed']

    # Just create one detailed summary plot for the Delayed class
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], X_sample, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Values - Impact on Delay Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}shap_summary_delayed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved shap_summary_delayed.png")

    # =========================================================================
    # 3. Calculate feature importance from SHAP
    # =========================================================================
    print("\n▶ Calculating SHAP-based feature importance...")

    # Debug shape info
    if isinstance(shap_values, list):
        print(f"  → SHAP values: list of {len(shap_values)} arrays")
        print(f"  → First array shape: {shap_values[0].shape}")
        # Multi-class: use first class or average
        mean_abs_shap = np.abs(shap_values[2]).mean(axis=0)  # Use delayed class
    else:
        print(f"  → SHAP values shape: {shap_values.shape}")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Ensure 1D
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()

    print(f"  → Mean SHAP shape: {mean_abs_shap.shape}")
    print(f"  → Number of features: {len(feature_names)}")

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(mean_abs_shap)],
        'SHAP_Importance': mean_abs_shap[:len(feature_names)]
    }).sort_values('SHAP_Importance', ascending=False)

    top_features = importance_df['Feature'].head(10).tolist()
    print("  ✓ Feature importance calculated")

    # =========================================================================
    # 4. SHAP Waterfall Plot for a single prediction
    # =========================================================================
    print("\n▶ Generating SHAP waterfall plot...")

    try:
        # Get one example
        example_idx = 0
        if isinstance(shap_values, list):
            sv = shap_values[2][example_idx]  # Delayed class
            base_val = explainer.expected_value[2]
        else:
            sv = shap_values[example_idx]
            base_val = explainer.expected_value

        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base_val,
                data=X_sample.iloc[example_idx],
                feature_names=feature_names
            ),
            show=False,
            max_display=12
        )
        plt.title('SHAP Explanation - Single Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{FIGURES_PATH}shap_waterfall_example.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved shap_waterfall_example.png")
    except Exception as e:
        print(f"  ⚠ Waterfall plot skipped: {str(e)[:50]}")

    # =========================================================================
    # 5. Feature Importance Comparison
    # =========================================================================
    print("\n▶ Creating feature importance comparison...")

    # Model-based importance
    if hasattr(model, 'feature_importances_'):
        importance_df['Model_Importance'] = model.feature_importances_

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot top 15 features
    plot_df = importance_df.head(15).iloc[::-1]  # Reverse for horizontal bar

    x = np.arange(len(plot_df))
    width = 0.35

    ax.barh(x - width/2, plot_df['SHAP_Importance'], width, label='SHAP', color='#FF9900')
    if 'Model_Importance' in plot_df.columns:
        ax.barh(x + width/2, plot_df['Model_Importance'], width, label='Model (Gini)', color='#232F3E')

    ax.set_yticks(x)
    ax.set_yticklabels(plot_df['Feature'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance: SHAP vs Model (Top 15)', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}shap_vs_model_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved shap_vs_model_importance.png")

    # =========================================================================
    # Save SHAP summary
    # =========================================================================
    importance_df.to_csv(f"reports/shap_importance.csv", index=False)
    print(f"\n✓ Saved SHAP importance to reports/shap_importance.csv")

    # Print top features
    print("\n" + "=" * 60)
    print("TOP 5 MOST IMPORTANT FEATURES (SHAP)")
    print("=" * 60)
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['Feature']}: {row['SHAP_Importance']:.4f}")

    print("\n" + "=" * 60)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll SHAP visualizations saved to: {FIGURES_PATH}")

if __name__ == "__main__":
    main()
