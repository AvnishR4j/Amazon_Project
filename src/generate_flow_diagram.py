"""
Generate Flow Diagram for Amazon Supply Chain Intelligence
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_flow_diagram():
    """Create a professional flow diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Colors
    amazon_orange = '#FF9900'
    amazon_dark = '#232F3E'
    success_green = '#28a745'
    warning_yellow = '#ffc107'
    danger_red = '#dc3545'
    light_blue = '#e7f3ff'

    # Title
    ax.text(8, 11.5, 'Amazon Supply Chain Intelligence',
            fontsize=20, fontweight='bold', ha='center', color=amazon_dark)
    ax.text(8, 11, 'ML Pipeline Flow Diagram',
            fontsize=14, ha='center', color='gray')

    # Box positions (x, y, width, height)
    boxes = [
        # Row 1 - Data Input
        (1, 9, 2.5, 1, 'Data Input\n43,739 Records', amazon_orange, 'white'),

        # Row 2 - Preprocessing
        (4.5, 9, 2.5, 1, 'Data Cleaning\n& Preprocessing', light_blue, amazon_dark),
        (8, 9, 2.5, 1, 'Feature\nEngineering', light_blue, amazon_dark),
        (11.5, 9, 2.5, 1, 'Target\nCreation', light_blue, amazon_dark),

        # Row 3 - Balancing and Split
        (3, 6.5, 2.5, 1, 'SMOTE\nBalancing', '#e8f5e9', amazon_dark),
        (6.5, 6.5, 2.5, 1, 'Train/Test\nSplit (80/20)', '#e8f5e9', amazon_dark),

        # Row 4 - Models
        (1, 4, 1.5, 0.8, 'Logistic\nRegression', '#fff3e0', amazon_dark),
        (3, 4, 1.5, 0.8, 'Random\nForest', '#fff3e0', amazon_dark),
        (5, 4, 1.5, 0.8, 'XGBoost', success_green, 'white'),
        (7, 4, 1.5, 0.8, 'Gradient\nBoosting', '#fff3e0', amazon_dark),
        (9, 4, 1.5, 0.8, 'SVM', '#fff3e0', amazon_dark),
        (11, 4, 1.5, 0.8, 'KNN', '#fff3e0', amazon_dark),
        (13, 4, 1.5, 0.8, 'Decision\nTree', '#fff3e0', amazon_dark),

        # Row 5 - Evaluation
        (5.5, 2, 3, 1, 'Model Evaluation\n5-Fold CV', '#e3f2fd', amazon_dark),
        (10, 2, 3, 1, 'Best Model\nXGBoost (79.3%)', success_green, 'white'),

        # Row 6 - Output
        (7, 0.3, 4, 1, 'Streamlit Dashboard\nReal-time Predictions', amazon_orange, 'white'),
    ]

    # Draw boxes
    for (x, y, w, h, text, facecolor, textcolor) in boxes:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=facecolor, edgecolor=amazon_dark, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=9, fontweight='bold', color=textcolor)

    # Draw arrows
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=4"
    arrows = [
        # Row 1 to Row 2
        ((2.25, 9), (4.5, 9.5)),
        ((5.75, 9), (8, 9.5)),
        ((9.25, 9), (11.5, 9.5)),

        # Row 2 to Row 3
        ((12.75, 9), (12.75, 7.8)),
        ((12.75, 7.5), (9, 7.5)),
        ((6.5, 7), (5.5, 7)),
        ((4.25, 6.5), (6.5, 7)),

        # Row 3 to Row 4
        ((7.75, 6.5), (7.75, 5)),

        # Model row to evaluation
        ((5.75, 4), (5.75, 3)),
        ((7, 3.2), (8.5, 2.5)),

        # Evaluation to best model
        ((8.5, 2.5), (10, 2.5)),

        # Best model to dashboard
        ((11.5, 2), (11.5, 1.5)),
        ((11, 1.3), (11, 1.3)),
    ]

    # Simplified arrows
    ax.annotate('', xy=(4.5, 9.5), xytext=(3.5, 9.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))
    ax.annotate('', xy=(8, 9.5), xytext=(7, 9.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))
    ax.annotate('', xy=(11.5, 9.5), xytext=(10.5, 9.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # Vertical arrows
    ax.annotate('', xy=(4.25, 7.5), xytext=(4.25, 9),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))
    ax.annotate('', xy=(6.5, 7.5), xytext=(5.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # To models
    ax.annotate('', xy=(7.75, 4.8), xytext=(7.75, 6.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # To evaluation
    ax.annotate('', xy=(7, 3), xytext=(7, 4),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # To best model
    ax.annotate('', xy=(10, 2.5), xytext=(8.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # To dashboard
    ax.annotate('', xy=(9, 1.3), xytext=(10, 2),
                arrowprops=dict(arrowstyle='->', color=amazon_dark, lw=2))

    # Legend
    legend_items = [
        mpatches.Patch(color=amazon_orange, label='Input/Output'),
        mpatches.Patch(color=light_blue, label='Preprocessing'),
        mpatches.Patch(color='#e8f5e9', label='Data Preparation'),
        mpatches.Patch(color='#fff3e0', label='ML Models'),
        mpatches.Patch(color=success_green, label='Best Model'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=9)

    # Add metrics box
    metrics_text = """Performance Metrics:
• Accuracy: 79.3%
• Precision: 79.3%
• Recall: 79.3%
• F1-Score: 0.79"""

    props = dict(boxstyle='round', facecolor='white', edgecolor=amazon_dark, alpha=0.9)
    ax.text(14.5, 5.5, metrics_text, fontsize=9, verticalalignment='top',
            bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig('reports/figures/flow_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Flow diagram saved to reports/figures/flow_diagram.png")

if __name__ == "__main__":
    create_flow_diagram()
