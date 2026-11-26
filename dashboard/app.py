"""
Amazon Supply Chain Intelligence
Interactive Streamlit Dashboard
================================

This dashboard provides:
- Real-time delivery delay risk prediction
- Data exploration and analytics
- Model performance visualization
- Business insights and recommendations

Author: ARUN AK
Course: UCS321 - AI for Engineers

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta

from features import fit_preprocessor, raw_features, to_model_input

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Amazon Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main theme colors - Amazon inspired */
    :root {
        --amazon-orange: #FF9900;
        --amazon-dark: #232F3E;
        --amazon-light: #37475A;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .main-header h1 {
        color: #FF9900;
        margin: 0;
        font-size: 2.2rem;
    }

    .main-header p {
        color: #ffffff;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #FF9900;
    }

    /* Prediction result cards */
    .prediction-ontime {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
    }

    .prediction-atrisk {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 10px;
    }

    .prediction-delayed {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 10px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #232F3E;
    }

    /* Button styling */
    .stButton > button {
        background-color: #FF9900;
        color: #232F3E;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }

    .stButton > button:hover {
        background-color: #e68a00;
        color: white;
    }

    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load processed dataset."""
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
        return df
    except FileNotFoundError:
        st.error("Data not found. Please run preprocessing first.")
        return None


@st.cache_resource
def load_preprocessor():
    """Rebuild the category codes and scaler used in training."""
    return fit_preprocessor(pd.read_csv("data/processed/processed_data.csv"))


@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        model = joblib.load("models/best_model.joblib")
        model_info = joblib.load("models/model_info.joblib")
        return model, model_info
    except FileNotFoundError:
        st.error("Model not found. Please run model training first.")
        return None, None


@st.cache_data
def load_model_results():
    """Load model comparison results."""
    try:
        return pd.read_csv("reports/model_comparison.csv")
    except FileNotFoundError:
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_prediction_color(prediction):
    """Return color based on prediction class."""
    colors = {0: '#28a745', 1: '#ffc107', 2: '#dc3545'}
    return colors.get(prediction, '#6c757d')


def get_prediction_label(prediction):
    """Return label based on prediction class."""
    labels = {0: 'On-Time', 1: 'At-Risk', 2: 'Delayed'}
    return labels.get(prediction, 'Unknown')


def get_prediction_emoji(prediction):
    """Return emoji based on prediction class."""
    emojis = {0: '✅', 1: '⚠️', 2: '🚨'}
    return emojis.get(prediction, '❓')


def get_recommendation(prediction, features):
    """Generate business recommendations based on prediction."""
    recommendations = []

    if prediction == 2:  # Delayed
        recommendations.append("🚨 **Priority Alert**: High delay risk detected!")
        recommendations.append("📞 Proactively notify customer about potential delay")
        recommendations.append("🚚 Consider expedited shipping options")

        if features.get('weather_severity', 0) > 3:
            recommendations.append("🌧️ Weather conditions are unfavorable - monitor closely")
        if features.get('distance_km', 0) > 20:
            recommendations.append("📍 Long distance delivery - allocate experienced driver")

    elif prediction == 1:  # At-Risk
        recommendations.append("⚠️ **Moderate Risk**: Monitor this delivery closely")
        recommendations.append("📊 Add to priority tracking queue")

        if features.get('is_rush_hour', 0) == 1:
            recommendations.append("🕐 Rush hour delivery - expect traffic delays")

    else:  # On-Time
        recommendations.append("✅ **Low Risk**: Delivery expected on time")
        recommendations.append("📦 Standard processing is sufficient")

    return recommendations


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #FF9900;">📦 Amazon SCI</h2>
        <p style="color: #888;">Supply Chain Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict Delay", "📊 Data Analytics", "🤖 Model Performance", "💡 Business Insights"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Model info
    model, model_info = load_model()
    if model_info:
        st.sidebar.markdown("### 🎯 Model Info")
        st.sidebar.markdown(f"**Model:** {model_info.get('model_name', 'N/A')}")
        st.sidebar.markdown(f"**Test F1:** {model_info.get('cv_score', 0):.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        <p>Built for UCS321</p>
        <p>AI for Engineers</p>
        <p>© 2024 ARUN AK</p>
    </div>
    """, unsafe_allow_html=True)

    return page


# ============================================================================
# HOME PAGE
# ============================================================================

def render_home():
    """Render home page."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📦 Amazon Supply Chain Intelligence</h1>
        <p>AI-Powered Delivery Delay Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data for stats
    df = load_data()
    if df is None:
        return

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📦 Total Orders",
            value=f"{len(df):,}",
            delta="Analyzed"
        )

    with col2:
        on_time_pct = (df['delivery_status'] == 0).mean() * 100
        st.metric(
            label="✅ On-Time Rate",
            value=f"{on_time_pct:.1f}%",
            delta=f"{on_time_pct - 85:.1f}% vs target"
        )

    with col3:
        delayed_pct = (df['delivery_status'] == 2).mean() * 100
        st.metric(
            label="🚨 Delayed Rate",
            value=f"{delayed_pct:.1f}%",
            delta=f"-{delayed_pct:.1f}%",
            delta_color="inverse"
        )

    with col4:
        avg_distance = df['distance_km'].mean()
        st.metric(
            label="📍 Avg Distance",
            value=f"{avg_distance:.1f} km",
            delta="Mean delivery"
        )

    st.markdown("---")

    # Quick overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Delivery Status Distribution")

        status_counts = df['delivery_status'].value_counts().sort_index()
        labels = ['On-Time', 'At-Risk', 'Delayed']
        colors = ['#28a745', '#ffc107', '#dc3545']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=status_counts.values,
            hole=0.4,
            marker_colors=colors
        )])
        fig.update_layout(
            showlegend=True,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🚚 Orders by Category")

        category_counts = df['category'].value_counts().head(8)

        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            color=category_counts.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Number of Orders",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    # Features overview
    st.markdown("### 🎯 System Features")

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔮 Real-Time Prediction</h4>
            <p>Instant delivery delay risk assessment for new orders</p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Data Analytics</h4>
            <p>Comprehensive analysis of delivery patterns and trends</p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Smart Insights</h4>
            <p>AI-driven recommendations for logistics optimization</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PREDICTION PAGE
# ============================================================================

def render_prediction():
    """Render prediction page."""
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Delivery Delay Prediction</h1>
        <p>Enter order details to predict delivery risk</p>
    </div>
    """, unsafe_allow_html=True)

    model, model_info = load_model()
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        return

    df = load_data()
    if df is None:
        return
    codes, scaler = load_preprocessor()
    months = sorted(int(m) for m in df['month'].unique())
    delays = sorted(int(d) for d in df['pickup_delay_mins'].unique())

    # Input form
    st.markdown("### 📝 Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        agent_age = st.slider("Agent Age", int(df['agent_age'].min()), int(df['agent_age'].max()), 30)
        agent_rating = st.slider("Agent Rating", float(df['agent_rating'].min()), 5.0, 4.5, 0.1)
        distance_km = st.slider("Distance (km)", 1.5, 21.0, 10.0, 0.5)

    with col2:
        weather = st.selectbox("Weather", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])
        traffic = st.selectbox("Traffic", ['Low', 'Medium', 'High', 'Jam'])
        vehicle = st.selectbox("Vehicle Type", sorted(codes['vehicle']))

    with col3:
        area = st.selectbox("Area Type", sorted(codes['area']))
        category = st.selectbox("Product Category", sorted(codes['category']))
        order_hour = st.slider("Order Hour", 0, 23, 12)

    # Additional features
    st.markdown("### ⚙️ Additional Parameters")
    col1, col2 = st.columns(2)

    with col1:
        pickup_delay = st.selectbox("Pickup Delay (minutes, as in training data)", delays)
        st.caption("Weekend and rush hour are worked out from the day and order hour.")

    with col2:
        day_of_week = st.selectbox("Day of Week", list(range(7)),
                                   format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        month = st.selectbox("Month", months,
                             format_func=lambda m: datetime(2022, m, 1).strftime('%B'))

    # Predict button
    if st.button("🔮 Predict Delivery Risk", use_container_width=True):

        # Prepare features exactly as in training
        features = raw_features(
            codes, agent_age=agent_age, agent_rating=agent_rating,
            distance_km=distance_km, day_of_week=day_of_week, month=month,
            order_hour=order_hour, pickup_delay_mins=pickup_delay,
            weather=weather, traffic=traffic, vehicle=vehicle,
            area=area, category=category
        )
        input_df = to_model_input([features], scaler)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Display result
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        prediction_class = ['prediction-ontime', 'prediction-atrisk', 'prediction-delayed'][prediction]
        prediction_label = get_prediction_label(prediction)
        prediction_emoji = get_prediction_emoji(prediction)

        st.markdown(f"""
        <div class="{prediction_class}">
            <h2 style="margin: 0;">{prediction_emoji} Predicted Status: {prediction_label}</h2>
            <p style="margin: 0.5rem 0 0 0;">Confidence: {probabilities[prediction]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("#### 📊 Probability Distribution")

        prob_df = pd.DataFrame({
            'Status': ['On-Time', 'At-Risk', 'Delayed'],
            'Probability': probabilities * 100
        })

        fig = px.bar(
            prob_df, x='Status', y='Probability',
            color='Status',
            color_discrete_map={'On-Time': '#28a745', 'At-Risk': '#ffc107', 'Delayed': '#dc3545'}
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("#### 💡 Recommendations")
        recommendations = get_recommendation(prediction, features)
        for rec in recommendations:
            st.markdown(rec)


# ============================================================================
# ANALYTICS PAGE
# ============================================================================

def render_analytics():
    """Render data analytics page."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Analytics</h1>
        <p>Explore delivery patterns and trends</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    selected_weather = st.sidebar.multiselect(
        "Weather",
        df['weather'].unique().tolist(),
        default=df['weather'].unique().tolist()
    )
    selected_traffic = st.sidebar.multiselect(
        "Traffic",
        df['traffic'].unique().tolist(),
        default=df['traffic'].unique().tolist()
    )

    # Filter data
    filtered_df = df[
        (df['weather'].isin(selected_weather)) &
        (df['traffic'].isin(selected_traffic))
    ]

    st.markdown(f"**Showing {len(filtered_df):,} orders** (filtered from {len(df):,} total)")

    # Analytics tabs
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📅 Time Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Delivery Time Distribution")
            fig = px.histogram(
                filtered_df, x='delivery_time', nbins=50,
                color_discrete_sequence=['#FF9900']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Distance Distribution")
            fig = px.histogram(
                filtered_df, x='distance_km', nbins=50,
                color_discrete_sequence=['#232F3E']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Delivery Status by Weather Condition")
        weather_status = pd.crosstab(
            filtered_df['weather'],
            filtered_df['delivery_status'],
            normalize='index'
        ) * 100

        fig = px.bar(
            weather_status,
            barmode='stack',
            color_discrete_map={0: '#28a745', 1: '#ffc107', 2: '#dc3545'}
        )
        fig.update_layout(height=400, yaxis_title="Percentage")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Feature Correlations with Delivery Time")

        numeric_cols = ['agent_age', 'agent_rating', 'distance_km', 'delivery_time',
                       'weather_severity', 'is_rush_hour', 'is_weekend']
        numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]

        corr_matrix = filtered_df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Delivery Patterns by Hour")

        hourly_stats = filtered_df.groupby('order_hour').agg({
            'delivery_time': 'mean',
            'delivery_status': lambda x: (x == 2).mean() * 100
        }).reset_index()
        hourly_stats.columns = ['Hour', 'Avg Delivery Time', 'Delay Rate %']

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=hourly_stats['Hour'], y=hourly_stats['Avg Delivery Time'],
                   name='Avg Delivery Time', marker_color='#FF9900'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=hourly_stats['Hour'], y=hourly_stats['Delay Rate %'],
                       name='Delay Rate %', line=dict(color='#dc3545', width=3)),
            secondary_y=True
        )
        fig.update_layout(height=400)
        fig.update_yaxes(title_text="Avg Delivery Time (mins)", secondary_y=False)
        fig.update_yaxes(title_text="Delay Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================

def render_model_performance():
    """Render model performance page."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Model Performance</h1>
        <p>Evaluation metrics and comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    results_df = load_model_results()

    if results_df is None:
        st.warning("Model results not found. Please train models first.")
        return

    # Best model highlight
    best_model = results_df.iloc[0]
    st.markdown("### 🏆 Best Performing Model")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Model", best_model['Model'])
    with col2:
        st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    with col3:
        st.metric("Precision", f"{best_model['Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{best_model['Recall']:.4f}")
    with col5:
        st.metric("F1-Score", f"{best_model['F1-Score']:.4f}")

    st.markdown("---")

    # Model comparison
    st.markdown("### 📊 Model Comparison")

    # Bar chart comparison
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF9900', '#232F3E', '#37475A', '#2ecc71']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            marker_color=colors[i]
        ))

    fig.update_layout(
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("### 📋 Detailed Results")
    st.dataframe(
        results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#90EE90'),
        use_container_width=True
    )

    # Load and display saved figures
    st.markdown("### 📈 Evaluation Visualizations")

    figures_path = "reports/figures/"

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(f"{figures_path}confusion_matrix.png"):
            st.image(f"{figures_path}confusion_matrix.png", caption="Confusion Matrix")

    with col2:
        if os.path.exists(f"{figures_path}roc_curves.png"):
            st.image(f"{figures_path}roc_curves.png", caption="ROC Curves")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(f"{figures_path}feature_importance.png"):
            st.image(f"{figures_path}feature_importance.png", caption="Feature Importance")

    with col2:
        if os.path.exists(f"{figures_path}learning_curves.png"):
            st.image(f"{figures_path}learning_curves.png", caption="Learning Curves")


# ============================================================================
# BUSINESS INSIGHTS PAGE
# ============================================================================

def render_insights():
    """Render business insights page."""
    st.markdown("""
    <div class="main-header">
        <h1>💡 Business Insights</h1>
        <p>Data-driven recommendations for logistics optimization</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    st.warning("The findings, impact figures and ROI projection on this page are illustrative. "
               "They are typed into the app, not calculated from the data or the model.")

    # Key insights
    st.markdown("### 🎯 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📍 Distance Impact</h4>
            <p>Orders with distance > 20 km have <strong>3x higher</strong> delay probability.
            Consider prioritizing nearby warehouse allocation for long-distance orders.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>🌧️ Weather Sensitivity</h4>
            <p>Stormy and Sandstorm conditions increase delays by <strong>45%</strong>.
            Implement weather-aware routing and customer notifications.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🕐 Rush Hour Effect</h4>
            <p>Orders placed during rush hours (8-10 AM, 5-7 PM) show <strong>25%</strong>
            higher delay rates. Consider traffic-aware scheduling.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>⭐ Agent Performance</h4>
            <p>Agents with rating > 4.5 deliver <strong>20%</strong> faster on average.
            Invest in training programs for lower-rated agents.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.markdown("### 📋 Strategic Recommendations")

    recommendations = [
        {
            "priority": "High",
            "area": "Real-time Monitoring",
            "action": "Implement real-time delay prediction for all incoming orders",
            "impact": "Reduce late deliveries by 15-20%"
        },
        {
            "priority": "High",
            "area": "Customer Communication",
            "action": "Proactive notifications for at-risk deliveries",
            "impact": "Improve customer satisfaction scores by 25%"
        },
        {
            "priority": "Medium",
            "area": "Route Optimization",
            "action": "Integrate weather and traffic data into routing algorithms",
            "impact": "Reduce delivery times by 10-15%"
        },
        {
            "priority": "Medium",
            "area": "Agent Allocation",
            "action": "Assign high-rated agents to high-risk deliveries",
            "impact": "Improve on-time delivery rate by 8%"
        },
        {
            "priority": "Low",
            "area": "Demand Forecasting",
            "action": "Predict high-volume periods for better resource allocation",
            "impact": "Optimize operational costs by 10%"
        }
    ]

    rec_df = pd.DataFrame(recommendations)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    # ROI Calculator
    st.markdown("---")
    st.markdown("### 💰 ROI Calculator")

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_orders = st.number_input("Monthly Orders", value=100000, step=10000)

    with col2:
        current_delay_rate = st.slider("Current Delay Rate (%)", 0, 50, 23)

    with col3:
        cost_per_delay = st.number_input("Cost per Delayed Order ($)", value=15, step=1)

    improvement = 0.15  # 15% improvement expected
    new_delay_rate = current_delay_rate * (1 - improvement)
    delays_prevented = monthly_orders * (current_delay_rate - new_delay_rate) / 100
    monthly_savings = delays_prevented * cost_per_delay

    st.markdown("#### Projected Savings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("New Delay Rate", f"{new_delay_rate:.1f}%", f"-{current_delay_rate - new_delay_rate:.1f}%")
    with col2:
        st.metric("Delays Prevented/Month", f"{delays_prevented:,.0f}")
    with col3:
        st.metric("Monthly Savings", f"${monthly_savings:,.0f}", f"+${monthly_savings:,.0f}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""

    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🔮 Predict Delay":
        render_prediction()
    elif page == "📊 Data Analytics":
        render_analytics()
    elif page == "🤖 Model Performance":
        render_model_performance()
    elif page == "💡 Business Insights":
        render_insights()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Amazon Supply Chain Intelligence</strong> | Built with Streamlit</p>
        <p>UCS321 - AI for Engineers | ARUN AK | 2024</p>
        <p>Student project. Not affiliated with or endorsed by Amazon.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
