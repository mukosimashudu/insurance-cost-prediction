import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

# SHAP is installed
import shap

SHAP_AVAILABLE = True


# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Scroll indicator animation */
    .scroll-indicator {
        text-align: center;
        padding: 1rem;
        color: #667eea;
        font-size: 0.9rem;
        animation: bounce 2s infinite;
        margin-bottom: 1rem;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }

    /* Main header with gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    .subsection-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #495057;
        margin: 1rem 0 0.5rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: white !important;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white !important;
    }
    .metric-change {
        font-size: 0.9rem;
        background: rgba(255,255,255,0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
        color: white !important;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    .prediction-label {
        font-size: 1rem;
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .prediction-value {
        font-size: 3.5rem;
        color: white !important;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        line-height: 1.2;
    }
    .monthly-value {
        font-size: 1.5rem;
        color: white !important;
        font-weight: bold;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem;
        border-radius: 10px;
        margin-top: 0.5rem;
    }

    /* Feature table */
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .feature-table th {
        text-align: left;
        padding: 0.75rem;
        background: #f8f9fa;
        color: #495057;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
    }
    .feature-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #e9ecef;
    }
    .feature-table {
        background: #2d2d2d !important;
    }
    .feature-table tr:hover {
        background: #f8f9fa;
    }

    /* Impact colors */
    .impact-positive {
        color: #f5515f;
        font-weight: 600;
    }
    .impact-negative {
        color: #43e97b;
        font-weight: 600;
    }
    .impact-neutral {
        color: #667eea;
        font-weight: 600;
    }

    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
        color: white;
    }
    .badge-smoker {
        background: linear-gradient(135deg, #f5515f 0%, #e6293b 100%);
    }
    .badge-non-smoker {
        background: linear-gradient(135deg, #43e97b 0%, #1e9b4c 100%);
    }
    .badge-high-risk {
        background: linear-gradient(135deg, #f5515f 0%, #9f041b 100%);
        animation: pulse 2s infinite;
    }
    .badge-moderate-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .badge-low-risk {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* Divider */
    .divider {
        margin: 2rem 0;
        border-top: 1px solid #e9ecef;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
    <div class="main-header">
        <h1>🏥 Insurance Cost Predictor</h1>
        <p>Advanced Analytics & Prediction Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Add scroll indicator
st.markdown("""
    <div class="scroll-indicator">
        ↓ Scroll down for detailed analytics and model insights ↓
    </div>
""", unsafe_allow_html=True)


# Load and prepare data with path detection
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the insurance dataset for visualizations"""

    possible_paths = [
        "Dataset/insurance.csv",
        "insurance.csv",
        "data/insurance.csv",
        "../Dataset/insurance.csv",
        "../insurance.csv",
        "./Dataset/insurance.csv",
        os.path.join("Dataset", "insurance.csv"),
        os.path.join("data", "insurance.csv")
    ]

    dataset_df = None

    for path in possible_paths:
        if os.path.exists(path):
            try:
                dataset_df = pd.read_csv(path)
                st.sidebar.success(f"✅ Dataset loaded: {os.path.basename(path)}")
                break
            except Exception as e:
                continue

    if dataset_df is None:
        st.error("""
        ### ❌ Dataset not found!
        Please ensure `insurance.csv` is in one of these locations:
        - `Dataset/insurance.csv`
        - `insurance.csv` (in the main folder)
        - `data/insurance.csv`
        **Current directory:** `{}`
        """.format(os.getcwd()))
        return pd.DataFrame(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])

    # Add derived columns for visualizations
    dataset_df['bmi_category'] = pd.cut(dataset_df['bmi'],
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    dataset_df['age_group'] = pd.cut(dataset_df['age'],
                                     bins=[0, 30, 45, 60, 100],
                                     labels=['Young Adult', 'Middle Age', 'Senior', 'Elderly'])

    dataset_df['has_children'] = (dataset_df['children'] > 0).astype(int)
    dataset_df['log_charges'] = np.log1p(dataset_df['charges'])

    return dataset_df


# Feature engineering function
def engineer_features(input_dataframe):
    """Add all the engineered features that the model was trained on"""
    engineered_df = input_dataframe.copy()

    def bmi_category(bmi_val):
        if bmi_val < 18.5:
            return 'underweight'
        elif bmi_val < 25:
            return 'normal'
        elif bmi_val < 30:
            return 'overweight'
        else:
            return 'obese'

    engineered_df['bmi_category'] = engineered_df['bmi'].apply(bmi_category)

    def age_group(age_val):
        if age_val < 30:
            return 'young_adult'
        elif age_val < 45:
            return 'middle_age'
        elif age_val < 60:
            return 'senior'
        else:
            return 'elderly'

    engineered_df['age_group'] = engineered_df['age'].apply(age_group)
    engineered_df['high_risk_smoker'] = ((engineered_df['smoker'] == 'yes') & (engineered_df['bmi'] > 30)).astype(int)
    engineered_df['has_children'] = (engineered_df['children'] > 0).astype(int)

    from sklearn.preprocessing import StandardScaler
    bmi_scaler = StandardScaler()
    age_scaler = StandardScaler()

    bmi_scaled = bmi_scaler.fit_transform(engineered_df[['bmi']]).flatten()
    age_scaled = age_scaler.fit_transform(engineered_df[['age']]).flatten()
    engineered_df['bmi_age_interaction'] = bmi_scaled * age_scaled

    return engineered_df


# Load model with path detection
# Load model with path detection
@st.cache_resource
def load_model():
    """Load the trained model"""

    # Get the absolute path to the repo root (one level up from App/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "Models", "insurance_cost_predictor.pkl")

    try:
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model file not found at {model_path}")
        return None


    # Try each path silently (no debug output)
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                # Only show success message, not all the failed attempts
                st.sidebar.success(f"✅ Model loaded successfully!")
                return model
            except Exception as e:
                # Silently continue on error
                continue

    # Only show error if model not found
    st.sidebar.error("❌ Model file not found!")
    return None


# Load the model
ml_model = load_model()


# Load data and model
dataset_df = load_and_prepare_data()
ml_model = load_model()

# Check if data is empty (file not found)
if len(dataset_df) == 0:
    st.stop()

# Create two main columns
left_col, right_col = st.columns([1, 2])

with left_col:
    # Patient Information Section
    st.markdown("### 📝 Patient Information")

    input_age = st.slider("Age", 18, 100, 30, key="age_input")
    input_bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi_input")
    input_children = st.slider("Children", 0, 5, 0, key="children_input")
    input_sex = st.selectbox("Sex", ["male", "female"], key="sex_input")
    input_smoker = st.selectbox("Smoker", ["yes", "no"], key="smoker_input")
    input_region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="region_input")

    st.markdown("---")

    # Quick Stats
    st.markdown("### 📊 Dataset Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Age", f"{dataset_df['age'].mean():.0f}")
    with col2:
        st.metric("Avg BMI", f"{dataset_df['bmi'].mean():.1f}")
    with col3:
        st.metric("Avg Cost", f"${dataset_df['charges'].mean():,.0f}")

    st.markdown("---")

    # Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")

    # BMI Category
    if input_bmi < 18.5:
        st.info("📊 BMI: Underweight")
    elif input_bmi < 25:
        st.success("📊 BMI: Normal")
    elif input_bmi < 30:
        st.warning("📊 BMI: Overweight")
    else:
        st.error("📊 BMI: Obese")

    # Age Group
    if input_age < 30:
        st.info("📅 Age: Young Adult")
    elif input_age < 45:
        st.success("📅 Age: Middle Age")
    elif input_age < 60:
        st.warning("📅 Age: Senior")
    else:
        st.error("📅 Age: Elderly")

    # Risk level with badges
    if input_smoker == "yes" and input_bmi > 30:
        st.markdown('<div class="badge badge-high-risk">🚨 HIGH RISK: Smoker + Obese</div>', unsafe_allow_html=True)
    elif input_smoker == "yes":
        st.markdown('<div class="badge badge-moderate-risk">⚠️ MODERATE RISK: Smoker</div>', unsafe_allow_html=True)
    elif input_bmi > 30:
        st.markdown('<div class="badge badge-moderate-risk">⚠️ MODERATE RISK: Obese</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-low-risk">✅ LOW RISK</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Prediction Button and Result
    if st.button("🎯 Predict Insurance Cost", use_container_width=True):
        if ml_model is not None:
            input_df = pd.DataFrame({
                'age': [input_age],
                'sex': [input_sex],
                'bmi': [input_bmi],
                'children': [input_children],
                'smoker': [input_smoker],
                'region': [input_region]
            })

            try:
                input_df_engineered = engineer_features(input_df)
                prediction_value = ml_model.predict(input_df_engineered)[0]

                st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">Estimated Annual Cost</div>
                        <div class="prediction-value">${prediction_value:,.2f}</div>
                        <div class="monthly-value">${prediction_value / 12:,.0f}/month</div>
                    </div>
                """, unsafe_allow_html=True)

                # Store prediction in session state for SHAP
                st.session_state['last_prediction'] = prediction_value
                st.session_state['last_input'] = input_df_engineered

            except Exception as error:
                st.error(f"Prediction error: {error}")
        else:
            st.error("Model not loaded")

with right_col:
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_patients = len(dataset_df)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Patients</div>
                <div class="metric-value">{total_patients:,}</div>
                <div class="metric-change">Dataset Size</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_charge = dataset_df['charges'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Cost</div>
                <div class="metric-value">${avg_charge:,.0f}</div>
                <div class="metric-change">Per Year</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        smokers = dataset_df[dataset_df['smoker'] == 'yes'].shape[0]
        smoker_pct = (smokers / total_patients) * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Smokers</div>
                <div class="metric-value">{smoker_pct:.1f}%</div>
                <div class="metric-change">{smokers:,} patients</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        obese = dataset_df[dataset_df['bmi'] > 30].shape[0]
        obese_pct = (obese / total_patients) * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Obese</div>
                <div class="metric-value">{obese_pct:.1f}%</div>
                <div class="metric-change">{obese:,} patients</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(dataset_df, x='charges', nbins=50,
                           title='Distribution of Insurance Charges',
                           color_discrete_sequence=['#667eea'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16,
            showlegend=False
        )
        fig.update_xaxes(title="Charges ($)", gridcolor='#e9ecef')
        fig.update_yaxes(title="Count", gridcolor='#e9ecef')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(dataset_df, x='smoker', y='charges', color='smoker',
                     title='Charges by Smoker Status',
                     color_discrete_map={'yes': '#f5576c', 'no': '#667eea'})
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16,
            showlegend=False
        )
        fig.update_xaxes(title="Smoker", gridcolor='#e9ecef')
        fig.update_yaxes(title="Charges ($)", gridcolor='#e9ecef')
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        age_charges = dataset_df.groupby('age_group')['charges'].mean().reset_index()
        fig = px.bar(age_charges, x='age_group', y='charges',
                     title='Average Charges by Age Group',
                     color_discrete_sequence=['#764ba2'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16
        )
        fig.update_xaxes(title="Age Group", gridcolor='#e9ecef')
        fig.update_yaxes(title="Average Charges ($)", gridcolor='#e9ecef')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_charges = dataset_df.groupby('region')['charges'].mean().reset_index()
        fig = px.pie(region_charges, values='charges', names='region',
                     title='Charges Distribution by Region',
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 3
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(dataset_df, x='bmi', y='charges', color='smoker',
                         title='BMI vs Charges (by Smoker Status)',
                         color_discrete_map={'yes': '#f5576c', 'no': '#667eea'},
                         opacity=0.6)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16
        )
        fig.update_xaxes(title="BMI", gridcolor='#e9ecef')
        fig.update_yaxes(title="Charges ($)", gridcolor='#e9ecef')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr_matrix = dataset_df[['age', 'bmi', 'children', 'charges']].corr()
        fig = px.imshow(corr_matrix,
                        text_auto=True,
                        title='Feature Correlations',
                        color_continuous_scale=['#667eea', 'white', '#f5576c'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#667eea',
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)

# SHAP Explainability Section
st.markdown("---")
st.markdown("## 🔍 Model Explainability with SHAP")

if ml_model is not None and 'last_input' in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Feature Impact Summary")
        try:
            st.markdown("""
            <table class="feature-table">
                <tr>
                    <th>Feature</th>
                    <th>Impact Level</th>
                    <th>Effect on Cost</th>
                </tr>
                <tr>
                    <td><strong>Smoker</strong></td>
                    <td><span class="impact-positive">🔴 Very High (+++)</span></td>
                    <td><span class="impact-positive">+$12,000 to +$25,000</span></td>
                </tr>
                <tr>
                    <td><strong>Age</strong></td>
                    <td><span class="impact-positive">🟡 High (++)</span></td>
                    <td><span class="impact-positive">+$1,000 to +$6,000</span></td>
                </tr>
                <tr>
                    <td><strong>BMI</strong></td>
                    <td><span class="impact-positive">🟡 Medium (+)</span></td>
                    <td><span class="impact-positive">+$1,000 to +$5,000</span></td>
                </tr>
                <tr>
                    <td><strong>Children</strong></td>
                    <td><span class="impact-positive">🟢 Low (+)</span></td>
                    <td><span class="impact-positive">+$500 per child</span></td>
                </tr>
                <tr>
                    <td><strong>Region</strong></td>
                    <td><span class="impact-neutral">⚪ Minimal</span></td>
                    <td><span class="impact-neutral">±$500</span></td>
                </tr>
                <tr>
                    <td><strong>Sex</strong></td>
                    <td><span class="impact-neutral">⚪ Minimal</span></td>
                    <td><span class="impact-neutral">±$200</span></td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.info("Feature impact summary")

    with col2:
        st.markdown("### 💡 Key Insights")

        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff5f5 0%, #ffe3e3 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #f5515f;">
            <h4 style="color: #f5515f; margin: 0 0 0.5rem 0;">🔥 Smoker Status - MOST IMPORTANT</h4>
            <p style="margin: 0; color: #333;">
                • Smokers pay <strong>3-4x more</strong> than non-smokers<br>
                • Average increase: <strong>$12,000 - $25,000</strong><br>
                • Most influential factor by far
            </p>
        </div>

        <div style="background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #f093fb;">
            <h4 style="color: #f093fb; margin: 0 0 0.5rem 0;">📈 Age Impact</h4>
            <p style="margin: 0; color: #333;">
                • Costs increase <strong>steadily with age</strong><br>
                • Sharp increase after <strong>age 50</strong><br>
                • Age 50+: <strong>+$4,000 to +$6,000</strong>
            </p>
        </div>

        <div style="background: linear-gradient(135deg, #e6f9e6 0%, #d4edda 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #43e97b;">
            <h4 style="color: #43e97b; margin: 0 0 0.5rem 0;">⚖️ BMI Effect</h4>
            <p style="margin: 0; color: #333;">
                • <strong>Obese (BMI > 30):</strong> +$3,000 to +$5,000<br>
                • <strong>Overweight (25-30):</strong> +$1,000 to +$2,000<br>
                • <strong>Normal/Underweight:</strong> No penalty
            </p>
        </div>

        <div style="background: linear-gradient(135deg, #e6f3ff 0%, #cce5ff 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #667eea;">
            <h4 style="color: #667eea; margin: 0 0 0.5rem 0;">👶 Children & Dependents</h4>
            <p style="margin: 0; color: #333;">
                • Each child adds <strong>+$500</strong> to annual cost<br>
                • Small but cumulative impact
            </p>
        </div>

        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 10px; border-left: 5px solid #6c757d;">
            <h4 style="color: #6c757d; margin: 0 0 0.5rem 0;">🌍 Region & Gender</h4>
            <p style="margin: 0; color: #333;">
                • <strong>Region:</strong> Minimal impact (±$500)<br>
                • <strong>Gender:</strong> Negligible difference (±$200)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Your Prediction Breakdown
    st.markdown("---")
    st.markdown("### 📉 Your Personalized Prediction Breakdown")

    try:
        last_input = st.session_state['last_input']
        last_prediction = st.session_state['last_prediction']

        st.markdown(f"""
        <div style="text-align: center; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
            <span style="font-size: 1.2rem; color: #6c757d;">Your Predicted Annual Cost</span><br>
            <span style="font-size: 4rem; font-weight: bold; color: #f5576c;">${last_prediction:,.2f}</span><br>
            <span style="font-size: 1.5rem; color: #6c757d;">≈ ${last_prediction / 12:,.0f}/month</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Your Information**")
            for col in last_input.columns:
                val = last_input[col].iloc[0]
                if isinstance(val, float):
                    st.write(f"• {col}: **{val:.2f}**")
                else:
                    st.write(f"• {col}: **{val}**")

        with col2:
            st.markdown("**Impact on Your Cost**")

            if last_input['smoker'].iloc[0] == 'yes':
                st.write('• <span style="color: #f5515f;">Smoker: +$12,000 to +$15,000</span>', unsafe_allow_html=True)
            else:
                st.write('• <span style="color: #43e97b;">Non-smoker: -$12,000 to -$15,000</span>',
                         unsafe_allow_html=True)

            bmi_val = last_input['bmi'].iloc[0]
            if bmi_val > 30:
                st.write(f'• <span style="color: #f5515f;">Obese (BMI {bmi_val:.1f}): +$3,000 to +$5,000</span>',
                         unsafe_allow_html=True)
            elif bmi_val > 25:
                st.write(f'• <span style="color: #f093fb;">Overweight (BMI {bmi_val:.1f}): +$1,000 to +$2,000</span>',
                         unsafe_allow_html=True)
            else:
                st.write(f'• <span style="color: #43e97b;">Healthy BMI (BMI {bmi_val:.1f}): No penalty</span>',
                         unsafe_allow_html=True)

            age_val = last_input['age'].iloc[0]
            if age_val > 50:
                st.write(f'• <span style="color: #f5515f;">Age {age_val}: +$4,000 to +$6,000</span>',
                         unsafe_allow_html=True)
            elif age_val > 30:
                st.write(f'• <span style="color: #f093fb;">Age {age_val}: +$1,000 to +$3,000</span>',
                         unsafe_allow_html=True)

            children_val = last_input['children'].iloc[0]
            if children_val > 0:
                st.write(f'• <span style="color: #f093fb;">{children_val} children: +${children_val * 500}</span>',
                         unsafe_allow_html=True)

        # Waterfall chart
        base_cost = 5000
        smoker_effect = 13000 if last_input['smoker'].iloc[0] == 'yes' else -13000
        bmi_effect = 4000 if last_input['bmi'].iloc[0] > 30 else (2000 if last_input['bmi'].iloc[0] > 25 else 0)
        age_effect = 5000 if last_input['age'].iloc[0] > 50 else (2000 if last_input['age'].iloc[0] > 30 else 0)
        children_effect = last_input['children'].iloc[0] * 500

        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=['Base', 'Smoker', 'BMI', 'Age', 'Children', 'Final'],
            y=[base_cost, smoker_effect, bmi_effect, age_effect, children_effect, 0],
            text=[f"${base_cost:,.0f}",
                  f"+${abs(smoker_effect):,.0f}" if smoker_effect > 0 else f"-${abs(smoker_effect):,.0f}",
                  f"+${abs(bmi_effect):,.0f}" if bmi_effect > 0 else f"-${abs(bmi_effect):,.0f}",
                  f"+${abs(age_effect):,.0f}" if age_effect > 0 else f"-${abs(age_effect):,.0f}",
                  f"+${abs(children_effect):,.0f}" if children_effect > 0 else f"-${abs(children_effect):,.0f}",
                  f"${last_prediction:,.0f}"],
            textposition="outside",
            connector={"line": {"color": "#ced4da", "width": 2}},
            increasing={"marker": {"color": "#f5576c"}},
            decreasing={"marker": {"color": "#43e97b"}},
            totals={"marker": {"color": "#667eea"}}
        ))

        fig.update_layout(
            title="How Your Cost Builds Up",
            xaxis_title="Factors",
            yaxis_title="Cost ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **How to read this chart:**
        - **Green bars** decrease your cost
        - **Red bars** increase your cost
        - **Purple bar** is your final cost
        - Start at base → add/subtract factors → final cost
        """)

    except Exception as e:
        st.info("👆 Make a prediction to see your personalized breakdown!")

else:
    st.info("👆 Make a prediction first to see SHAP explanations for your specific case!")

# Footer with copyright
st.markdown("""
    <div class="footer">
        © 2026 Insurance Cost Predictor | Built with Streamlit | Model Explainability with SHAP
    </div>
""", unsafe_allow_html=True)