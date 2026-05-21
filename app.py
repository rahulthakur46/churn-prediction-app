import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Dark Cyber Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --neon-cyan: #00f5ff;
    --neon-pink: #ff006e;
    --neon-green: #39ff14;
    --dark-bg: #050a15;
    --card-bg: #0d1b2a;
    --card-border: #1a3a5c;
    --text-primary: #e0f7ff;
    --text-muted: #5a8fa8;
}

/* Base background */
.stApp {
    background: var(--dark-bg);
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(0,245,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(255,0,110,0.04) 0%, transparent 60%);
}

/* All text */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #080f1c !important;
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--neon-cyan) !important;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0d1b2a !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.3) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--neon-cyan) !important;
}
.stSlider > div > div > div {
    background: #1a3a5c !important;
}

/* Number input */
.stNumberInput input {
    background: #0d1b2a !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00f5ff22, #ff006e22) !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.6rem 2rem !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00f5ff44, #ff006e44) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.5), 0 0 40px rgba(0,245,255,0.2) !important;
    transform: translateY(-2px) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--card-border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
}

/* Divider */
hr { border-color: var(--card-border) !important; }

/* Dataframe */
.stDataFrame { border: 1px solid var(--card-border) !important; border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--neon-cyan); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_prediction_dataset.csv")
    return df

@st.cache_resource
def train_model(df):
    """Train a fresh logistic regression on the dataset (since pkl has version issues)."""
    data = df.copy()
    data['Churn_bin'] = (data['Churn'] == 'Yes').astype(int)

    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod']

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col + '_enc'] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    feature_cols = ['gender_enc','SeniorCitizen','Partner_enc','Dependents_enc','tenure',
                    'PhoneService_enc','MultipleLines_enc','InternetService_enc',
                    'OnlineSecurity_enc','OnlineBackup_enc','DeviceProtection_enc',
                    'TechSupport_enc','StreamingTV_enc','StreamingMovies_enc',
                    'Contract_enc','PaperlessBilling_enc','PaymentMethod_enc',
                    'MonthlyCharges','TotalCharges']

    X = data[feature_cols]
    y = data['Churn_bin']

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, encoders, feature_cols

def predict_churn(model, encoders, feature_cols, inputs):
    row = {}
    cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies','Contract',
                'PaperlessBilling','PaymentMethod']
    for col in cat_cols:
        le = encoders[col]
        val = inputs[col]
        if val in le.classes_:
            row[col + '_enc'] = le.transform([val])[0]
        else:
            row[col + '_enc'] = 0

    row['SeniorCitizen'] = inputs['SeniorCitizen']
    row['tenure'] = inputs['tenure']
    row['MonthlyCharges'] = inputs['MonthlyCharges']
    row['TotalCharges'] = inputs['TotalCharges']

    X = pd.DataFrame([row])[feature_cols]
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return prob, pred


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
df = load_data()
model, encoders, feature_cols = train_model(df)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem 0;">
    <div style="font-family:'Orbitron',monospace; font-size:2.8rem; font-weight:900;
                background: linear-gradient(135deg, #00f5ff, #ff006e);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                letter-spacing: 0.05em; line-height: 1.1;">
        ⚡ CHURNSENSE AI
    </div>
    <div style="font-family:'Rajdhani',sans-serif; color:#5a8fa8; font-size:1rem;
                letter-spacing:0.3em; text-transform:uppercase; margin-top:0.4rem;">
        Customer Retention Intelligence Platform
    </div>
    <div style="width:120px; height:2px; background:linear-gradient(90deg,#00f5ff,#ff006e);
                margin: 1rem auto 0; border-radius:2px;"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP KPI STRIP
# ─────────────────────────────────────────────
total = len(df)
churned = (df['Churn'] == 'Yes').sum()
churn_rate = churned / total * 100
avg_tenure = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("📊 Total Customers", f"{total:,}")
with k2:
    st.metric("🚨 Churned", f"{churned:,}")
with k3:
    st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
with k4:
    st.metric("💰 Avg Monthly Charge", f"₹{avg_monthly:.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  PREDICT", "📊  ANALYTICS", "🗂  DATA"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00f5ff;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem;">
            ◈ Customer Profile
        </div>
        """, unsafe_allow_html=True)

        # ── Demographics
        st.markdown('<p style="color:#5a8fa8;font-size:0.75rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem;">— Demographics —</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])

        c4, c5 = st.columns(2)
        with c4:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with c5:
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        # ── Services
        st.markdown('<p style="color:#5a8fa8;font-size:0.75rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem;margin-top:0.8rem;">— Services —</p>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with c7:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        with c8:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        # ── Billing
        st.markdown('<p style="color:#5a8fa8;font-size:0.75rem;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem;margin-top:0.8rem;">— Billing —</p>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 200.0, 65.0, step=0.5)
        with c10:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            total_charges = st.number_input("Total Charges (₹)", 0.0, 10000.0, float(monthly_charges * tenure), step=1.0)

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ ANALYZE CHURN RISK")

    with col_result:
        st.markdown("""
        <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00f5ff;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem;">
            ◈ Risk Assessment
        </div>
        """, unsafe_allow_html=True)

        if predict_btn:
            inputs = {
                'gender': gender,
                'SeniorCitizen': 1 if senior == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }

            prob, pred = predict_churn(model, encoders, feature_cols, inputs)
            risk_pct = prob * 100
            retain_pct = (1 - prob) * 100

            # Verdict banner
            if pred == 1:
                color = "#ff006e"
                label = "HIGH CHURN RISK"
                icon = "🚨"
                glow = "rgba(255,0,110,0.4)"
            else:
                color = "#39ff14"
                label = "LOW CHURN RISK"
                icon = "✅"
                glow = "rgba(57,255,20,0.4)"

            st.markdown(f"""
            <div style="background: {color}18; border: 1px solid {color};
                        border-radius: 12px; padding: 1.5rem; text-align:center;
                        box-shadow: 0 0 30px {glow}; margin-bottom:1.5rem;">
                <div style="font-size:2.5rem;">{icon}</div>
                <div style="font-family:'Orbitron',monospace; color:{color};
                            font-size:1.4rem; font-weight:900; letter-spacing:0.1em;
                            margin-top:0.5rem;">{label}</div>
                <div style="font-family:'Rajdhani',sans-serif; color:#aaa;
                            font-size:0.9rem; margin-top:0.3rem;">
                    Confidence: {max(risk_pct, retain_pct):.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Churn Probability", 'font': {'color': '#5a8fa8', 'family': 'Rajdhani', 'size': 14}},
                number={'suffix': "%", 'font': {'color': '#e0f7ff', 'family': 'Orbitron', 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': '#5a8fa8', 'size': 10}},
                    'bar': {'color': color, 'thickness': 0.3},
                    'bgcolor': '#0d1b2a',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': '#0d2a1a'},
                        {'range': [30, 70], 'color': '#2a1a0d'},
                        {'range': [70, 100], 'color': '#2a0d1a'},
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 3},
                        'thickness': 0.8,
                        'value': risk_pct
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(t=30, b=10, l=20, r=20),
                font_color='#e0f7ff'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk breakdown donut
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Churn Risk', 'Retention'],
                values=[risk_pct, retain_pct],
                hole=0.65,
                marker_colors=['#ff006e', '#00f5ff'],
                textinfo='label+percent',
                textfont=dict(family='Rajdhani', size=12, color='#e0f7ff'),
                hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
            )])
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=230,
                margin=dict(t=10, b=10, l=0, r=0),
                showlegend=False,
                annotations=[dict(
                    text=f"{risk_pct:.0f}%<br>Risk",
                    x=0.5, y=0.5, font_size=16,
                    font_color=color, font_family='Orbitron',
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Recommendations
            st.markdown("""
            <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#00f5ff;
                        letter-spacing:0.12em; text-transform:uppercase; margin:1rem 0 0.5rem;">
                ◈ Recommended Actions
            </div>
            """, unsafe_allow_html=True)

            recs = []
            if contract == "Month-to-month":
                recs.append("🔒 Offer a discounted 1-year contract lock-in")
            if online_security == "No":
                recs.append("🛡️ Bundle online security package at no extra cost")
            if tech_support == "No":
                recs.append("📞 Provide complimentary tech support trial")
            if monthly_charges > 70:
                recs.append("💸 Send personalized loyalty discount coupon")
            if tenure < 12:
                recs.append("🎁 Onboarding reward for first-year retention")
            if not recs:
                recs.append("✨ Customer is stable — maintain regular engagement")

            for r in recs:
                st.markdown(f"""
                <div style="background:#0d1b2a; border-left:3px solid #00f5ff;
                            padding:0.5rem 0.8rem; border-radius:0 6px 6px 0;
                            margin-bottom:0.4rem; font-family:'Rajdhani';
                            font-size:0.9rem; color:#c0e8f5;">
                    {r}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem;
                        border:1px dashed #1a3a5c; border-radius:16px;
                        background:rgba(0,245,255,0.02);">
                <div style="font-size:4rem; margin-bottom:1rem;">🔮</div>
                <div style="font-family:'Orbitron',monospace; color:#1a3a5c;
                            font-size:0.9rem; letter-spacing:0.15em;">
                    FILL THE FORM &amp; PRESS ANALYZE
                </div>
                <div style="color:#2a5a7c; font-family:'Rajdhani'; margin-top:0.5rem; font-size:0.85rem;">
                    AI-powered churn risk assessment awaits
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00f5ff;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem;">
        ◈ Dataset Intelligence
    </div>
    """, unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    # Churn by Contract
    with r1c1:
        churn_contract = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        fig1 = px.bar(churn_contract, x='Contract', y='count', color='Churn',
                      barmode='group',
                      color_discrete_map={'Yes': '#ff006e', 'No': '#00f5ff'},
                      title='Churn by Contract Type')
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff')),
            xaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            yaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Churn by Internet Service
    with r1c2:
        churn_internet = df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        fig2 = px.bar(churn_internet, x='InternetService', y='count', color='Churn',
                      barmode='stack',
                      color_discrete_map={'Yes': '#ff006e', 'No': '#00f5ff'},
                      title='Churn by Internet Service')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff')),
            xaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            yaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    # Monthly Charges distribution
    with r2c1:
        fig3 = go.Figure()
        for churn_val, color_val in [('No', '#00f5ff'), ('Yes', '#ff006e')]:
            subset = df[df['Churn'] == churn_val]['MonthlyCharges']
            fig3.add_trace(go.Histogram(
                x=subset, name=f'Churn: {churn_val}',
                marker_color=color_val, opacity=0.75,
                nbinsx=20
            ))
        fig3.update_layout(
            barmode='overlay',
            title='Monthly Charges Distribution',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff')),
            xaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8'), title='Monthly Charges (₹)'),
            yaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8'), title='Count'),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Tenure vs Churn scatter
    with r2c2:
        fig4 = px.scatter(
            df, x='tenure', y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'Yes': '#ff006e', 'No': '#00f5ff'},
            title='Tenure vs Monthly Charges',
            opacity=0.65, size_max=8
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff')),
            xaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8'), title='Tenure (months)'),
            yaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8'), title='Monthly Charges (₹)'),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Payment method pie
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        pay_churn = df[df['Churn'] == 'Yes']['PaymentMethod'].value_counts()
        fig5 = go.Figure(data=[go.Pie(
            labels=pay_churn.index,
            values=pay_churn.values,
            hole=0.5,
            marker_colors=['#ff006e', '#00f5ff', '#39ff14', '#ff9f1c'],
            textfont=dict(family='Rajdhani', size=11, color='#e0f7ff'),
        )])
        fig5.update_layout(
            title='Churned Customers — Payment Methods',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff', size=10)),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig5, use_container_width=True)

    with r3c2:
        # Gender churn comparison
        gender_churn = df.groupby(['gender', 'Churn']).size().unstack(fill_value=0)
        gender_churn['churn_rate'] = gender_churn['Yes'] / (gender_churn['Yes'] + gender_churn['No']) * 100
        fig6 = go.Figure(data=[
            go.Bar(name='Retained', x=gender_churn.index,
                   y=gender_churn['No'], marker_color='#00f5ff'),
            go.Bar(name='Churned', x=gender_churn.index,
                   y=gender_churn['Yes'], marker_color='#ff006e'),
        ])
        fig6.update_layout(
            barmode='group', title='Churn by Gender',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='#e0f7ff'),
            title_font=dict(family='Orbitron', size=13, color='#00f5ff'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e0f7ff')),
            xaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            yaxis=dict(gridcolor='#1a3a5c', tickfont=dict(color='#5a8fa8')),
            height=320, margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — DATA
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00f5ff;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:1.2rem;">
        ◈ Raw Dataset
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        churn_filter = st.selectbox("Filter by Churn", ["All", "Yes", "No"])
    with fc2:
        contract_filter = st.selectbox("Filter by Contract", ["All"] + sorted(df['Contract'].unique().tolist()))
    with fc3:
        gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"])

    filtered = df.copy()
    if churn_filter != "All":
        filtered = filtered[filtered['Churn'] == churn_filter]
    if contract_filter != "All":
        filtered = filtered[filtered['Contract'] == contract_filter]
    if gender_filter != "All":
        filtered = filtered[filtered['gender'] == gender_filter]

    st.markdown(f"""
    <div style="font-family:'Rajdhani'; color:#5a8fa8; font-size:0.85rem;
                letter-spacing:0.1em; margin-bottom:0.8rem;">
        Showing <span style="color:#00f5ff;">{len(filtered)}</span> of
        <span style="color:#00f5ff;">{len(df)}</span> records
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        filtered.reset_index(drop=True),
        use_container_width=True,
        height=450
    )

    # Summary stats
    st.markdown("""
    <div style="font-family:'Orbitron',monospace; font-size:1rem; color:#00f5ff;
                letter-spacing:0.15em; text-transform:uppercase; margin: 1.5rem 0 0.8rem;">
        ◈ Statistical Summary
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        filtered[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().round(2),
        use_container_width=True
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; border-top: 1px solid #1a3a5c; margin-top:2rem;">
    <span style="font-family:'Orbitron',monospace; color:#1a3a5c; font-size:0.7rem; letter-spacing:0.2em;">
        CHURNSENSE AI ◈ POWERED BY LOGISTIC REGRESSION ◈ ANTHROPIC
    </span>
</div>
""", unsafe_allow_html=True)