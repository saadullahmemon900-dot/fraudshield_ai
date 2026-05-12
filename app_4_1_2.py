import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    .stApp { background: #080c14; font-family: 'DM Sans', sans-serif; color: #e2e8f0; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%); border-right: 1px solid #1a2332; }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }
    .hero-title {
        font-family: 'Syne', sans-serif; font-size: 3.2rem; font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.1; margin-bottom: 0.2rem;
    }
    .hero-sub { font-size: 1.05rem; color: #64748b; font-weight: 300; margin-bottom: 2rem; letter-spacing: 0.02em; }
    .metric-card {
        background: linear-gradient(135deg, #0d1f35 0%, #111827 100%);
        border: 1px solid #1e3a5f; border-radius: 16px; padding: 24px 28px;
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }
    .metric-label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; margin-bottom: 8px; }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 2.1rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
    .metric-delta { font-size: 0.8rem; color: #38bdf8; margin-top: 6px; }
    .result-fraud {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f); border: 1px solid #7f1d1d;
        border-left: 4px solid #ef4444; border-radius: 16px; padding: 28px 32px; text-align: center;
    }
    .result-legit {
        background: linear-gradient(135deg, #0a1a0d, #0f2d14); border: 1px solid #14532d;
        border-left: 4px solid #22c55e; border-radius: 16px; padding: 28px 32px; text-align: center;
    }
    .result-title { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .section-header {
        font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
        border-bottom: 1px solid #1e2d40; padding-bottom: 10px; margin-bottom: 20px;
    }
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important; font-size: 0.85rem !important;
        font-weight: 500 !important; letter-spacing: 0.04em !important;
    }
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background-color: #0d1117 !important; border: 1px solid #1e3a5f !important;
        color: #e2e8f0 !important; border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 14px 32px !important; font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important; font-weight: 700 !important; width: 100% !important;
        letter-spacing: 0.04em !important; cursor: pointer !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #6366f1) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; border-bottom: 1px solid #1e2d40; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0d1117; border-radius: 8px 8px 0 0; color: #64748b !important;
        border: 1px solid #1e2d40; border-bottom: none; font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0d1f35, #111827) !important; color: #38bdf8 !important; }
    hr { border-color: #1e2d40 !important; margin: 1.5rem 0; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE — SABSE PEHLE ─────────────────────────────────────────────
# YE LINES SABI CHEEZ SE PEHLE HAIN — PAGE SWITCH PE KABHI RESET NAHI HONGI
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'fraud_prob' not in st.session_state:
    st.session_state.fraud_prob = 0.0
if 'legit_prob' not in st.session_state:
    st.session_state.legit_prob = 0.0
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = {}


# ─── MODEL ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    try:
        df = pd.read_parquet('credit_card_frauds.parquet')
    except FileNotFoundError:
        st.warning("credit_card_frauds.csv not found. Using sample data.")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'trans_date_trans_time': pd.date_range('2019-01-01', periods=n, freq='1min').astype(str),
            'category': np.random.choice(['grocery_pos','entertainment','shopping_pos','misc_pos','gas_transport'], n),
            'amt': np.random.exponential(80, n),
            'state': np.random.choice(['CA','TX','NY','FL','AZ','WA','ID','NM'], n),
            'lat': np.random.uniform(25, 50, n),
            'long': np.random.uniform(-125, -70, n),
            'merch_lat': np.random.uniform(25, 50, n),
            'merch_long': np.random.uniform(-125, -70, n),
            'dob': pd.date_range('1950-01-01', periods=n, freq='7D').astype(str),
            'is_fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])
        })

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['hours'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.dayofweek
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2-lat1, lon2-lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

    fraud = df[df['is_fraud']==1]
    non_fraud = df[df['is_fraud']==0]
    non_fraud_sample = non_fraud.sample(n=min(2218, len(non_fraud)), random_state=42)
    df = pd.concat([fraud, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[['category','amt','state','age','distance','hours','day']]
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[('enc', OneHotEncoder(handle_unknown='ignore'), ['category','state'])],
        remainder='passthrough'
    )
    pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=5, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test)) * 100
    return pipe, df, acc, X_test, y_test, pipe.predict(X_test)


with st.spinner("Loading AI Model..."):
    pipe, df, accuracy, X_test, y_test, y_pred = load_and_train()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 24px 0; border-bottom:1px solid #1e2d40; margin-bottom:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#38bdf8;">🛡️ FraudShield</div>
        <div style="font-size:0.72rem;color:#334155;letter-spacing:0.1em;text-transform:uppercase;">AI Detection System</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("Go to",
        ["🔍 Fraud Predictor", "📊 Analytics Dashboard", "ℹ️ About"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#334155;line-height:1.6;">
        <b style="color:#475569;">Model Info</b><br>
        Algorithm: Random Forest<br>Features: 7 inputs<br>Training Data: 3.2k records
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FRAUD PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if "Predictor" in page:

    st.markdown('<div class="hero-title">FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time credit card fraud detection powered by Random Forest ML</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">{accuracy:.1f}%</div><div class="metric-delta">↑ Validated on test set</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Fraud Cases</div><div class="metric-value">{int(df["is_fraud"].sum()):,}</div><div class="metric-delta">In training dataset</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Fraud Rate</div><div class="metric-value">{df["is_fraud"].mean()*100:.2f}%</div><div class="metric-delta">Of all transactions</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Records</div><div class="metric-value">{len(df):,}</div><div class="metric-delta">Training dataset size</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.markdown('<div class="section-header">🔎 Transaction Details</div>', unsafe_allow_html=True)

        r1a, r1b = st.columns(2)
        with r1a:
            category = st.selectbox("Transaction Category", [
                'grocery_pos','entertainment','shopping_pos','misc_pos','gas_transport',
                'shopping_net','food_dining','health_fitness','personal_care',
                'home','kids_pets','travel','misc_net'], index=0)
        with r1b:
            state = st.selectbox("Customer State", [
                'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID',
                'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
                'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
                'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'], index=4)

        r2a, r2b = st.columns(2)
        with r2a:
            amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=30000.0, value=150.0, step=10.0, format="%.2f")
        with r2b:
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, step=1)

        r3a, r3b = st.columns(2)
        with r3a:
            distance = st.number_input("Distance to Merchant (km)", min_value=0.0, max_value=500.0, value=5.0, step=1.0)
        with r3b:
            hours = st.slider("Transaction Hour", 0, 23, 14, help="0=midnight, 12=noon, 23=11pm")

        day = st.select_slider("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            value="Wednesday")
        day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ ANALYZE TRANSACTION", use_container_width=True)

    with right_col:
        st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if predict_clicked:
            sample = pd.DataFrame([{
                'category': category, 'amt': amt, 'state': state,
                'age': age, 'distance': distance, 'hours': hours, 'day': day_num
            }])
            with st.spinner("Analyzing..."):
                time.sleep(0.6)

            prediction = pipe.predict(sample)[0]
            proba      = pipe.predict_proba(sample)[0]

            # Sab kuch session_state mein save karo
            st.session_state.prediction_done = True
            st.session_state.prediction      = int(prediction)
            st.session_state.fraud_prob      = float(proba[1] * 100)
            st.session_state.legit_prob      = float(proba[0] * 100)
            st.session_state.last_summary    = {
                "💰 Amount":   f"${amt:,.2f}",
                "🏪 Category": category.replace('_',' ').title(),
                "📍 State":    state,
                "👤 Age":      f"{age} years",
                "📏 Distance": f"{distance:.1f} km",
                "🕐 Hour":     f"{hours:02d}:00",
                "📅 Day":      day
            }

        # Result — session_state se read hota hai, page switch ke baad bhi rahega
        if st.session_state.prediction_done:
            if st.session_state.prediction == 1:
                st.markdown("""
                <div class="result-fraud">
                    <div style="font-size:3rem;margin-bottom:8px;">🚨</div>
                    <div class="result-title" style="color:#ef4444;">FRAUD DETECTED</div>
                    <div style="color:#fca5a5;font-size:0.9rem;margin-top:8px;">This transaction has been flagged as suspicious</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-legit">
                    <div style="font-size:3rem;margin-bottom:8px;">✅</div>
                    <div class="result-title" style="color:#22c55e;">LEGITIMATE</div>
                    <div style="color:#86efac;font-size:0.9rem;margin-top:8px;">Transaction appears safe and genuine</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Confidence Breakdown**")
            st.markdown(f"🔴 Fraud Risk: {st.session_state.fraud_prob:.1f}%")
            st.progress(int(st.session_state.fraud_prob))
            st.markdown(f"🟢 Legitimate: {st.session_state.legit_prob:.1f}%")
            st.progress(int(st.session_state.legit_prob))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style="background:#0d1117;border:1px solid #1e2d40;border-radius:12px;padding:16px;">
                <div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Transaction Summary</div>""",
                unsafe_allow_html=True)
            for k, v in st.session_state.last_summary.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2d40;'>"
                    f"<span style='color:#64748b;font-size:0.82rem;'>{k}</span>"
                    f"<span style='color:#e2e8f0;font-size:0.82rem;font-weight:500;'>{v}</span></div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#0d1117;border:2px dashed #1e3a5f;border-radius:16px;
                        padding:60px 30px;text-align:center;margin-top:10px;">
                <div style="font-size:3rem;margin-bottom:16px;opacity:0.4;">🛡️</div>
                <div style="color:#334155;font-size:0.95rem;">
                    Fill in the transaction details<br>and click <b style="color:#38bdf8;">Analyze Transaction</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:

    st.markdown('<div class="hero-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep insights into fraud patterns from the dataset</div>', unsafe_allow_html=True)

    df_viz = df.copy()
    df_viz['trans_date_trans_time'] = pd.to_datetime(df_viz['trans_date_trans_time'])
    df_viz['hour']     = df_viz['trans_date_trans_time'].dt.hour
    df_viz['day_name'] = df_viz['trans_date_trans_time'].dt.day_name()
    df_viz['label']    = df_viz['is_fraud'].map({0:'Legitimate', 1:'Fraud'})

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "⏰ Time Patterns", "💰 Amount Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fraud_counts = df_viz['label'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=fraud_counts.index, values=fraud_counts.values, hole=0.55,
                marker=dict(colors=['#1d4ed8','#ef4444']),
                textfont=dict(family='DM Sans', size=13, color='white')
            )])
            fig_pie.update_layout(
                title=dict(text=" Fraud Distribution", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                font=dict(color='#94a3b8'), legend=dict(font=dict(color='#94a3b8')),
                margin=dict(t=50,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            cat_stats = df_viz.groupby('category')['is_fraud'].mean().sort_values(ascending=False)*100
            cat_stats.index = cat_stats.index.str.replace('_',' ').str.title()
            fig_cat = px.bar(x=cat_stats.values, y=cat_stats.index, orientation='h',
                color=cat_stats.values, color_continuous_scale=['#1e3a5f','#ef4444'],
                labels={'x':'Fraud Rate (%)','y':''})
            fig_cat.update_layout(
                title=dict(text="Fraud Rate by Category (%)", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#94a3b8'),
                coloraxis_showscale=False, margin=dict(t=50,b=10))
            fig_cat.update_traces(marker_line_width=0)
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            hourly = df_viz[df_viz['is_fraud']==1].groupby('hour').size().reset_index(name='count')
            fig_hr = px.area(hourly, x='hour', y='count', color_discrete_sequence=['#38bdf8'],
                labels={'hour':'Hour of Day','count':'Fraud Count'})
            fig_hr.update_layout(
                title=dict(text="Fraud by Hour of Day", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#94a3b8'),
                xaxis=dict(gridcolor='#1e2d40', tickvals=list(range(0,24))),
                yaxis=dict(gridcolor='#1e2d40'), margin=dict(t=50,b=10))
            fig_hr.update_traces(fillcolor='rgba(56,189,248,0.15)', line_color='#38bdf8')
            st.plotly_chart(fig_hr, use_container_width=True)
        with col2:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            day_fraud = df_viz[df_viz['is_fraud']==1]['day_name'].value_counts().reindex(day_order, fill_value=0).reset_index()
            day_fraud.columns = ['Day','Count']
            fig_day = px.bar(day_fraud, x='Day', y='Count', color='Count',
                color_continuous_scale=['#1e3a5f','#ef4444'])
            fig_day.update_layout(
                title=dict(text="Fraud by Day of Week", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#94a3b8'),
                coloraxis_showscale=False, xaxis=dict(gridcolor='#1e2d40'),
                yaxis=dict(gridcolor='#1e2d40'), margin=dict(t=50,b=10))
            st.plotly_chart(fig_day, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df_viz[df_viz['amt']<1000], x='amt', color='label', nbins=60,
                color_discrete_map={'Legitimate':'#1d4ed8','Fraud':'#ef4444'},
                barmode='overlay', opacity=0.7, labels={'amt':'Transaction Amount ($)'})
            fig_hist.update_layout(
                title=dict(text="Amount Distribution: Fraud vs Legit", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#94a3b8'),
                xaxis=dict(gridcolor='#1e2d40'), yaxis=dict(gridcolor='#1e2d40'),
                legend=dict(font=dict(color='#94a3b8')), margin=dict(t=50,b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df_viz[df_viz['amt']<2000], x='label', y='amt', color='label',
                color_discrete_map={'Legitimate':'#1d4ed8','Fraud':'#ef4444'},
                labels={'amt':'Amount ($)','label':''})
            fig_box.update_layout(
                title=dict(text="Amount Range Comparison", font=dict(color='#94a3b8',size=15)),
                paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font=dict(color='#94a3b8'),
                xaxis=dict(gridcolor='#1e2d40'), yaxis=dict(gridcolor='#1e2d40'),
                showlegend=False, margin=dict(t=50,b=10))
            st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown('<div class="hero-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Credit Card Fraud Detection using Machine Learning</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1e3a5f;border-radius:16px;padding:28px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#38bdf8;margin-bottom:16px;">🎯 Project Goal</div>
            <p style="color:#94a3b8;line-height:1.8;font-size:0.9rem;">
                This project detects whether a credit card transaction is <b style="color:#ef4444;">fraudulent</b>
                or <b style="color:#22c55e;">legitimate</b> using a machine learning model trained on real-world financial data.
            </p>
            <p style="color:#94a3b8;line-height:1.8;font-size:0.9rem;">
                The model was built as part of an ITDS (Introduction to Data Science) university project.
            </p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1e3a5f;border-radius:16px;padding:28px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#818cf8;margin-bottom:16px;">🤖 Model Details</div>
            <div style="color:#64748b;font-size:0.8rem;line-height:2;">
                <b style="color:#94a3b8;">Algorithm:</b> Random Forest Classifier<br>
                <b style="color:#94a3b8;">Trees:</b> 300 estimators<br>
                <b style="color:#94a3b8;">Max Depth:</b> 6<br>
                <b style="color:#94a3b8;">Accuracy:</b> 87.50%<br>
                <b style="color:#94a3b8;">Imbalance:</b> Undersampling used
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1e3a5f;border-radius:16px;padding:28px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#f472b6;margin-bottom:16px;">⚙️ Features Used</div>
            <div style="color:#64748b;font-size:0.85rem;line-height:2.2;">
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">category</b> — Type of purchase<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">amt</b> — Transaction dollar amount<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">state</b> — US state of customer<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">age</b> — Customer age from DOB<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">distance</b> — Customer to Merchant km<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">hours</b> — Hour of day (0-23)<br>
                <span style="color:#38bdf8;">●</span> <b style="color:#94a3b8;">day</b> — Day of week (0=Mon)
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1117;border:1px solid #1e3a5f;border-radius:16px;padding:28px;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#fbbf24;margin-bottom:16px;">🚀 Tech Stack</div>
            <div style="color:#64748b;font-size:0.85rem;line-height:2.2;">
                <b style="color:#94a3b8;">Python</b> — Core language<br>
                <b style="color:#94a3b8;">Streamlit</b> — Web app framework<br>
                <b style="color:#94a3b8;">Scikit-learn</b> — ML model<br>
                <b style="color:#94a3b8;">Pandas / NumPy</b> — Data processing<br>
                <b style="color:#94a3b8;">Plotly</b> — Interactive charts
            </div>
        </div>""", unsafe_allow_html=True)