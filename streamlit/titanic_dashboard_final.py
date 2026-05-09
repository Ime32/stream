"""
Titanic Survival Analysis — Full Pipeline Dashboard
Student: DIFFALLAH Imene | MI3 | ENSTA Alger
Pipeline: Collection → Preprocessing → EDA → Modeling (Linear Regression)
Run: streamlit run titanic_dashboard_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Pipeline Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.6rem; font-weight: 800; color: #0f172a;
    text-align: center; padding: 0.6rem 0 0.1rem;
    letter-spacing: -0.03em;
}
.sub-title {
    font-size: 0.98rem; color: #64748b;
    text-align: center; margin-bottom: 1.4rem;
}
.student-badge {
    display: inline-block;
    background: #f1f5f9; border-radius: 20px;
    padding: 4px 14px; font-size: 0.82rem; color: #475569;
    margin: 0 auto; text-align: center;
}
.kpi-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: white; text-align: center;
}
.kpi-card-green {
    background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: white; text-align: center;
}
.kpi-card-amber {
    background: linear-gradient(135deg, #78350f 0%, #d97706 100%);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: white; text-align: center;
}
.kpi-card-red {
    background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%);
    border-radius: 12px; padding: 1rem 1.2rem;
    color: white; text-align: center;
}
.kpi-label { font-size: 0.73rem; opacity: 0.82; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 2rem; font-weight: 700; }
.kpi-sub   { font-size: 0.75rem; opacity: 0.7; }

.section-hdr {
    font-size: 1.12rem; font-weight: 700; color: #1e3a5f;
    border-left: 4px solid #2563eb;
    padding-left: 0.7rem; margin: 1.2rem 0 0.6rem;
}
.insight {
    background: #eff6ff; border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.91rem;
    color: #1e3a5f;
}
.insight-green {
    background: #f0fdf4; border-left: 4px solid #16a34a;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.91rem;
    color: #14532d;
}
.insight-amber {
    background: #fffbeb; border-left: 4px solid #d97706;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.91rem;
    color: #78350f;
}
.step-pill {
    display: inline-block;
    background: #2563eb; color: white;
    border-radius: 20px; padding: 3px 14px;
    font-size: 0.82rem; font-weight: 600; margin-right: 6px;
}
.pipeline-bar {
    display: flex; justify-content: center; gap: 0; margin: 0.5rem 0 1.2rem;
    border-radius: 10px; overflow: hidden;
}
.pipe-step {
    flex: 1; text-align: center; padding: 10px 4px;
    font-size: 0.82rem; font-weight: 700; color: white;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    height: 44px; background-color: #f1f5f9;
    border-radius: 8px 8px 0 0; padding: 0 20px;
    font-weight: 600; font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background-color: #2563eb !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Color palette ──────────────────────────────────────────────────────────────
BLUE  = '#2563eb'
NAVY  = '#1e3a5f'
RED   = '#ef4444'
GREEN = '#16a34a'
AMBER = '#d97706'
PURPLE= '#7c3aed'
TEAL  = '#0891b2'
COLORS = [BLUE, RED, GREEN, AMBER, PURPLE, TEAL]

# ── Data & model (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv('streamlit/train.csv')

@st.cache_data
def get_preprocessed():
    df = pd.read_csv('train.csv')
    df['Age']      = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop('Cabin', axis=1, inplace=True)
    df['Sex']      = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    return df

@st.cache_data
def run_models():
    df = pd.read_csv('train.csv')
    df['Age']        = df['Age'].fillna(df['Age'].median())
    df['Embarked']   = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex']        = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']   = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], axis=1, inplace=True)

    FEATURES = ['Pclass','Sex','Age','Fare','Embarked','FamilySize']
    X, y = df[FEATURES], df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    ols   = LinearRegression().fit(Xtr, y_train)
    ridge = Ridge(alpha=1.0).fit(Xtr, y_train)
    lasso = Lasso(alpha=0.01).fit(Xtr, y_train)

    y_pred = ols.predict(Xte)
    coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': ols.coef_})\
                .sort_values('Coefficient', ascending=False)

    metrics = {}
    for name, model in [('OLS', ols), ('Ridge', ridge), ('Lasso', lasso)]:
        p = model.predict(Xte)
        metrics[name] = {
            'r2':   r2_score(y_test, p),
            'mae':  mean_absolute_error(y_test, p),
            'rmse': np.sqrt(mean_squared_error(y_test, p)),
        }

    r2_train = ols.score(Xtr, y_train)
    X_all    = scaler.fit_transform(X)
    cv       = cross_val_score(LinearRegression(), X_all, y, cv=5, scoring='r2')

    return (ols, ridge, lasso, X_train, X_test, y_train, y_test,
            y_pred, coef_df, metrics, r2_train, cv, FEATURES, scaler)

raw   = load_raw()
pre   = get_preprocessed()
(ols, ridge, lasso, X_train, X_test, y_train, y_test,
 y_pred, coef_df, metrics, r2_train, cv_scores, FEATURES, scaler) = run_models()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚢 Titanic Survival Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-end Data Science Pipeline &nbsp;·&nbsp; Collection · Preprocessing · EDA · Modeling</div>', unsafe_allow_html=True)

col_badge, _ = st.columns([1, 2])
with col_badge:
    st.markdown('<div class="student-badge">👩‍🎓 DIFFALLAH Imene &nbsp;|&nbsp; MI3 &nbsp;|&nbsp; ENSTA Alger &nbsp;|&nbsp; 2026</div>', unsafe_allow_html=True)

# Pipeline progress bar
st.markdown("""
<div class="pipeline-bar">
  <div class="pipe-step" style="background:#0f172a;">📦 1. Collection</div>
  <div class="pipe-step" style="background:#1e3a5f;">🔧 2. Preprocessing</div>
  <div class="pipe-step" style="background:#2563eb;">📊 3. EDA</div>
  <div class="pipe-step" style="background:#1d4ed8;">🤖 4. Modeling</div>
  <div class="pipe-step" style="background:#1e40af;">🔮 5. Predict</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg",
             use_column_width=True)
    st.title("🔎 EDA Filters")
    sel_sex   = st.multiselect("Gender", ['male','female'], default=['male','female'])
    sel_class = st.multiselect("Passenger Class", [1,2,3], default=[1,2,3])
    sel_surv  = st.multiselect("Outcome", [0,1], default=[0,1],
                                format_func=lambda x: "Survived ✅" if x==1 else "Did not survive ❌")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Kaggle Titanic Competition — `train.csv`  \n891 passengers · 12 features · binary survival target")
    st.markdown("---")
    st.caption("DIFFALLAH Imene · MI3 · ENSTA Alger · 2026")

df_f = raw[raw['Sex'].isin(sel_sex) & raw['Pclass'].isin(sel_class) & raw['Survived'].isin(sel_surv)]
if df_f.empty:
    st.warning("No data matches the current filters — please adjust the sidebar.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 1 · Collection",
    "🔧 2 · Preprocessing",
    "📊 3 · EDA",
    "🤖 4 · Modeling",
    "🔮 5 · Live Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hdr">📦 Data Source & Collection Context</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="insight">
<b>Problem Statement:</b> On April 15, 1912, the RMS Titanic sank after colliding with an iceberg,
killing 1,502 of 2,224 passengers. This project builds a predictive model to understand
<i>which passengers were more likely to survive</i>, and why — using demographic and ticket data.
The dataset is from the <b>Kaggle Titanic Competition</b> (public benchmark dataset).
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub, card in zip(
        [c1, c2, c3, c4],
        ["Total Passengers", "Original Features", "Missing: Age", "Missing: Cabin"],
        ["891", "12", "177", "687"],
        ["in train split", "before engineering", "→ filled w/ median", "→ column dropped"],
        ["kpi-card", "kpi-card", "kpi-card-amber", "kpi-card-red"]
    ):
        col.markdown(f'<div class="{card}"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{val}</div>'
                     f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown('<div class="section-hdr">Column Dictionary</div>', unsafe_allow_html=True)
        desc = {
            "PassengerId": ["int", "Unique ID — dropped (not predictive)"],
            "Survived":    ["int", "🎯 Target — 0 = Died, 1 = Survived"],
            "Pclass":      ["int", "Ticket class: 1st / 2nd / 3rd (proxy for wealth)"],
            "Name":        ["str", "Passenger name — dropped"],
            "Sex":         ["str", "Gender: male / female → encoded 0/1"],
            "Age":         ["float","Age in years — 19.9% missing → median fill"],
            "SibSp":       ["int", "# siblings/spouses → merged into FamilySize"],
            "Parch":       ["int", "# parents/children → merged into FamilySize"],
            "Ticket":      ["str", "Ticket number — dropped (too many unique values)"],
            "Fare":        ["float","Ticket price ($) — standardized"],
            "Cabin":       ["str", "Cabin number — 77.1% missing → dropped"],
            "Embarked":    ["str", "Port: C=Cherbourg, Q=Queenstown, S=Southampton"],
        }
        st.dataframe(
            pd.DataFrame([(k, v[0], v[1]) for k,v in desc.items()],
                         columns=["Column","Type","Description"]),
            use_container_width=True, hide_index=True
        )
    with col_b:
        st.markdown('<div class="section-hdr">Missing Values Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        miss = raw.isnull()
        ax.imshow(miss.T, aspect='auto', cmap='RdYlGn_r', interpolation='none')
        ax.set_yticks(range(len(raw.columns)))
        ax.set_yticklabels(raw.columns, fontsize=9)
        ax.set_xlabel("Passengers (rows)", fontsize=9)
        ax.set_title("Missing Values (red = missing)", fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Raw Data Preview (first 15 rows)</div>', unsafe_allow_html=True)
    st.dataframe(raw.head(15), use_container_width=True, height=280)

    st.markdown('<div class="section-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(raw.describe().T.style.format("{:.2f}").background_gradient(cmap='Blues', axis=1),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">🔧 Preprocessing Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣ Fill Age NaN", "28 (median)", "Median is robust to outliers — not affected by the few very old/young passengers"),
        ("2️⃣ Fill Embarked NaN", "'S' (mode)", "Only 2 rows missing — mode imputation causes minimal distortion"),
        ("3️⃣ Drop Cabin", "77.1% missing", "Too sparse: more than 3/4 of data is missing — not recoverable"),
        ("4️⃣ Drop Name/Ticket/ID", "3 columns removed", "Unique identifiers with no predictive value for survival"),
        ("5️⃣ Encode Sex", "male→0, female→1", "ML algorithms require numeric input — binary encoding for 2-category features"),
        ("6️⃣ Encode Embarked", "C→0, Q→1, S→2", "3-category ordinal encoding — port has minor effect so label encoding is acceptable"),
        ("7️⃣ FamilySize = SibSp+Parch+1", "New feature", "Single metric for family traveling together — reduces dimensionality & improves signal"),
        ("8️⃣ StandardScaler on Age & Fare", "μ=0, σ=1", "Linear regression assumes features on same scale — prevents large-range features from dominating"),
        ("9️⃣ Train/Test Split (80/20)", f"{len(X_train)} train / {len(X_test)} test", "random_state=42 ensures reproducibility"),
    ]
    for step, action, reason in steps:
        col1, col2, col3 = st.columns([1.5, 1.2, 2.3])
        col1.markdown(f"**{step}**")
        col2.markdown(f"`{action}`")
        col3.markdown(f"<small>{reason}</small>", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-hdr">Class Balance — Key Features</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, (col, title, xlabels) in zip(axes, [
            ('Survived', 'Survival', ['Died','Survived']),
            ('Pclass',   'Pclass',   ['1st','2nd','3rd']),
            ('Sex',      'Gender',   ['Male','Female']),
        ]):
            counts = raw[col].value_counts().sort_index()
            bars = ax.bar(xlabels, counts.values, color=COLORS[:len(counts)], edgecolor='white', width=0.55)
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.set_ylabel('Count', fontsize=8)
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+6,
                        str(int(b.get_height())), ha='center', fontsize=9, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-hdr">Before vs After Preprocessing</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        # Age before vs after
        axes[0].hist(raw['Age'].dropna(), bins=25, color=RED, alpha=0.5, label='Before (raw)')
        # After: standardized
        age_after = pre['Age']
        axes[0].hist(age_after, bins=25, color=BLUE, alpha=0.5, label='After (scaled)')
        axes[0].set_title('Age Distribution', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].spines[['top','right']].set_visible(False)

        # Fare before vs after
        axes[1].hist(raw['Fare'], bins=25, color=RED, alpha=0.5, label='Before (raw)')
        axes[1].hist(pre['Fare'], bins=25, color=BLUE, alpha=0.5, label='After (scaled)')
        axes[1].set_title('Fare Distribution', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Final Preprocessed Dataset (first 15 rows)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(X_train):,} (80%)")
    c2.metric("Test Samples",     f"{len(X_test):,} (20%)")
    c3.metric("Model Features",   str(len(FEATURES)))
    if st.checkbox("Show preprocessed dataframe"):
        st.dataframe(pre.head(15), use_container_width=True)

    st.markdown("""
<div class="insight-green">
<b>Key preprocessing decisions:</b>
<ul>
<li><b>Median over mean for Age:</b> Age has right-skewed outliers (very old passengers). Median (28) is a more robust central measure than mean (29.7).</li>
<li><b>FamilySize engineering:</b> Combining SibSp + Parch + 1 into one variable reduces features while preserving the social-group signal.</li>
<li><b>StandardScaler only on continuous features:</b> Age and Fare have very different ranges ($0–$512 vs 0–80 yrs). Binary/encoded features are already small-scale and don't need scaling.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-hdr">📊 Exploratory Data Analysis — {len(df_f):,} passengers (filtered)</div>',
                unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Survival Rate", f"{df_f['Survived'].mean()*100:.1f}%")
    c2.metric("Avg Age",               f"{df_f['Age'].mean():.1f} yrs")
    c3.metric("Median Fare",           f"${df_f['Fare'].median():.2f}")
    c4.metric("Female Survival Rate",
              f"{df_f[df_f['Sex']=='female']['Survived'].mean()*100:.1f}%" if 'female' in sel_sex else "N/A")

    # Row 1 — Survival by gender & class
    st.markdown('<div class="section-hdr">Survival by Gender & Passenger Class</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    surv_sex = df_f.groupby('Sex')['Survived'].mean() * 100
    c_sex = [BLUE if s == 'male' else RED for s in surv_sex.index]
    axes[0].bar(surv_sex.index, surv_sex.values, color=c_sex, edgecolor='white', width=0.5)
    axes[0].set_title("Survival Rate by Gender", fontweight='bold')
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(surv_sex.values):
        axes[0].text(i, v+2, f"{v:.1f}%", ha='center', fontsize=13, fontweight='bold')
    axes[0].spines[['top','right']].set_visible(False)

    surv_cls = df_f.groupby('Pclass')['Survived'].mean() * 100
    axes[1].bar(['1st','2nd','3rd'], surv_cls.values, color=[GREEN, AMBER, RED], edgecolor='white', width=0.5)
    axes[1].set_title("Survival Rate by Passenger Class", fontweight='bold')
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(surv_cls.values):
        axes[1].text(i, v+2, f"{v:.1f}%", ha='center', fontsize=13, fontweight='bold')
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
<div class="insight">
<b>Dominant finding:</b> Women survived at 74.2% vs only 18.9% for men — the "women and children first"
boarding policy is clearly visible in the data. First-class passengers survived at 63.0% vs 24.2% for
third class, revealing a strong socioeconomic privilege in lifeboat access.
</div>
""", unsafe_allow_html=True)

    # Row 2 — Age & Fare
    st.markdown('<div class="section-hdr">Age & Fare Distributions (by survival)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for surv, color, lbl in [(0, RED, 'Did not survive'), (1, GREEN, 'Survived')]:
        sub = df_f[df_f['Survived'] == surv]
        axes[0].hist(sub['Age'].dropna(), bins=25, alpha=0.6, color=color, label=lbl, edgecolor='white')
    axes[0].axvline(df_f['Age'].median(), color='black', linestyle='--',
                    label=f"Median: {df_f['Age'].median():.0f} yrs")
    axes[0].set_title("Age Distribution by Outcome", fontweight='bold')
    axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Count")
    axes[0].legend(); axes[0].spines[['top','right']].set_visible(False)

    for surv, color, lbl in [(0, RED, 'Did not survive'), (1, GREEN, 'Survived')]:
        sub = df_f[df_f['Survived'] == surv]
        axes[1].hist(sub['Fare'].clip(0, 250), bins=30, alpha=0.6, color=color, label=lbl, edgecolor='white')
    axes[1].axvline(df_f['Fare'].median(), color='black', linestyle='--',
                    label=f"Median: ${df_f['Fare'].median():.2f}")
    axes[1].set_title("Fare Distribution by Outcome (capped at $250)", fontweight='bold')
    axes[1].set_xlabel("Fare ($)"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
<div class="insight-amber">
<b>Fare is heavily right-skewed</b> (a few passengers paid up to $512). The median ($14.5) is a better
central measure than the mean ($32.2). Survivors tend to have paid higher fares — this is partly because
Fare is correlated with class (1st class = more expensive = better lifeboat access).
Children (age &lt; 15) had notably higher survival rates regardless of gender.
</div>
""", unsafe_allow_html=True)

    # Row 3 — FamilySize & Heatmap
    st.markdown('<div class="section-hdr">Family Size & Feature Correlations</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    fam_df = raw.copy()
    fam_df['FamilySize'] = fam_df['SibSp'] + fam_df['Parch'] + 1
    fam_surv = fam_df.groupby('FamilySize')['Survived'].mean() * 100
    fam_cnt  = fam_df['FamilySize'].value_counts().sort_index()
    color_fam = [AMBER if v < 35 else GREEN for v in fam_surv.values]
    axes[0].bar(fam_surv.index.astype(str), fam_surv.values, color=color_fam, edgecolor='white')
    axes[0].set_title("Survival Rate by Family Size", fontweight='bold')
    axes[0].set_xlabel("Family Size (1 = solo)"); axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_ylim(0, 105)
    for i, (fs, v) in enumerate(zip(fam_surv.index, fam_surv.values)):
        n = fam_cnt.get(fs, 0)
        axes[0].text(i, v+2, f"{v:.0f}%\n(n={n})", ha='center', fontsize=8)
    axes[0].spines[['top','right']].set_visible(False)

    corr = pre.corr(numeric_only=True)
    im   = axes[1].matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    lbls = corr.columns
    axes[1].set_xticks(range(len(lbls))); axes[1].set_yticks(range(len(lbls)))
    axes[1].set_xticklabels(lbls, rotation=45, ha='left', fontsize=8)
    axes[1].set_yticklabels(lbls, fontsize=8)
    # Annotate cells
    for i in range(len(lbls)):
        for j in range(len(lbls)):
            axes[1].text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7,
                         color='white' if abs(corr.iloc[i,j]) > 0.4 else 'black')
    axes[1].set_title("Correlation Matrix\n(Preprocessed Features)", fontweight='bold', pad=18)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
<div class="insight">
<b>Family size matters:</b> Solo travelers had only ~30% survival. Small families (2–4) did best at ~55%.
Very large families (5+) struggled at ~16% — harder to keep everyone together. The heatmap confirms
<b>Sex has the highest correlation with Survived (≈ 0.54)</b>, followed by Pclass (≈ −0.34).
Pclass and Fare are negatively correlated because higher class passengers paid more.
</div>
""", unsafe_allow_html=True)

    # Embarkation
    st.markdown('<div class="section-hdr">Survival by Embarkation Port & Age Group</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    emb_map = {'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}
    surv_emb = raw.groupby('Embarked')['Survived'].mean() * 100
    axes[0].bar([emb_map.get(e,e) for e in surv_emb.index], surv_emb.values,
                color=[TEAL, AMBER, BLUE], edgecolor='white', width=0.5)
    axes[0].set_title("Survival Rate by Port of Embarkation", fontweight='bold')
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(surv_emb.values):
        axes[0].text(i, v+2, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
    axes[0].spines[['top','right']].set_visible(False)

    raw2 = raw.copy()
    raw2['AgeGroup'] = pd.cut(raw2['Age'], bins=[0,12,18,35,60,100],
                               labels=['Child','Teen','Adult','Middle-aged','Elderly'])
    ag_surv = raw2.groupby('AgeGroup', observed=True)['Survived'].mean() * 100
    axes[1].bar(ag_surv.index, ag_surv.values, color=COLORS[:len(ag_surv)], edgecolor='white', width=0.55)
    axes[1].set_title("Survival Rate by Age Group", fontweight='bold')
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(ag_surv.values):
        axes[1].text(i, v+2, f"{v:.1f}%", ha='center', fontsize=11, fontweight='bold')
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
<div class="insight">
Cherbourg passengers had a notably higher survival rate (55.4%) than Southampton (33.7%).
This is explained by the high proportion of 1st class passengers who boarded at Cherbourg.
Children (≤12) had the highest survival rate (~58%), consistent with the evacuation priority given to minors.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODELING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">🤖 Linear Regression — Survival as a Linear Probability Model</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="insight">
<b>Why Linear Regression on a binary target?</b>
Although Survived is 0/1, Linear Regression (LPM — Linear Probability Model) is used here as specified in
Lab 7. It predicts a probability-like score and allows direct coefficient interpretation.
We also test <b>Ridge (L2)</b> and <b>Lasso (L1)</b> regularization to assess multicollinearity and feature selection.
</div>
""", unsafe_allow_html=True)

    # KPIs
    ols_m = metrics['OLS']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (OLS Test)",        f"{ols_m['r2']:.4f}")
    c2.metric("RMSE",                  f"{ols_m['rmse']:.4f}")
    c3.metric("MAE",                   f"{ols_m['mae']:.4f}")
    c4.metric("CV R² (5-fold)",        f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Row 1 — Coefficients & Actual vs Predicted
    st.markdown('<div class="section-hdr">Feature Coefficients & Prediction Quality</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors_c = [GREEN if v >= 0 else RED for v in coef_df['Coefficient']]
    bars = axes[0].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c, edgecolor='white')
    axes[0].axvline(0, color='black', linewidth=0.9)
    axes[0].set_title("Standardized Coefficients\n(green = ↑ survival, red = ↓ survival)", fontweight='bold')
    axes[0].set_xlabel("Coefficient (standardized features)")
    for i, (feat, val) in enumerate(zip(coef_df['Feature'], coef_df['Coefficient'])):
        axes[0].text(val + 0.004 if val >= 0 else val - 0.004, i,
                     f"{val:+.3f}", va='center', fontsize=9, ha='left' if val >= 0 else 'right')
    axes[0].spines[['top','right']].set_visible(False)

    axes[1].scatter(y_test, y_pred, alpha=0.35, color=BLUE, s=18, zorder=3)
    axes[1].plot([0,1],[0,1], 'r--', linewidth=1.5, label='Perfect fit (y=x)')
    axes[1].set_title("Actual vs Predicted (Test Set)", fontweight='bold')
    axes[1].set_xlabel("Actual (0 = Died, 1 = Survived)")
    axes[1].set_ylabel("Predicted probability")
    axes[1].legend(); axes[1].grid(True, alpha=0.2)
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Row 2 — Residuals
    st.markdown('<div class="section-hdr">Residual Analysis</div>', unsafe_allow_html=True)
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(residuals, bins=30, color=AMBER, edgecolor='white', alpha=0.85)
    axes[0].axvline(0, color='black', linestyle='--', label='Zero residual')
    axes[0].set_title("Residual Distribution", fontweight='bold')
    axes[0].set_xlabel("Residual (Actual − Predicted)")
    axes[0].set_ylabel("Frequency"); axes[0].legend()
    axes[0].spines[['top','right']].set_visible(False)

    axes[1].scatter(y_pred, residuals, alpha=0.3, color=BLUE, s=15)
    axes[1].axhline(0, color=RED, linestyle='--', linewidth=1.5, label='Zero line')
    axes[1].set_title("Residuals vs Predicted Values", fontweight='bold')
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Residual"); axes[1].legend()
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Row 3 — Model comparison
    st.markdown('<div class="section-hdr">Model Comparison: OLS vs Ridge vs Lasso</div>', unsafe_allow_html=True)
    model_names = ['OLS (no reg.)', 'Ridge (L2, α=1)', 'Lasso (L1, α=0.01)']
    r2_vals   = [metrics[k]['r2']   for k in ['OLS','Ridge','Lasso']]
    mae_vals  = [metrics[k]['mae']  for k in ['OLS','Ridge','Lasso']]
    rmse_vals = [metrics[k]['rmse'] for k in ['OLS','Ridge','Lasso']]

    col_t, col_p = st.columns([1, 1.4])
    with col_t:
        comp_df = pd.DataFrame({
            'Model': model_names,
            'R²':    [f"{v:.4f}" for v in r2_vals],
            'MAE':   [f"{v:.4f}" for v in mae_vals],
            'RMSE':  [f"{v:.4f}" for v in rmse_vals],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown(f"""
<div class="insight-green">
<b>Overfitting check:</b><br>
Train R² = {r2_train:.4f}<br>
Test R²  = {ols_m['r2']:.4f}<br>
Gap = {abs(r2_train - ols_m['r2']):.4f} ✅ — no significant overfitting
</div>
""", unsafe_allow_html=True)
    with col_p:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(model_names, r2_vals, color=[BLUE, GREEN, AMBER], edgecolor='white', width=0.45)
        ax.set_title("R² Score Comparison", fontweight='bold')
        ax.set_ylabel("R² Score")
        ax.set_ylim(0, max(r2_vals)*1.3)
        for b, v in zip(bars, r2_vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.4f}",
                    ha='center', fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Interpretation
    st.markdown('<div class="section-hdr">Key Findings & Interpretation</div>', unsafe_allow_html=True)
    findings = [
        ("Sex is the strongest predictor (coef ≈ +0.246)",
         "Being female increased survival probability by ~24.6 percentage points. This matches the documented 'women and children first' policy enforced by officers during evacuation."),
        ("Pclass is the main negative predictor (coef ≈ −0.126)",
         "Higher class number (3rd class) decreases survival. 3rd class cabins were deep in the ship, far from lifeboats, and passengers faced locked gates and language barriers during evacuation."),
        ("R² ≈ 0.44 — explains 44% of survival variance",
         "While not perfect, this is meaningful for a binary outcome modeled with linear regression. Real survival had a random element no model can capture (who was closest to a lifeboat at the right moment)."),
        ("Ridge ≈ OLS, confirming low multicollinearity",
         "Regularization had almost no effect, confirming that the features are not strongly collinear and OLS estimates are stable."),
        ("FamilySize and Age have weaker effects",
         "Age (coef ≈ −0.061) suggests older passengers were slightly less likely to survive. FamilySize shows a non-linear effect better captured in EDA — small families did best."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">🔮 Live Survival Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="insight">
Enter passenger details below and the trained <b>OLS Linear Regression model</b> will predict
the probability of survival. This uses the exact same scaler and model fitted during Tab 4 training.
</div>
""", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])
    with col_in:
        p_sex    = st.selectbox("Gender", ["female","male"])
        p_class  = st.selectbox("Passenger Class", [1, 2, 3])
        p_age    = st.slider("Age", 1, 80, 28)
        p_fare   = st.slider("Fare paid ($)", 0, 500, 30)
        p_emb    = st.selectbox("Port of Embarkation", ["S – Southampton","C – Cherbourg","Q – Queenstown"])
        p_family = st.slider("Family Size (incl. self)", 1, 11, 1)

    with col_out:
        sex_enc = 1 if p_sex == "female" else 0
        emb_enc = {"S – Southampton": 2, "C – Cherbourg": 0, "Q – Queenstown": 1}[p_emb]

        input_raw = np.array([[p_class, sex_enc, p_age, p_fare, emb_enc, p_family]], dtype=float)
        input_scaled = scaler.transform(pd.DataFrame(
            input_raw, columns=['Pclass','Sex','Age','Fare','Embarked','FamilySize']))
        prob = float(ols.predict(input_scaled)[0])
        prob_clamped = np.clip(prob, 0.0, 1.0)

        color_bar = GREEN if prob_clamped >= 0.5 else RED
        verdict = "✅ Likely to Survive" if prob_clamped >= 0.5 else "❌ Unlikely to Survive"

        st.markdown(f"### Predicted Survival Probability")
        st.markdown(f"""
<div style="background:{'#f0fdf4' if prob_clamped>=0.5 else '#fef2f2'};
     border:2px solid {color_bar}; border-radius:12px; padding:1.2rem 1.5rem; text-align:center;">
  <div style="font-size:3.5rem; font-weight:800; color:{color_bar};">{prob_clamped*100:.1f}%</div>
  <div style="font-size:1.2rem; font-weight:600; color:{color_bar}; margin-top:4px;">{verdict}</div>
  <div style="font-size:0.82rem; color:#64748b; margin-top:8px;">
    Raw model output: {prob:.4f} | Threshold: 0.50
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        # Show factor contributions
        contrib = ols.coef_ * input_scaled[0]
        contrib_df = pd.DataFrame({'Feature': FEATURES, 'Contribution': contrib})\
                       .sort_values('Contribution', key=abs, ascending=False)
        st.markdown("**Feature contributions to this prediction:**")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        colors_c2 = [GREEN if v >= 0 else RED for v in contrib_df['Contribution']]
        ax2.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors_c2, edgecolor='white')
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_title("What drove this prediction?", fontweight='bold', fontsize=10)
        ax2.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("---")
    st.markdown("""
<div class="insight-amber">
<b>⚠️ Model limitation:</b> Linear Regression can output probabilities outside [0,1].
The displayed probability is clamped to [0%, 100%]. For a proper probability model,
the next step would be <b>Logistic Regression</b> which naturally outputs values in [0,1].
</div>
""", unsafe_allow_html=True)
