"""
Titanic Survival Analysis — Full Pipeline Dashboard
Student: DIFFALLAH Imene | MI3 | ENSTA Alger
Covers: Collection → Preprocessing → EDA → Modeling (Lab 7)
Run: streamlit run titanic_dashboard.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / 'train.csv'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #0f172a;
        text-align: center; padding: 0.5rem 0 0.1rem 0;
    }
    .sub-title {
        font-size: 1rem; color: #64748b;
        text-align: center; margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 12px; padding: 1rem 1.2rem;
        color: white; text-align: center;
    }
    .kpi-label { font-size: 0.78rem; opacity: 0.82; }
    .kpi-value { font-size: 2rem; font-weight: 700; }
    .section-hdr {
        font-size: 1.15rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 0.7rem; margin: 1rem 0 0.6rem 0;
    }
    .insight {
        background: #eff6ff; border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem; margin: 0.5rem 0; font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 42px; background-color: #f1f5f9;
        border-radius: 8px 8px 0 0; padding: 0 18px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

BLUE   = '#2563eb'
NAVY   = '#1e3a5f'
RED    = '#ef4444'
GREEN  = '#16a34a'
AMBER  = '#d97706'
COLORS = [BLUE, RED, GREEN, AMBER, '#7c3aed', '#0891b2']

# ── Data loading & preprocessing ──────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_preprocessed():
    df = pd.read_csv(DATA_PATH)
    df['Age']        = df['Age'].fillna(df['Age'].median())
    df['Embarked']   = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop('Cabin', axis=1, inplace=True)
    df['Sex']        = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']   = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    return df

@st.cache_data
def run_models():
    df = pd.read_csv(DATA_PATH)
    df['Age']        = df['Age'].fillna(df['Age'].median())
    df['Embarked']   = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex']        = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']   = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], axis=1, inplace=True)

    FEATURES = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
    X = df[FEATURES]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

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
    X_all = scaler.fit_transform(X)
    cv    = cross_val_score(LinearRegression(), X_all, y, cv=5, scoring='r2')

    return (ols, ridge, lasso, X_train, X_test, y_train, y_test,
            y_pred, coef_df, metrics, r2_train, cv, FEATURES)

raw = load_raw()
pre = get_preprocessed()
(ols, ridge, lasso, X_train, X_test, y_train, y_test,
 y_pred, coef_df, metrics, r2_train, cv_scores, FEATURES) = run_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚢 Titanic Survival Analysis</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-end pipeline: Collection · Preprocessing · EDA · Linear Regression Modeling</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔎 Filters")
    sel_sex   = st.multiselect("Sex",   ['male', 'female'], default=['male', 'female'])
    sel_class = st.multiselect("Pclass", [1, 2, 3],         default=[1, 2, 3])
    sel_surv  = st.multiselect("Survived", [0, 1],           default=[0, 1],
                               format_func=lambda x: "Survived" if x == 1 else "Died")
    st.markdown("---")
    st.caption("DIFFALLAH Imene · MI3 · ENSTA Alger · 2026")

df_f = raw[
    raw['Sex'].isin(sel_sex) &
    raw['Pclass'].isin(sel_class) &
    raw['Survived'].isin(sel_surv)
]
if df_f.empty:
    st.warning("No data matches the filters. Please adjust the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "1 — Collection",
    "2 — Preprocessing",
    "3 — EDA",
    "4 — Modeling",
])

# ==============================================================================
# TAB 1 — COLLECTION
# ==============================================================================
with tab1:
    st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1, c2, c3, c4],
        ["Total Passengers", "Features", "Missing (Age)", "Missing (Cabin)"],
        ["891", "12", "177 (19.9%)", "687 (77.1%)"]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-hdr">Column Descriptions</div>', unsafe_allow_html=True)
    desc = {
        "PassengerId": "Unique ID — not used in modeling",
        "Survived":    "Target variable: 0 = Died, 1 = Survived",
        "Pclass":      "Ticket class: 1st (top) / 2nd / 3rd (bottom)",
        "Name":        "Passenger name — dropped (not predictive)",
        "Sex":         "Gender: male / female",
        "Age":         "Age in years (19.9% missing → filled with median)",
        "SibSp":       "# siblings/spouses aboard → merged into FamilySize",
        "Parch":       "# parents/children aboard → merged into FamilySize",
        "Ticket":      "Ticket number — dropped",
        "Fare":        "Ticket price paid",
        "Cabin":       "Cabin number (77.1% missing → dropped)",
        "Embarked":    "Port: C=Cherbourg, Q=Queenstown, S=Southampton",
    }
    st.dataframe(pd.DataFrame(desc.items(), columns=["Column", "Description"]),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-hdr">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw.head(20), use_container_width=True, height=280)

    st.markdown('<div class="section-hdr">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(raw.describe().style.format("{:.2f}"), use_container_width=True)

# ==============================================================================
# TAB 2 — PREPROCESSING
# ==============================================================================
with tab2:
    st.markdown('<div class="section-hdr">Preprocessing Steps Applied</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Missing Values")
        mv = raw.isnull().sum().reset_index()
        mv.columns = ['Feature', 'Missing']
        mv['% Missing'] = (mv['Missing'] / len(raw) * 100).round(1)
        mv['Action'] = mv.apply(lambda r: (
            'Fill with median' if r['Feature'] == 'Age' else
            'Fill with mode'   if r['Feature'] == 'Embarked' else
            'Drop column'      if r['Feature'] == 'Cabin' else
            'No action needed'), axis=1)
        st.dataframe(mv, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Feature Engineering")
        st.markdown("""
| Step | Action | Reason |
|---|---|---|
| `Cabin` dropped | 77% missing | Too sparse to be useful |
| `Name`, `Ticket`, `PassengerId` dropped | No predictive value | Unique per passenger |
| `Sex` encoded | male→0, female→1 | ML requires numbers |
| `Embarked` encoded | C→0, Q→1, S→2 | ML requires numbers |
| `FamilySize` created | SibSp + Parch + 1 | Single family metric |
| `Age`, `Fare` scaled | StandardScaler | Needed for regression |
        """)

    st.markdown("#### Class Balance — Key Categorical Features")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, [
        ('Survived', 'Survival (0=Died, 1=Survived)'),
        ('Pclass',   'Passenger Class'),
        ('Sex',      'Gender'),
    ]):
        counts = raw[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values,
               color=COLORS[:len(counts)], edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 8, str(v), ha='center', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### Train / Test Split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training samples", f"{len(X_train):,} (80%)")
    c2.metric("Test samples",     f"{len(X_test):,} (20%)")
    c3.metric("Features used",    str(len(FEATURES)))

    st.markdown('<div class="insight"><b>Summary:</b> Age filled with median (28.0 yrs). Embarked filled with mode (S). Cabin dropped. Sex & Embarked label-encoded. FamilySize engineered. Age & Fare standardized. Split 80/20, random_state=42.</div>',
                unsafe_allow_html=True)

    if st.checkbox('Show preprocessed dataframe'):
        st.dataframe(pre.head(20), use_container_width=True)

# ==============================================================================
# TAB 3 — EDA
# ==============================================================================
with tab3:
    st.markdown(f'<div class="section-hdr">Exploratory Data Analysis ({len(df_f):,} passengers filtered)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Survival Rate", f"{df_f['Survived'].mean()*100:.1f}%")
    c2.metric("Avg Age",               f"{df_f['Age'].mean():.1f} yrs")
    c3.metric("Avg Fare",              f"${df_f['Fare'].mean():.2f}")
    c4.metric("Female Survival",       f"{df_f[df_f['Sex']=='female']['Survived'].mean()*100:.1f}%")

    st.markdown("#### Survival by Gender & Passenger Class")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    surv_sex = df_f.groupby('Sex')['Survived'].mean() * 100
    axes[0].bar(surv_sex.index, surv_sex.values,
                color=[BLUE if s == 'male' else RED for s in surv_sex.index],
                edgecolor='white', width=0.5)
    axes[0].set_title("Survival Rate by Gender", fontweight='bold')
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(surv_sex.values):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    surv_class = df_f.groupby('Pclass')['Survived'].mean() * 100
    axes[1].bar(surv_class.index.astype(str), surv_class.values,
                color=COLORS[:3], edgecolor='white', width=0.5)
    axes[1].set_title("Survival Rate by Passenger Class", fontweight='bold')
    axes[1].set_xlabel("Class (1=First, 2=Second, 3=Third)")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(surv_class.values):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="insight"><b>Key finding:</b> Women had a 74.2% survival rate vs 18.9% for men — nearly 4x higher. 1st class passengers survived at 63.0% vs 24.2% for 3rd class. Gender was the dominant survival factor.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Age & Fare Distributions")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(df_f['Age'].dropna(), bins=30, color=BLUE,
                 edgecolor='white', alpha=0.85)
    axes[0].axvline(df_f['Age'].median(), color=RED, linestyle='--',
                    label=f"Median: {df_f['Age'].median():.0f} yrs")
    axes[0].set_title("Age Distribution", fontweight='bold')
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].hist(df_f['Fare'], bins=40, color=AMBER,
                 edgecolor='white', alpha=0.85)
    axes[1].axvline(df_f['Fare'].median(), color=RED, linestyle='--',
                    label=f"Median: ${df_f['Fare'].median():.2f}")
    axes[1].set_title("Fare Distribution (right-skewed)", fontweight='bold')
    axes[1].set_xlabel("Fare ($)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="insight"><b>Fare is heavily right-skewed</b> — a few passengers paid extremely high fares (up to $512). The median ($14.5) is a more representative central measure than the mean ($32.2). This is why we used the median to describe fare in our analysis.</div>',
                unsafe_allow_html=True)

    st.markdown("#### Family Size & Survival + Correlation Heatmap")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    fam_df = raw.copy()
    fam_df['FamilySize'] = fam_df['SibSp'] + fam_df['Parch'] + 1
    fam_surv = fam_df.groupby('FamilySize')['Survived'].mean() * 100
    axes[0].bar(fam_surv.index.astype(str), fam_surv.values,
                color=BLUE, edgecolor='white')
    axes[0].set_title("Survival Rate by Family Size", fontweight='bold')
    axes[0].set_xlabel("Family Size (1 = solo traveler)")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].spines[['top', 'right']].set_visible(False)

    corr = pre.corr(numeric_only=True)
    im = axes[1].matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    labels = corr.columns
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='left', fontsize=8)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_title("Correlation Heatmap\n(Preprocessed Features)", fontweight='bold', pad=18)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="insight"><b>Family size matters:</b> solo travelers had only 30.4% survival. Small families (2-4 members) had the highest survival at ~55%. Very large families (5+) had the lowest at 16%. The heatmap shows Pclass and Fare are negatively correlated, and Sex has the strongest correlation with Survived.</div>',
                unsafe_allow_html=True)

# ==============================================================================
# TAB 4 — MODELING
# ==============================================================================
with tab4:
    st.markdown('<div class="section-hdr">Linear Regression — Predicting Survival (Linear Probability Model)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    ols_m = metrics['OLS']
    c1.metric("R² Score (OLS)",  f"{ols_m['r2']:.4f}")
    c2.metric("RMSE",            f"{ols_m['rmse']:.4f}")
    c3.metric("MAE",             f"{ols_m['mae']:.4f}")
    c4.metric("CV R² (5-fold)",  f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    st.markdown("#### Standardized Coefficients — Feature Impact on Survival")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors_c = [GREEN if v >= 0 else RED for v in coef_df['Coefficient']]
    axes[0].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title("Standardized Coefficients\n(green = increases survival, red = decreases)",
                      fontweight='bold')
    axes[0].set_xlabel("Coefficient (standardized)")
    axes[0].spines[['top', 'right']].set_visible(False)
    for i, (feat, val) in enumerate(zip(coef_df['Feature'], coef_df['Coefficient'])):
        axes[0].text(val + 0.003 if val >= 0 else val - 0.003, i,
                     f"{val:+.3f}", va='center', fontsize=9,
                     ha='left' if val >= 0 else 'right')

    axes[1].scatter(y_test, y_pred, alpha=0.35, color=BLUE, s=18)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect fit')
    axes[1].set_title("Actual vs Predicted Survival Probability", fontweight='bold')
    axes[1].set_xlabel("Actual (0 = Died, 1 = Survived)")
    axes[1].set_ylabel("Predicted probability")
    axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### Residual Analysis")
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(residuals, bins=30, color=AMBER, edgecolor='white', alpha=0.85)
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set_title("Residual Distribution", fontweight='bold')
    axes[0].set_xlabel("Residual (Actual − Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].scatter(y_pred, residuals, alpha=0.3, color=BLUE, s=15)
    axes[1].axhline(0, color=RED, linestyle='--', linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted Values", fontweight='bold')
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Residual")
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("#### Model Comparison — OLS vs Ridge vs Lasso")
    model_names = ['OLS (Linear)', 'Ridge (L2, α=1)', 'Lasso (L1, α=0.01)']
    r2_vals  = [metrics[k]['r2']   for k in ['OLS', 'Ridge', 'Lasso']]
    mae_vals = [metrics[k]['mae']  for k in ['OLS', 'Ridge', 'Lasso']]
    rmse_vals= [metrics[k]['rmse'] for k in ['OLS', 'Ridge', 'Lasso']]

    comp_df = pd.DataFrame({
        'Model': model_names,
        'R²':    [f"{v:.4f}" for v in r2_vals],
        'MAE':   [f"{v:.4f}" for v in mae_vals],
        'RMSE':  [f"{v:.4f}" for v in rmse_vals],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(model_names, r2_vals, color=[BLUE, GREEN, AMBER], edgecolor='white', width=0.45)
    ax.set_title("R² Score by Model", fontweight='bold')
    ax.set_ylabel("R² Score")
    ax.set_ylim(0, max(r2_vals) * 1.25)
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.003, f"{v:.4f}", ha='center', fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f"""
<div class="insight">
<b>Overfitting check:</b> Train R² = {r2_train:.4f} vs Test R² = {ols_m['r2']:.4f} 
(difference = {abs(r2_train - ols_m['r2']):.4f}) — the model generalizes well, no significant overfitting.
</div>""", unsafe_allow_html=True)

    st.markdown("#### Key Findings & Interpretation")
    findings = [
        ("Sex is by far the strongest predictor (coef ≈ +0.246)",
         "Being female increased survival probability by ~24.6 percentage points (after standardization). This aligns with the 'women and children first' evacuation policy."),
        ("Pclass is the strongest negative predictor (coef ≈ -0.126)",
         "Higher class number (3rd class) strongly decreases survival probability. 1st class passengers had privileged access to lifeboats and were housed on higher decks."),
        ("R² ≈ 0.44 — the model explains 44% of variance",
         "While not perfect, this is meaningful for a binary outcome modeled with linear regression. The model captures real social patterns from the Titanic disaster."),
        ("Ridge ≈ OLS, Lasso slightly lower",
         "Ridge regression produced nearly identical results to OLS, confirming that multicollinearity is not severe here. Lasso slightly reduced performance by shrinking small coefficients."),
        ("Age and Fare have weak negative/positive effects",
         "Older passengers had slightly lower survival (Age coef ≈ -0.061). Higher fare correlated with slightly better survival (coef ≈ +0.019), likely because it's a proxy for wealth and cabin location."),
    ]
    for title, body in findings:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{body}</div>',
                    unsafe_allow_html=True)
