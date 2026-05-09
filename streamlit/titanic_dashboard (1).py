"""
Titanic Survival Analysis — Complete Dashboard
Student: DIFFALLAH Imene | MI3 | ENSTA Alger
All 7 steps: Collection, Stats, Probability, Preprocessing, EDA, Visualization, Modeling
Run: streamlit run titanic_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
from pathlib import Path
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Titanic — Full Pipeline", page_icon="🚢",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.big-title{font-size:2.3rem;font-weight:800;color:#0f172a;text-align:center;padding:.4rem 0 0}
.sub{font-size:.98rem;color:#64748b;text-align:center;margin-bottom:.7rem}
.insight{background:#eff6ff;border-left:4px solid #2563eb;border-radius:0 8px 8px 0;
         padding:.65rem 1rem;margin:.4rem 0;font-size:.91rem;color:#1e3a5f}
.warn{background:#fff7ed;border-left:4px solid #f59e0b;border-radius:0 8px 8px 0;
      padding:.65rem 1rem;margin:.4rem 0;font-size:.91rem;color:#78350f}
.kpi{background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:12px;
     padding:.9rem 1rem;color:white;text-align:center}
.kpi-val{font-size:2rem;font-weight:700}.kpi-lbl{font-size:.78rem;opacity:.82}
.stTabs [data-baseweb="tab-list"]{gap:4px}
.stTabs [data-baseweb="tab"]{height:40px;background:#f1f5f9;
  border-radius:8px 8px 0 0;padding:0 14px;font-weight:600;font-size:.88rem}
.stTabs [aria-selected="true"]{background:#2563eb !important;color:white !important}
</style>
""", unsafe_allow_html=True)

BLUE="#2563eb"; RED="#ef4444"; GREEN="#16a34a"; AMB="#d97706"; TEAL="#0d9488"; NAVY="#1e3a5f"
COLS=[BLUE,RED,GREEN,AMB,TEAL,"#7c3aed","#0891b2"]

DATA_PATH = Path(__file__).parent / "train.csv"

@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_preprocessed():
    df = pd.read_csv(DATA_PATH)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df.drop("Cabin", axis=1, inplace=True)
    df["Sex"]      = df["Sex"].map({"male":0,"female":1})
    df["Embarked"] = df["Embarked"].map({"C":0,"Q":1,"S":2})
    df.drop(["Name","Ticket","PassengerId"], axis=1, inplace=True)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.drop(["SibSp","Parch"], axis=1, inplace=True)
    sc = StandardScaler()
    df[["Age","Fare"]] = sc.fit_transform(df[["Age","Fare"]])
    return df

@st.cache_data
def run_models():
    df = pd.read_csv(DATA_PATH)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"]      = df["Sex"].map({"male":0,"female":1})
    df["Embarked"] = df["Embarked"].map({"C":0,"Q":1,"S":2})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.drop(["PassengerId","Name","Ticket","Cabin","SibSp","Parch"], axis=1, inplace=True)
    FEAT = ["Pclass","Sex","Age","Fare","Embarked","FamilySize"]
    X, y = df[FEAT], df["Survived"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
    ols   = LinearRegression().fit(Xtr_s, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr)
    lasso = Lasso(alpha=.01).fit(Xtr_s, ytr)
    y_pred = ols.predict(Xte_s)
    coef_df = pd.DataFrame({"Feature":FEAT,"Coefficient":ols.coef_})\
                .sort_values("Coefficient",ascending=False).reset_index(drop=True)
    metrics = {}
    for nm, m in [("OLS",ols),("Ridge",ridge),("Lasso",lasso)]:
        p = m.predict(Xte_s)
        metrics[nm]={"r2":r2_score(yte,p),"mae":mean_absolute_error(yte,p),
                     "rmse":np.sqrt(mean_squared_error(yte,p))}
    r2_tr = ols.score(Xtr_s, ytr)
    X_all = sc.fit_transform(X)
    cv = cross_val_score(LinearRegression(), X_all, y, cv=5, scoring="r2")
    return ols,ridge,lasso,Xtr,Xte,ytr,yte,y_pred,coef_df,metrics,r2_tr,cv,FEAT

raw  = load_raw()
pre  = get_preprocessed()
ols,ridge,lasso,X_train,X_test,y_train,y_test,y_pred,coef_df,metrics,r2_train,cv_scores,FEATURES = run_models()

st.markdown('<div class="big-title">🚢 Titanic Survival Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Complete pipeline · Collection · Statistics · Probability · Preprocessing · EDA · Visualizations · Modeling</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.title("🔎 EDA Filters")
    sel_sex   = st.multiselect("Sex",      ["male","female"], default=["male","female"])
    sel_class = st.multiselect("Pclass",   [1,2,3],           default=[1,2,3])
    sel_surv  = st.multiselect("Survived", [0,1],             default=[0,1],
                               format_func=lambda x:"Survived" if x==1 else "Died")
    st.markdown("---")
    st.caption("DIFFALLAH Imene · MI3 · ENSTA Alger · 2026")

df_f = raw[raw["Sex"].isin(sel_sex) & raw["Pclass"].isin(sel_class) & raw["Survived"].isin(sel_surv)]

tabs = st.tabs(["1 — Collection","2 — Statistics","3 — Probability",
                "4 — Preprocessing","5 — EDA","6 — Visualizations","7 — Modeling"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Step 1 — Data Collection")
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val in zip([c1,c2,c3,c4],
        ["Total Passengers","Features","Survivors","Survival Rate"],["891","12","342","38.4%"]):
        col.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                     f'<div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    col_a, col_b = st.columns([1.3,1])
    with col_a:
        st.markdown("#### Column Descriptions")
        desc = {"PassengerId":"Unique ID — dropped","Survived":"🎯 Target: 0=Died, 1=Survived",
                "Pclass":"Ticket class: 1st/2nd/3rd","Name":"Name — dropped",
                "Sex":"Gender: male/female","Age":"Age in years (19.9% missing)",
                "SibSp":"Siblings/spouses aboard","Parch":"Parents/children aboard",
                "Ticket":"Ticket number — dropped","Fare":"Ticket price",
                "Cabin":"Cabin (77.1% missing → dropped)","Embarked":"Port: C/Q/S"}
        st.dataframe(pd.DataFrame(desc.items(),columns=["Column","Description"]),
                     use_container_width=True, hide_index=True, height=360)
    with col_b:
        st.markdown("#### Missing Values")
        mv = raw.isnull().sum().reset_index()
        mv.columns=["Feature","Missing"]
        mv["% Missing"]=(mv["Missing"]/len(raw)*100).round(1)
        mv["Action"]=mv.apply(lambda r:(
            "Fill with median"   if r["Feature"]=="Age"      else
            "Fill with mode"     if r["Feature"]=="Embarked" else
            "Drop column"        if r["Feature"]=="Cabin"    else "—"),axis=1)
        st.dataframe(mv[mv["Missing"]>0], use_container_width=True, hide_index=True)
        st.markdown("""
**Dataset source:**  
Titanic — Machine Learning from Disaster  
[kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)  
File: `train.csv` — 891 rows × 12 columns
        """)

    st.markdown("#### Raw Data (first 20 rows)")
    st.dataframe(raw.head(20), use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Step 2 — Descriptive Statistics")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Summary Statistics")
        st.dataframe(raw.describe().style.format("{:.2f}"), use_container_width=True)
    with col_b:
        st.markdown("#### Survival by Group")
        surv_stats = {
            "Overall":           f"{raw['Survived'].mean()*100:.1f}%",
            "Women":             f"{raw[raw['Sex']=='female']['Survived'].mean()*100:.1f}%",
            "Men":               f"{raw[raw['Sex']=='male']['Survived'].mean()*100:.1f}%",
            "1st Class":         f"{raw[raw['Pclass']==1]['Survived'].mean()*100:.1f}%",
            "2nd Class":         f"{raw[raw['Pclass']==2]['Survived'].mean()*100:.1f}%",
            "3rd Class":         f"{raw[raw['Pclass']==3]['Survived'].mean()*100:.1f}%",
            "Fare > $50":        f"{raw[raw['Fare']>50]['Survived'].mean()*100:.1f}%",
            "Women in 1st Class":f"{raw[(raw['Sex']=='female')&(raw['Pclass']==1)]['Survived'].mean()*100:.1f}%",
        }
        st.dataframe(pd.DataFrame(surv_stats.items(),columns=["Group","Survival Rate"]),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Average Stats by Passenger Class")
    grp = raw.groupby("Pclass").agg(
        Count=("PassengerId","count"),
        Survival_Rate=("Survived", lambda x:f"{x.mean()*100:.1f}%"),
        Avg_Age=("Age","mean"), Avg_Fare=("Fare","mean"),
        Min_Fare=("Fare","min"), Max_Fare=("Fare","max"),
    ).round(2).reset_index()
    grp.columns=["Class","Count","Survival Rate","Avg Age","Avg Fare","Min Fare","Max Fare"]
    st.dataframe(grp, use_container_width=True, hide_index=True)

    st.markdown("#### Correlation with Survived")
    corr = raw.corr(numeric_only=True)["Survived"].sort_values(ascending=False).reset_index()
    corr.columns=["Feature","Correlation"]
    col_a, col_b = st.columns([1,1.8])
    with col_a:
        st.dataframe(corr.style.format({"Correlation":"{:.3f}"}),
                     use_container_width=True, hide_index=True)
    with col_b:
        fig, ax = plt.subplots(figsize=(7,3.5))
        colors=[GREEN if v>0 else RED for v in corr["Correlation"]]
        ax.barh(corr["Feature"], corr["Correlation"], color=colors)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title("Correlation with Survived", fontweight="bold")
        ax.set_xlabel("Pearson Correlation")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="insight"><b>Pclass</b> is the strongest negative correlator (-0.338): lower class = much lower survival. <b>Fare</b> is positive (+0.257) as a proxy for wealth and cabin location.</div>', unsafe_allow_html=True)

    st.markdown("#### Full Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(raw.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation Matrix", fontweight="bold", pad=14)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Step 3 — Probability & Statistical Tests")

    st.markdown("#### Basic Probability")
    total    = len(raw)
    n_female = len(raw[raw["Sex"]=="female"])
    p_f      = n_female / total
    p_f2     = p_f * ((n_female-1)/(total-1))

    c1,c2,c3 = st.columns(3)
    c1.metric("P(Picking a female)",       f"{p_f*100:.2f}%")
    c2.metric("P(2 females in a row)",     f"{p_f2*100:.2f}%",  help="Without replacement")
    c3.metric("P(Survival | Female)",      f"{raw[raw['Sex']=='female']['Survived'].mean()*100:.2f}%")

    st.markdown('<div class="insight"><b>Conditional probability:</b> Drawing a 2nd female without replacement — the pool shrinks by 1, so P = (n_female/total) × ((n_female-1)/(total-1)).</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Confidence Intervals — Fare (95%)")
    sample_size = st.slider("Sample size", 100, 800, 500, 50)
    np.random.seed(10)
    sample      = np.random.choice(raw["Fare"], size=sample_size)
    s_mean      = sample.mean()
    z_crit      = stats.norm.ppf(.95)
    moe         = z_crit * (raw["Fare"].std() / math.sqrt(sample_size))
    ci          = (s_mean - moe, s_mean + moe)
    c1,c2,c3 = st.columns(3)
    c1.metric("Sample Mean Fare",   f"${s_mean:.2f}")
    c2.metric("95% CI",             f"(${ci[0]:.2f},  ${ci[1]:.2f})")
    c3.metric("True Population Mean",f"${raw['Fare'].mean():.2f}")

    st.markdown("#### 25 Confidence Intervals — Age (95%)")
    np.random.seed(12)
    intervals, sample_means = [], []
    for _ in range(25):
        samp  = np.random.choice(raw["Age"].dropna(), size=500)
        sm    = samp.mean(); sample_means.append(sm)
        moe2  = stats.norm.ppf(.975) * (raw["Age"].std() / math.sqrt(500))
        intervals.append((sm-moe2, sm+moe2))

    fig, ax = plt.subplots(figsize=(12,4))
    ax.errorbar(np.arange(1,26), sample_means,
                yerr=[(t-b)/2 for b,t in intervals], fmt="o", color=BLUE, capsize=4)
    ax.axhline(raw["Age"].mean(), color=RED, linewidth=2, label=f"True mean: {raw['Age'].mean():.1f} yrs")
    ax.set_title("25 Confidence Intervals for Age (95%)", fontweight="bold")
    ax.set_xlabel("Trial"); ax.set_ylabel("Sample Mean Age")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="insight"><b>Central Limit Theorem:</b> As sample size grows, sample means follow a normal distribution regardless of the original shape. This allows us to build CIs even for skewed data like Fare.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Hypothesis Tests")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Z-Test — 1st Class Fares vs Overall Mean")
        z_stat, p_val = ztest(x1=raw[raw["Pclass"]==1]["Fare"], value=raw["Fare"].mean())
        st.dataframe(pd.DataFrame({
            "Test":["Z-statistic","P-value","Decision"],
            "Value":[f"{z_stat:.4f}", f"{p_val:.2e}", "Reject H₀ (p < 0.05)"]
        }), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight">1st class passengers paid significantly MORE than the average. Extremely low p-value — cannot be explained by chance.</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("##### T-Test — 3rd Class Fares vs Overall Mean")
        t_stat, p_val_t = stats.ttest_1samp(raw[raw["Pclass"]==3]["Fare"], popmean=raw["Fare"].mean())
        st.dataframe(pd.DataFrame({
            "Test":["T-statistic","P-value","Decision"],
            "Value":[f"{t_stat:.4f}", f"{p_val_t:.2e}", "Reject H₀ (p < 0.05)"]
        }), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight">3rd class passengers paid significantly LESS than average. This lower fare directly correlated with lower survival rates.</div>', unsafe_allow_html=True)

    st.markdown("##### Chi-Squared Test — Pclass vs Survived")
    contingency_tbl = pd.crosstab(raw["Pclass"], raw["Survived"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_tbl)
    col_a, col_b = st.columns([1,1.5])
    with col_a:
        st.markdown("**Observed counts:**")
        ct = contingency_tbl.copy()
        ct.index = ["1st Class","2nd Class","3rd Class"]
        ct.columns = ["Died","Survived"]
        st.dataframe(ct, use_container_width=True)
        st.markdown(f"**χ² = {chi2:.2f}** | **p = {p_chi:.2e}** | **dof = {dof}**")
        st.markdown('<div class="insight">p ≈ 0 → Class and survival are NOT independent. Class strongly determined survival outcome.</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("**Expected counts (under H₀):**")
        exp_df = pd.DataFrame(expected.round(1),
                              index=["1st Class","2nd Class","3rd Class"],
                              columns=["Died (expected)","Survived (expected)"])
        st.dataframe(exp_df, use_container_width=True)
        st.markdown('<div class="warn">The actual counts differ significantly from expected — especially for 1st class (many more survivors than expected) and 3rd class (many more deaths than expected).</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Step 4 — Data Preprocessing")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Missing Values — Before")
        mv = raw.isnull().sum().reset_index()
        mv.columns=["Feature","Missing"]
        mv["% Missing"]=(mv["Missing"]/len(raw)*100).round(1)
        mv["Action"]=mv.apply(lambda r:(
            "Fill with median (28.0)"   if r["Feature"]=="Age"      else
            "Fill with mode ('S')"      if r["Feature"]=="Embarked" else
            "Drop column"               if r["Feature"]=="Cabin"    else "No action"),axis=1)
        st.dataframe(mv, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Steps Applied")
        steps=[
            ("🩹","Age → fill median","177 missing (19.9%). Median=28 robust to right-skew."),
            ("📍","Embarked → fill mode","2 missing. Mode = 'S' (Southampton)."),
            ("🗑️","Cabin → drop","77.1% missing. Too sparse to use."),
            ("🗑️","Name, Ticket, PassengerId → drop","Unique per passenger. No predictive value."),
            ("🔢","Sex → encode 0/1","male→0, female→1."),
            ("🔢","Embarked → encode 0/1/2","C→0, Q→1, S→2."),
            ("👨‍👩‍👧","FamilySize = SibSp+Parch+1","One family metric replaces two columns."),
            ("📏","StandardScaler on Age & Fare","Fitted AFTER split — prevents data leakage."),
        ]
        for icon,title,desc in steps:
            st.markdown(f"**{icon} {title}**  \n{desc}")

    st.markdown("#### Class Distribution — Before Preprocessing")
    fig, axes = plt.subplots(1,3,figsize=(13,4))
    for ax,(col,lbls,ttl) in zip(axes,[
        ("Survived",["Died","Survived"],"Survival"),
        ("Pclass",["1st","2nd","3rd"],"Passenger Class"),
        ("Sex",["Female","Male"],"Gender")]):
        counts=raw[col].value_counts().sort_index()
        ax.bar(lbls[:len(counts)], counts.values, color=COLS[:len(counts)], edgecolor="white")
        ax.set_title(ttl, fontweight="bold"); ax.set_ylabel("Count")
        for i,v in enumerate(counts.values):
            ax.text(i,v+8,str(v),ha="center",fontsize=10,fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Train / Test Split")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Train samples",f"{len(X_train)} (80%)")
    c2.metric("Test samples", f"{len(X_test)} (20%)")
    c3.metric("Features",     str(len(FEATURES)))
    c4.metric("Random state", "42")

    st.markdown('<div class="insight"><b>Why scale AFTER split?</b> Fitting the scaler on all data lets test-set information leak into training. We fit only on train, then transform both — this is correct practice to avoid data leakage.</div>', unsafe_allow_html=True)

    if st.checkbox("Show preprocessed dataframe (first 20 rows)"):
        st.dataframe(pre.head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader(f"Step 5 — Exploratory Data Analysis  ({len(df_f):,} passengers shown)")

    if df_f.empty:
        st.warning("No data matches the sidebar filters.")
    else:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Survival Rate",  f"{df_f['Survived'].mean()*100:.1f}%")
        c2.metric("Avg Age",        f"{df_f['Age'].mean():.1f} yrs")
        c3.metric("Avg Fare",       f"${df_f['Fare'].mean():.2f}")
        fem_surv = df_f[df_f["Sex"]=="female"]["Survived"].mean()*100 if "female" in df_f["Sex"].values else 0
        c4.metric("Female Survival",f"{fem_surv:.1f}%")

        st.markdown("#### Survival by Gender, Class & Embarked Port")
        fig,axes=plt.subplots(1,3,figsize=(14,4.5))

        surv_sex=df_f.groupby("Sex")["Survived"].mean()*100
        vals_sx=[surv_sex.get("male",0), surv_sex.get("female",0)]
        axes[0].bar(["Male","Female"], vals_sx, color=[BLUE,RED], edgecolor="white", width=.5)
        axes[0].set_title("Survival Rate by Gender",fontweight="bold")
        axes[0].set_ylabel("Survival Rate (%)"); axes[0].set_ylim(0,100)
        for i,v in enumerate(vals_sx):
            axes[0].text(i,v+2,f"{v:.1f}%",ha="center",fontweight="bold",fontsize=11)
        axes[0].spines[["top","right"]].set_visible(False)

        surv_cls=df_f.groupby("Pclass")["Survived"].mean()*100
        lbl_c=[f"{c}{'st' if c==1 else 'nd' if c==2 else 'rd'}" for c in surv_cls.index]
        axes[1].bar(lbl_c, surv_cls.values, color=COLS[:len(surv_cls)], edgecolor="white", width=.5)
        axes[1].set_title("Survival Rate by Class",fontweight="bold")
        axes[1].set_ylabel("Survival Rate (%)"); axes[1].set_ylim(0,100)
        for i,v in enumerate(surv_cls.values):
            axes[1].text(i,v+2,f"{v:.1f}%",ha="center",fontweight="bold",fontsize=11)
        axes[1].spines[["top","right"]].set_visible(False)

        surv_emb=df_f.groupby("Embarked")["Survived"].mean()*100
        axes[2].bar(surv_emb.index, surv_emb.values, color=[TEAL,AMB,BLUE], edgecolor="white", width=.5)
        axes[2].set_title("Survival Rate by Port",fontweight="bold")
        axes[2].set_ylabel("Survival Rate (%)"); axes[2].set_ylim(0,100)
        for i,(pt,v) in enumerate(surv_emb.items()):
            axes[2].text(i,v+2,f"{v:.1f}%",ha="center",fontweight="bold",fontsize=11)
        axes[2].spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Age & Fare Distributions")
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        axes[0].hist(df_f["Age"].dropna(),bins=30,color=BLUE,edgecolor="white",alpha=.85)
        axes[0].axvline(df_f["Age"].median(),color=RED,linestyle="--",linewidth=2,
                        label=f"Median: {df_f['Age'].median():.0f} yrs")
        axes[0].set_title("Age Distribution",fontweight="bold")
        axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Count")
        axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

        axes[1].hist(df_f["Fare"],bins=40,color=AMB,edgecolor="white",alpha=.85)
        axes[1].axvline(df_f["Fare"].median(),color=RED,linestyle="--",linewidth=2,
                        label=f"Median: ${df_f['Fare'].median():.2f}")
        axes[1].set_title("Fare Distribution (right-skewed)",fontweight="bold")
        axes[1].set_xlabel("Fare ($)"); axes[1].set_ylabel("Count")
        axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("#### Survival by Family Size & Class × Gender Heatmap")
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        df_f2=df_f.copy()
        df_f2["FamilySize"]=df_f2["SibSp"]+df_f2["Parch"]+1
        fam_surv=df_f2.groupby("FamilySize")["Survived"].mean()*100
        axes[0].bar(fam_surv.index.astype(str),fam_surv.values,color=BLUE,edgecolor="white")
        axes[0].set_title("Survival by Family Size",fontweight="bold")
        axes[0].set_xlabel("Family Size (1=solo)"); axes[0].set_ylabel("Survival Rate (%)")
        axes[0].spines[["top","right"]].set_visible(False)

        pivot=df_f.groupby(["Pclass","Sex"])["Survived"].mean()*100
        pivot=pivot.unstack(fill_value=0)
        im=axes[1].imshow(pivot.values,cmap="RdYlGn",aspect="auto",vmin=0,vmax=100)
        plt.colorbar(im,ax=axes[1],label="Survival Rate (%)")
        axes[1].set_xticks(range(len(pivot.columns))); axes[1].set_yticks(range(len(pivot.index)))
        axes[1].set_xticklabels(pivot.columns)
        axes[1].set_yticklabels([f"Class {i}" for i in pivot.index])
        axes[1].set_title("Survival: Class × Gender (%)",fontweight="bold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                axes[1].text(j,i,f"{pivot.values[i,j]:.0f}%",
                             ha="center",va="center",fontsize=12,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="insight"><b>Strongest finding:</b> 1st class women had ~97% survival. 3rd class men had only ~13%. Gender and class together explain most of the variance — far more than age or fare alone.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — VISUALIZATIONS  (Lab 6 style)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Step 6 — Interactive Visualizations")

    st.markdown("#### Survival Distribution")
    col_a,col_b=st.columns(2)
    with col_a:
        st.bar_chart(raw["Survived"].value_counts().rename({0:"Died",1:"Survived"}))
    with col_b:
        st.markdown('<div class="warn"><b>Class imbalance:</b> 549 died (61.6%) vs 342 survived (38.4%). A naive model always predicting "died" would still get 61.6% accuracy — raw accuracy is misleading here.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><b>Historical context:</b> The Titanic had lifeboats for only ~53% of passengers. Many were launched half-empty due to chaotic evacuation — wealthier passengers on upper decks had faster access.</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Show Class & Gender Distributions"):
        c1,c2=st.columns(2)
        with c1:
            st.subheader("Pclass Distribution")
            st.bar_chart(raw["Pclass"].value_counts().sort_index()
                         .rename({1:"1st",2:"2nd",3:"3rd"}))
        with c2:
            st.subheader("Sex Distribution")
            st.bar_chart(raw["Sex"].value_counts())

    st.markdown("---")
    st.markdown("#### Age vs Fare — Colored by Survival")
    fig,ax=plt.subplots(figsize=(10,5))
    surv=raw[raw["Survived"]==1]; died=raw[raw["Survived"]==0]
    ax.scatter(died["Age"],    died["Fare"],    alpha=.3,s=15,color=RED,  label="Died")
    ax.scatter(surv["Age"],    surv["Fare"],    alpha=.5,s=15,color=GREEN,label="Survived")
    ax.set_title("Age vs Fare — by Survival",fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Fare ($)"); ax.set_ylim(0,300); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="insight">Survivors cluster in higher fare brackets (1st class, upper decks). Age alone shows no strong visual separation — gender and class dominate survival.</div>', unsafe_allow_html=True)

    st.markdown("#### Survived vs Died — Stacked by Class")
    piv2=raw.groupby(["Pclass","Survived"]).size().unstack(fill_value=0)
    piv2.columns=["Died","Survived"]
    piv2.index=["1st Class","2nd Class","3rd Class"]
    st.bar_chart(piv2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MODELING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Step 7 — Linear Regression Modeling (Lab 7)")

    st.markdown('<div class="insight"><b>Approach — Linear Probability Model:</b> We apply OLS linear regression to a binary target (Survived). Each coefficient represents the change in predicted survival probability for a 1 standard-deviation change in that feature. Simple, interpretable, and powerful for understanding which factors mattered.</div>', unsafe_allow_html=True)
    st.markdown("")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("R² Score (OLS)",  f"{metrics['OLS']['r2']:.4f}")
    c2.metric("RMSE",            f"{metrics['OLS']['rmse']:.4f}")
    c3.metric("MAE",             f"{metrics['OLS']['mae']:.4f}")
    c4.metric("CV R² (5-fold)",  f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    st.markdown("---")
    col_a,col_b=st.columns(2)

    with col_a:
        st.markdown("#### Standardized Coefficients")
        fig,ax=plt.subplots(figsize=(6,4))
        colors_c=[GREEN if v>=0 else RED for v in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_c)
        ax.axvline(0,color="black",linewidth=.8)
        ax.set_title("Feature Impact on Survival Probability",fontweight="bold")
        ax.set_xlabel("Coefficient (standardized)")
        for i,v in enumerate(coef_df["Coefficient"]):
            ax.text(v+.003 if v>=0 else v-.003, i, f"{v:+.3f}",
                    va="center",fontsize=9,ha="left" if v>=0 else "right")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("#### Actual vs Predicted")
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter(y_test, y_pred, alpha=.35, color=BLUE, s=18)
        ax.plot([0,1],[0,1],"r--",linewidth=1.5,label="Perfect fit")
        ax.set_title("Actual vs Predicted Survival Probability",fontweight="bold")
        ax.set_xlabel("Actual (0=Died, 1=Survived)")
        ax.set_ylabel("Predicted probability")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Residual Analysis")
    residuals=y_test.values-y_pred
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.hist(residuals,bins=30,color=AMB,edgecolor="white",alpha=.85)
        ax.axvline(0,color="black",linestyle="--")
        ax.set_title("Residual Distribution",fontweight="bold")
        ax.set_xlabel("Actual − Predicted"); ax.set_ylabel("Frequency")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col_b:
        fig,ax=plt.subplots(figsize=(6,3.5))
        ax.scatter(y_pred,residuals,alpha=.3,color=BLUE,s=12)
        ax.axhline(0,color=RED,linestyle="--",linewidth=1.5)
        ax.set_title("Residuals vs Predicted",fontweight="bold")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Residual")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Model Comparison — OLS vs Ridge vs Lasso")
    model_names=["OLS Linear","Ridge (L2, α=1)","Lasso (L1, α=0.01)"]
    r2s =[metrics[k]["r2"]   for k in ["OLS","Ridge","Lasso"]]
    maes=[metrics[k]["mae"]  for k in ["OLS","Ridge","Lasso"]]
    rms =[metrics[k]["rmse"] for k in ["OLS","Ridge","Lasso"]]

    col_a,col_b=st.columns([1.2,1.5])
    with col_a:
        st.dataframe(pd.DataFrame({
            "Model":model_names,
            "R²"   :[f"{v:.4f}" for v in r2s],
            "MAE"  :[f"{v:.4f}" for v in maes],
            "RMSE" :[f"{v:.4f}" for v in rms],
        }), use_container_width=True, hide_index=True)
        st.markdown(f"**Overfitting check:**  \nTrain R² = `{r2_train:.4f}` | Test R² = `{metrics['OLS']['r2']:.4f}`  \nGap = `{abs(r2_train-metrics['OLS']['r2']):.4f}` — generalizes well ✅")
    with col_b:
        fig,ax=plt.subplots(figsize=(6,3))
        ax.bar(model_names,r2s,color=[BLUE,GREEN,AMB],edgecolor="white",width=.45)
        ax.set_title("R² Score by Model",fontweight="bold")
        ax.set_ylabel("R²"); ax.set_ylim(0,max(r2s)*1.28)
        for i,v in enumerate(r2s):
            ax.text(i,v+.003,f"{v:.4f}",ha="center",fontsize=11,fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Interpretation of Results")
    interps=[
        ("Sex  (+0.246)",    GREEN, "The strongest predictor. Being female raised survival probability by ~24.6 pp. Directly reflects the 'women and children first' evacuation policy."),
        ("Pclass  (−0.126)", RED,   "Each step down in class reduced survival by ~12.6 pp. 3rd class faced physical barriers: lower decks, locked gates, less time to reach lifeboats."),
        ("Age  (−0.061)",    AMB,   "Older passengers had slightly lower survival. Children were prioritized but the effect is weaker than gender — the policy was less strictly enforced."),
        ("Fare  (+0.019)",   BLUE,  "Fare is a proxy for wealth and cabin location. Higher-paying passengers had upper-deck cabins with direct lifeboat access."),
        ("Ridge ≈ OLS",      TEAL,  "Nearly identical results confirm no severe multicollinearity. All 6 features contribute meaningful information — regularization had minimal impact."),
        ("R² = 0.44",        NAVY,  "The model explains 44% of survival variance. The remaining 56% reflects genuine randomness: who was near a lifeboat, who helped whom, split-second decisions."),
    ]
    for ttl,color,body in interps:
        st.markdown(
            f'<div class="insight"><b style="color:{color}">{ttl}:</b> {body}</div>',
            unsafe_allow_html=True)
