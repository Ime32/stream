# 🚢 Titanic Dashboard — Deployment Guide
### DIFFALLAH Imene | MI3 | ENSTA Alger | 2026

---

## 📁 Files You Need

```
your_project_folder/
├── titanic_dashboard_final.py   ← the dashboard (this file)
├── train.csv                    ← Titanic dataset
└── requirements.txt             ← Python dependencies
```

> ⚠️ **Important:** `train.csv` must be in the **same folder** as the `.py` file.

---

## ▶️ Option 1 — Run Locally (fastest, for testing)

### Step 1 — Install Python (if not already)
Download from https://python.org → version 3.9 or higher

### Step 2 — Install dependencies
Open a terminal (CMD / PowerShell on Windows, Terminal on Mac/Linux):
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Step 3 — Run the dashboard
```bash
cd path/to/your_project_folder
streamlit run titanic_dashboard_final.py
```

Your browser will open automatically at: **http://localhost:8501**

---

## ☁️ Option 2 — Deploy on Streamlit Community Cloud (FREE, public URL)

This is what you submit for your course — a public URL anyone can visit.

### Step 1 — Push your project to GitHub
1. Create a free account at https://github.com
2. Create a **new repository** (e.g., `titanic-dashboard`)
3. Upload these 3 files:
   - `titanic_dashboard_final.py`
   - `train.csv`
   - `requirements.txt`

Via GitHub website: click **"Add file" → "Upload files"** and drag all 3 files.

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `your-username/titanic-dashboard`
   - **Branch:** `main`
   - **Main file path:** `titanic_dashboard_final.py`
5. Click **"Deploy!"**

⏳ Wait ~2 minutes while it installs dependencies. Then you get a public URL like:
```
https://your-username-titanic-dashboard-titanic-dashboard-final-xxxxx.streamlit.app
```

### Step 3 — Share the URL
Copy the URL and submit it on GitHub and in your final submission.

---

## 🐙 GitHub Repository Setup (for the Lab 6 submission requirement)

Your GitHub repo should look like this:

```
Data-Analysis-ENSTA-[your-name]/
├── README.md
├── requirements.txt
├── train.csv
├── titanic_dashboard_final.py   ← Lab 6 deployment
├── Lab1pytjup.ipynb             ← Lab 1: Data collection
├── Lab2Jupyth.ipynb             ← Lab 2: EDA
├── dataset_preprocessing.ipynb  ← Preprocessing
├── lab7_titanic_regression.py   ← Lab 7: Modeling
└── lab7_regression_results.png  ← Results figure
```

### How to update an existing GitHub repo:
1. Go to your repo on github.com
2. Click **"Add file" → "Upload files"**
3. Upload `titanic_dashboard_final.py` and `requirements.txt`
4. Commit message: `"Add Lab 6 Streamlit deployment + requirements"`

---

## 🏃‍♀️ Quick Cheat Sheet

| Action | Command |
|--------|---------|
| Install all deps | `pip install -r requirements.txt` |
| Run locally | `streamlit run titanic_dashboard_final.py` |
| Stop the server | Press `Ctrl + C` in terminal |
| Change port | `streamlit run titanic_dashboard_final.py --server.port 8502` |

---

## 🔍 Dashboard Sections

| Tab | Content |
|-----|---------|
| 📦 1 · Collection | Dataset overview, column dictionary, missing values heatmap, raw data preview |
| 🔧 2 · Preprocessing | All 9 steps explained, before/after distributions, train/test split |
| 📊 3 · EDA | Survival by gender, class, age, fare, family size, embarkation port |
| 🤖 4 · Modeling | OLS/Ridge/Lasso comparison, coefficients, residual analysis, interpretations |
| 🔮 5 · Live Predictor | Real-time survival prediction for any passenger profile |

---

## ❓ Common Issues

**"FileNotFoundError: train.csv"**
→ Make sure `train.csv` is in the same folder as the `.py` file when you run it.
→ On Streamlit Cloud: make sure `train.csv` is uploaded to your GitHub repo.

**"ModuleNotFoundError: No module named 'streamlit'"**
→ Run: `pip install streamlit`

**App is slow on first load**
→ Normal! Streamlit Cloud takes ~30 seconds to wake up. Locally it's instant.

**Sidebar filters show "No data matches"**
→ You deselected all options in one filter. Click the filter and re-add options.
