"""
Healthcare Analytics for Doctor Visits - Streamlit app (fixed version)
Save as app.py and run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -------------------- Helper functions --------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def safe_numeric_cols(df):
    return df.select_dtypes(include=["number"]).columns.tolist()

def safe_categorical_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def make_age_bins(series, bins=[0,18,30,45,60,75,100]):
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return pd.cut(series.fillna(-1), bins=bins, labels=labels, include_lowest=True)

# -------------------- Page setup --------------------
st.set_page_config(page_title="Doctor Visits Analytics", layout="wide")
st.title("🏥 Healthcare Analytics — Doctor Visits")

# -------------------- Sidebar: data loading --------------------
st.sidebar.header("📂 Data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample = st.sidebar.checkbox("Use built-in sample (from upload)", value=True)

DEFAULT_PATH = "/mnt/data/1719219834-DoctorVisits - DA (1).csv"
SAMPLE_PATH = "/mnt/data/doctor_visits_sample.csv"

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to read uploaded file: {e}")
elif use_sample:
    try:
        df = load_csv(SAMPLE_PATH)
        st.sidebar.success("✅ Loaded built-in sample dataset.")
    except Exception:
        try:
            df = load_csv(DEFAULT_PATH)
            st.sidebar.success("✅ Loaded default dataset.")
        except Exception as e:
            st.sidebar.error(f"❌ Could not load sample or default file: {e}")
else:
    st.sidebar.info("Please upload a CSV file or enable the sample option.")

if df is None:
    st.stop()

# Clean / normalize column names
df = df.copy()
df.columns = [c.strip() for c in df.columns]

# -------------------- Overview --------------------
st.subheader("📊 Data Overview")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

with col2:
    numeric_cols = safe_numeric_cols(df)
    cat_cols = safe_categorical_cols(df)
    st.metric("Numeric Columns", len(numeric_cols))
    st.metric("Categorical Columns", len(cat_cols))

with col3:
    if "visits" in df.columns:
        st.metric("Avg. Visits", round(df["visits"].mean(), 2))
    if "age" in df.columns:
        st.metric("Avg. Age", round(df["age"].mean(), 1))

st.dataframe(df.head(20))

# -------------------- Column inspection --------------------
st.sidebar.header("🔍 Explore Data")
col_to_view = st.sidebar.selectbox("Select column to inspect", df.columns.tolist())

if col_to_view in numeric_cols:
    st.subheader(f"Distribution — {col_to_view}")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df[col_to_view].dropna(), bins=30, color="skyblue", edgecolor="black")
    ax.set_xlabel(col_to_view)
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.subheader(f"Value Counts — {col_to_view}")
    vc = df[col_to_view].value_counts().nlargest(20)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(vc.index.astype(str), vc.values, color="lightcoral", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# -------------------- Filters --------------------
st.sidebar.header("🎛️ Filters")
filters = {}

# Gender filter
if "gender" in df.columns:
    genders = ["All"] + sorted(df["gender"].dropna().astype(str).unique().tolist())
    f_gender = st.sidebar.selectbox("Gender", genders)
    filters["gender"] = None if f_gender == "All" else f_gender

# Private insurance filter
if "private" in df.columns:
    privs = ["All"] + sorted(df["private"].dropna().astype(str).unique().tolist())
    f_priv = st.sidebar.selectbox("Private insurance", privs)
    filters["private"] = None if f_priv == "All" else f_priv

# ✅ Fixed Age Slider (handles identical min/max)
if "age" in df.columns:
    valid_ages = df["age"].dropna()
    if valid_ages.nunique() > 1:
        min_age = int(valid_ages.min())
        max_age = int(valid_ages.max())
        if min_age < max_age:
            f_age = st.sidebar.slider(
                "Age range", min_value=min_age, max_value=max_age,
                value=(min_age, max_age), step=1
            )
        else:
            st.sidebar.info(f"Age has only one value ({min_age}). Slider disabled.")
            f_age = None
    else:
        st.sidebar.info("Age column has no variation. Slider disabled.")
        f_age = None
else:
    f_age = None

filters["age"] = f_age

# Apply filters
df_filtered = df.copy()
if filters.get("gender"):
    df_filtered = df_filtered[df_filtered["gender"].astype(str) == filters["gender"]]
if filters.get("private"):
    df_filtered = df_filtered[df_filtered["private"].astype(str) == filters["private"]]
if filters.get("age"):
    low, high = filters["age"]
    df_filtered = df_filtered[df_filtered["age"].between(low, high)]

st.markdown(f"**Filtered rows:** {df_filtered.shape[0]}")

# -------------------- Charts --------------------
st.subheader("📈 Analytics Charts")
c1, c2 = st.columns(2)

with c1:
    if "visits" in df_filtered.columns:
        st.markdown("**Visits Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(df_filtered["visits"].dropna(), bins=25, color="lightgreen", edgecolor="black")
        ax.set_xlabel("Visits")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    if "gender" in df_filtered.columns and "visits" in df_filtered.columns:
        st.markdown("**Average Visits by Gender**")
        agg = df_filtered.groupby("gender")["visits"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(agg.index.astype(str), agg.values, color="gold", edgecolor="black")
        ax.set_ylabel("Average Visits")
        st.pyplot(fig)

with c2:
    if "age" in df_filtered.columns and "visits" in df_filtered.columns:
        st.markdown("**Average Visits by Age Group**")
        df_filtered["age_bin"] = make_age_bins(df_filtered["age"])
        agg2 = df_filtered.groupby("age_bin")["visits"].mean().dropna()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(agg2.index.astype(str), agg2.values, color="orchid", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        ax.set_ylabel("Average Visits")
        st.pyplot(fig)

# -------------------- Illness / Chronic Overview --------------------
possible_ill = None
for cand in ["illness", "nchronic", "lchronic", "condition", "diagnosis"]:
    if cand in df_filtered.columns:
        possible_ill = cand
        break

if possible_ill:
    st.subheader("🩺 Illness / Chronic Overview")
    st.write(f"Using column: **{possible_ill}**")
    vc = df_filtered[possible_ill].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(vc.index.astype(str), vc.values, color="deepskyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# -------------------- Predictive Model --------------------
st.subheader("🤖 Predictive Model — Predict `visits`")

if "visits" not in df.columns:
    st.info("No 'visits' column found. Cannot train model.")
else:
    X = df_filtered.select_dtypes(include=["number"]).copy()
    X = X.drop(columns=[c for c in ["Unnamed: 0", "visits"] if c in X.columns], errors=True)
    y = df_filtered["visits"].fillna(0)

    if X.shape[1] == 0:
        st.warning("No numeric features available for training.")
    else:
        st.write("Features used:", X.columns.tolist())
        X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        st.metric("R² (test)", f"{r2:.3f}")
        st.metric("MSE (test)", f"{mse:.3f}")

        coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
        st.write("Top model coefficients:")
        st.dataframe(coefs.head(10))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, preds, alpha=0.5, color="steelblue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("True Visits")
        ax.set_ylabel("Predicted Visits")
        st.pyplot(fig)

# -------------------- Export Filtered Data --------------------
st.subheader("💾 Export Data")

def convert_df_to_csv_bytes(df_local):
    return df_local.to_csv(index=False).encode("utf-8")

csv_bytes = convert_df_to_csv_bytes(df_filtered)
st.download_button(
    "Download filtered dataset (CSV)",
    data=csv_bytes,
    file_name="doctor_visits_filtered.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit. Extend with ML, dashboards, or cloud integration.")
