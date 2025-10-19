# ============================================================
# 🎬 NETFLIX DATA ANALYSIS - Robust & Fixed Version
# ============================================================
# Usage: python netflix_analysis_fixed.py
# (This is a standalone script using pandas + matplotlib + seaborn + plotly)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textwrap import shorten

# ---------- Configuration ----------
# Default path (change if needed)
DATA_PATH = "/mnt/data/Netflix Dataset.csv"  # uploaded file path
# DATA_PATH = r"C:\Users\adity\OneDrive\Documents\Netflix Dataset.csv"  # optional alternate

sns.set_theme()
plt.rcParams.update({"figure.max_open_warning": 0})

# ---------- Helpers ----------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")
    return df

def normalize_columns(df):
    # Make column names predictable: lower, strip, replace spaces and special chars with underscore
    new_cols = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[ \-\/]+", "_", regex=True)
                  .str.replace(r"[^\w_]", "", regex=True)
    )
    df.columns = new_cols
    return df

def safe_rename(df):
    # Map common alternative names to canonical names used below
    rename_map = {
        "category": "type",
        "showid": "show_id",
        "show_id": "show_id",
        "release_year": "release_year",
        "releaseyear": "release_year",
        "date_added": "date_added",
        "dateadded": "date_added",
        "listed_in": "listed_in",
        "listedin": "listed_in",
        "duration": "duration",
        "rating": "rating",
        "description": "description",
        "title": "title",
        "director": "director",
        "cast": "cast",
        "country": "country"
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df.rename(columns=existing, inplace=True)
    return df

def ensure_columns(df, expected):
    for col, default in expected.items():
        if col not in df.columns:
            df[col] = default() if callable(default) else default
        else:
            # fill na with default if single value provided
            if not callable(default):
                df[col].fillna(default, inplace=True)
    return df

def safe_plot(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        plt.show()
    except Exception as e:
        print(f"[plotting skipped] {e}")

# ---------- Load & Normalize ----------
try:
    df = safe_read_csv(DATA_PATH)
except Exception as e:
    print("Error loading CSV:", e)
    sys.exit(1)

print("Original columns:", list(df.columns)[:50])
df = normalize_columns(df)
df = safe_rename(df)
print("Normalized columns:", list(df.columns)[:50])

# ---------- Ensure important columns exist and are filled ----------
expected_defaults = {
    "show_id": lambda: pd.Series([f"unknown_{i}" for i in range(len(df))]),
    "type": "Unknown",
    "title": "Unknown",
    "director": "Unknown",
    "cast": "Unknown",
    "country": "Unknown",
    "date_added": pd.NaT,
    "release_year": np.nan,
    "rating": "Unknown",
    "duration": "Unknown",
    "listed_in": "Unknown",
    "description": "Unknown"
}
df = ensure_columns(df, expected_defaults)

# Convert types safely
# date_added -> datetime
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
# year_added derived from date_added
df["year_added"] = df["date_added"].dt.year

# release_year -> numeric (coerce errors)
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

# Some normalization for 'type' (movie / tv show)
df["type"] = df["type"].astype(str).str.strip().str.title().replace({"Tv Show": "TV Show", "Tv show": "TV Show"})

# Clean listed_in -> keep as string
df["listed_in"] = df["listed_in"].astype(str).replace("nan", "Unknown")

print("\nAfter cleaning: shape =", df.shape)
print(df.head(3).to_string(index=False))

# ---------- QUICK CHECKS ----------
if df.empty:
    print("DataFrame is empty after cleaning — nothing to plot.")
    sys.exit(0)

# ---------- EDA & PLOTS ----------
# Use inline plotting; for scripts this will open windows (or you can save to files)
# 1) Movies vs TV Shows count
if "type" in df.columns:
    plt.figure(figsize=(6,4))
    safe_plot(lambda: sns.countplot(data=df, x="type", order=df["type"].value_counts().index, palette="coolwarm"))
    plt.title("Movies vs TV Shows on Netflix")
    plt.tight_layout()

# 2) Top 10 Countries
if "country" in df.columns:
    top_countries = df["country"].replace("Unknown", np.nan).dropna().str.split(",").explode().str.strip().value_counts().head(10)
    if not top_countries.empty:
        plt.figure(figsize=(8,5))
        safe_plot(lambda: sns.barplot(x=top_countries.values, y=top_countries.index, palette="magma"))
        plt.title("Top 10 Countries Producing Netflix Content")
        plt.xlabel("Number of Titles")
        plt.tight_layout()

# 3) Titles added per year (year_added)
if "year_added" in df.columns and df["year_added"].notna().any():
    year_counts = df["year_added"].value_counts().sort_index()
    if not year_counts.empty:
        plt.figure(figsize=(10,4))
        safe_plot(lambda: plt.bar(year_counts.index.astype(str), year_counts.values))
        plt.title("Number of Titles Added per Year")
        plt.xlabel("Year Added")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

# 4) Top genres (listed_in)
if "listed_in" in df.columns:
    genres = df["listed_in"].dropna().astype(str).str.split(",").explode().str.strip()
    top_genres = genres.value_counts().head(10)
    if not top_genres.empty:
        plt.figure(figsize=(9,5))
        safe_plot(lambda: sns.barplot(x=top_genres.values, y=top_genres.index, palette="viridis"))
        plt.title("Top 10 Genres on Netflix")
        plt.xlabel("Number of Titles")
        plt.tight_layout()

# 5) Ratings distribution
if "rating" in df.columns:
    rating_counts = df["rating"].value_counts()
    if not rating_counts.empty:
        plt.figure(figsize=(8,5))
        safe_plot(lambda: sns.barplot(x=rating_counts.values, y=rating_counts.index, palette="rocket"))
        plt.title("Distribution of Content Ratings")
        plt.xlabel("Count")
        plt.tight_layout()

# 6) Release year distribution
if "release_year" in df.columns and df["release_year"].notna().any():
    plt.figure(figsize=(10,4))
    safe_plot(lambda: sns.histplot(df["release_year"].dropna().astype(int), bins=30, kde=True))
    plt.title("Content Release Year Distribution")
    plt.xlabel("Release Year")
    plt.tight_layout()

# 7) Top directors
if "director" in df.columns:
    directors_series = df["director"].replace("Unknown", np.nan).dropna().str.split(",").explode().str.strip()
    top_directors = directors_series.value_counts().head(10)
    if not top_directors.empty:
        plt.figure(figsize=(9,5))
        safe_plot(lambda: sns.barplot(x=top_directors.values, y=top_directors.index, palette="crest"))
        plt.title("Top 10 Directors with Most Netflix Titles")
        plt.xlabel("Number of Titles")
        plt.tight_layout()

# 8) Top actors
if "cast" in df.columns:
    actors = df["cast"].replace("Unknown", np.nan).dropna().str.split(",").explode().str.strip()
    top_actors = actors.value_counts().head(10)
    if not top_actors.empty:
        plt.figure(figsize=(9,5))
        safe_plot(lambda: sns.barplot(x=top_actors.values, y=top_actors.index))
        plt.title("Top 10 Most Frequent Actors on Netflix")
        plt.xlabel("Number of Titles")
        plt.tight_layout()

# ---------- Plotly interactive examples (will open in browser if run in notebook/environment that supports it) ----------
try:
    # Content added by year (plotly)
    if "year_added" in df.columns and df["year_added"].notna().any():
        fig = px.histogram(df, x="year_added", color="type", title="Content Added Each Year by Type", barmode="group")
        fig.show()
    # Top genres interactive
    if "listed_in" in df.columns:
        genre_data = genres.value_counts().head(15)
        if not genre_data.empty:
            fig2 = px.bar(x=genre_data.values, y=genre_data.index, orientation="h", title="Top Genres (interactive)")
            fig2.show()
except Exception as e:
    print("[plotly skipped]", e)

# ---------- Simple CLI search utility ----------
def search_title(query, exact=False, max_results=10):
    if exact:
        mask = df["title"].str.lower() == query.strip().lower()
    else:
        mask = df["title"].str.contains(query, case=False, na=False)
    res = df.loc[mask, ["show_id", "title", "director", "release_year", "rating", "country", "listed_in", "description"]]
    if res.empty:
        print(f"No matches found for '{query}'.")
    else:
        print(f"\nTop {min(len(res), max_results)} results for '{query}':")
        print(res.head(max_results).to_string(index=False))

# Example usage (uncomment to test search from script)
# search_title("House of Cards", exact=False)

print("\nScript finished. Use the `search_title(query)` function in this script to query titles (if running interactively).")
