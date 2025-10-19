# =========================================================
# 🎬 Netflix Data Analysis Dashboard using Streamlit
# =========================================================

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Streamlit Page Setup
st.set_page_config(page_title="Netflix Data Analysis", layout="wide")
st.title("🎬 Netflix Data Analysis Dashboard")

# =========================================================
# 1️⃣ Upload Netflix CSV Dataset
# =========================================================
st.sidebar.header("Upload Your Netflix Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    st.success("CSV Loaded Successfully!")
    st.dataframe(df.head())

    # Fill missing values
    df.fillna("Unknown", inplace=True)

    # =========================================================
    # 2️⃣ Basic Statistics & Insights
    # =========================================================
    st.header("📈 Basic Statistics & Insights")
    st.write(df.describe(include='all'))

    # =========================================================
    # 3️⃣ Visualizations
    # =========================================================

    # Movies vs TV Shows
    if 'type' in df.columns:
        st.subheader("🎞 Distribution of Movies vs TV Shows")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='type', palette='coolwarm', ax=ax)
        ax.set_xlabel("Type of Content")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Top 10 Countries
    if 'country' in df.columns:
        st.subheader("🌍 Top 10 Countries with Most Titles")
        top_countries = df['country'].value_counts().head(10)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma', ax=ax2)
        ax2.set_xlabel("Number of Titles")
        ax2.set_ylabel("Country")
        st.pyplot(fig2)

    # Titles by Year
    if 'release_year' in df.columns:
        st.subheader("📅 Number of Titles Released Over the Years")
        year_count = df['release_year'].value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        sns.lineplot(x=year_count.index, y=year_count.values, marker='o', ax=ax3)
        ax3.set_xlabel("Release Year")
        ax3.set_ylabel("Number of Titles")
        st.pyplot(fig3)

    # Top 10 Genres
    if 'listed_in' in df.columns:
        st.subheader("🎭 Top 10 Genres on Netflix")
        genres = df['listed_in'].str.split(',').explode().str.strip()
        top_genres = genres.value_counts().head(10)
        fig4, ax4 = plt.subplots()
        sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax4)
        ax4.set_xlabel("Count")
        ax4.set_ylabel("Genre")
        st.pyplot(fig4)

    # Ratings Distribution
    if 'rating' in df.columns:
        st.subheader("⭐ Distribution of Ratings")
        fig5, ax5 = plt.subplots()
        sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index, palette='rocket', ax=ax5)
        ax5.set_xlabel("Count")
        ax5.set_ylabel("Rating")
        st.pyplot(fig5)

    # Duration of Movies
    if 'duration' in df.columns and 'type' in df.columns:
        st.subheader("⏱ Distribution of Movie Durations")
        movies = df[df['type'].str.lower() == "movie"].copy()
        movies['duration_mins'] = movies['duration'].str.replace(" min", "").replace("Unknown", "0").astype(int)
        fig6, ax6 = plt.subplots()
        sns.histplot(movies['duration_mins'], bins=20, kde=True, color='skyblue', ax=ax6)
        ax6.set_xlabel("Duration (minutes)")
        ax6.set_ylabel("Number of Movies")
        st.pyplot(fig6)

    # =========================================================
    # 4️⃣ Sidebar Movie Search
    # =========================================================
    st.sidebar.header("🔍 Search Netflix Titles")

    # Search input
    search_term = st.sidebar.text_input("Enter movie/show name")

    # Genre filter
    genre_filter = []
    if 'listed_in' in df.columns:
        genre_list = df['listed_in'].str.split(',').explode().str.strip().unique()
        genre_filter = st.sidebar.multiselect("Filter by Genre", genre_list)

    # Year filter
    year_filter = None
    if 'release_year' in df.columns:
        min_year = int(df['release_year'].min())
        max_year = int(df['release_year'].max())
        year_filter = st.sidebar.slider("Filter by Year", min_year, max_year, (min_year, max_year))

    # Filter DataFrame
    filtered_df = df.copy()
    if search_term:
        if 'title' in df.columns:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
        elif 'show_title' in df.columns:
            filtered_df = filtered_df[filtered_df['show_title'].str.contains(search_term, case=False, na=False)]
    if genre_filter and 'listed_in' in df.columns:
        filtered_df = filtered_df[filtered_df['listed_in'].apply(lambda x: any(g in x for g in genre_filter))]
    if year_filter and 'release_year' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_filter[0]) & (filtered_df['release_year'] <= year_filter[1])]

    # Display search results as cards
    st.header("🎬 Search Results")
    if not filtered_df.empty:
        cols = st.columns(3)
        for i, row in filtered_df.iterrows():
            with cols[i % 3]:
                # Try to show poster if column exists
                if 'poster' in df.columns and row['poster'] != "Unknown":
                    st.image(row['poster'], use_column_width=True)
                st.subheader(row.get('title', row.get('show_title', 'Unknown')))
                st.write(f"Genre: {row.get('listed_in', 'Unknown')}")
                st.write(f"Year: {row.get('release_year', 'Unknown')}")
    else:
        st.info("No results found for your search.")

else:
    st.warning("Please upload a CSV file to continue.")

# =========================================================
# ℹ️ About This Dataset
# =========================================================
st.header("ℹ️ About This Dataset")
st.markdown("""
**Source:** Netflix Titles Dataset  

**Common Columns:**
- `show_id`: Unique ID for every show
- `type`: Movie or TV Show
- `title`: Title of the show
- `director`: Director name
- `cast`: List of main actors
- `country`: Country of production
- `date_added`: Date added to Netflix
- `release_year`: Year released
- `rating`: Age rating
- `duration`: Duration or number of seasons
- `listed_in`: Genre/category
- `description`: Summary of the title
""")
st.markdown("Built with ❤️ using **Streamlit**, **Pandas**, and **Seaborn**.")
