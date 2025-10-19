# =========================================================
# 🎬 Netflix Data Analysis Dashboard using Streamlit
# =========================================================

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Setup
st.set_page_config(page_title="Netflix Data Analysis", layout="wide")
st.title("🎬 Netflix Data Analysis Dashboard")
import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Sample movie data with poster URLs
data = {
    'Title': ['Inception', 'Titanic', 'The Dark Knight', 'Avengers: Endgame', 'Interstellar','Raaz'],
    'Genre': ['Sci-Fi', 'Romance', 'Action', 'Action', 'Sci-Fi','Mystery'],
    'Year': [2010, 1997, 2008, 2019, 2014,2002],
    'Poster': [
        'https://www.mikeymo.nl/wp-content/uploads/2010/08/inception_poster.jpg',
        'https://m.media-amazon.com/images/I/71rNJQ2g-EL._AC_SL1178_.jpg',
        'https://m.media-amazon.com/images/I/91KkWf50SoL._SL1500_.jpg',
        'https://m.media-amazon.com/images/I/81ExhpBEbHL._AC_SL1500_.jpg',
        'https://m.media-amazon.com/images/I/91kFYg4fX3L._AC_SL1500_.jpg',
        'https://images-na.ssl-images-amazon.com/images/I/91jFV6-zr-L._RI_.jpg'
    ]
}

df = pd.DataFrame(data)

# Sidebar filters
st.sidebar.header("Search Movies")
search_term = st.sidebar.text_input("Movie Title")
genre_filter = st.sidebar.multiselect("Genre", df['Genre'].unique())
year_filter = st.sidebar.slider("Year", int(df['Year'].min()), int(df['Year'].max()), (int(df['Year'].min()), int(df['Year'].max())))

# Filter movies
filtered_df = df.copy()
if search_term:
    filtered_df = filtered_df[filtered_df['Title'].str.contains(search_term, case=False)]
if genre_filter:
    filtered_df = filtered_df[filtered_df['Genre'].isin(genre_filter)]
filtered_df = filtered_df[(filtered_df['Year'] >= year_filter[0]) & (filtered_df['Year'] <= year_filter[1])]

# Display movies in cards
st.header("Movies")
cols = st.columns(3)
for i, row in filtered_df.iterrows():
    with cols[i % 3]:
        st.image(row['Poster'], use_column_width=True)
        st.subheader(row['Title'])
        st.write(f"Genre: {row['Genre']}")
        st.write(f"Year: {row['Year']}")

# ---------------------------------------------
# 1️⃣ Load Dataset
# ---------------------------------------------
df = pd.read_csv(r"C:\Users\adity\OneDrive\Documents\Netflix Dataset.csv")

# Clean column names (lowercase, remove spaces)
df.columns = df.columns.str.strip().str.lower()

st.header("📊 Dataset Overview")
st.write("### Preview", df.head())
st.write("### Shape of Dataset:", df.shape)
st.write("### Columns:", list(df.columns))

# ---------------------------------------------
# 2️⃣ Data Cleaning
# ---------------------------------------------
# Replace missing values
for col in df.columns:
    df[col].fillna("Unknown", inplace=True)

# ---------------------------------------------
# 3️⃣ Basic Information
# ---------------------------------------------
st.header("📈 Basic Statistics & Insights")
st.write(df.describe(include='all'))

# ---------------------------------------------
# 4️⃣ Visualization 1 - Type of Content
# ---------------------------------------------
if 'type' in df.columns:
    st.subheader("🎞 Distribution of Movies vs TV Shows")

    # Count the values
    counts = df['type'].value_counts()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))  # slightly bigger figure for clarity
    sns.barplot(x=counts.index, y=counts.values, palette="coolwarm", ax=ax)

    # Add value labels on top of bars
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha='center', fontweight='bold', fontsize=12)

    # Clean style
    sns.despine(top=True, right=True)
    ax.set_xlabel("Type of Content", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Titles", fontsize=12, fontweight='bold')

    # Remove emoji from title to avoid missing glyph issues
    ax.set_title("Movies vs TV Shows on Netflix", fontsize=14, fontweight='bold')

    # Add grid for clarity
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    st.pyplot(fig)

# ---------------------------------------------
# 5️⃣ Visualization 2 - Top 10 Countries
# ---------------------------------------------
if 'country' in df.columns:
    st.subheader("🌍 Top 10 Countries with Most Titles")
    top_countries = df['country'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma', ax=ax2)
    ax2.set_xlabel("Number of Titles")
    ax2.set_ylabel("Country")
    st.pyplot(fig2)

# ---------------------------------------------
# 6️⃣ Visualization 3 - Content by Year
# ---------------------------------------------
if 'release_year' in df.columns:
    st.subheader("📅 Number of Titles Released Over the Years")
    year_count = df['release_year'].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    sns.lineplot(x=year_count.index, y=year_count.values, marker='o', ax=ax3)
    ax3.set_xlabel("Release Year")
    ax3.set_ylabel("Number of Titles")
    st.pyplot(fig3)

# ---------------------------------------------
# 7️⃣ Visualization 4 - Top 10 Genres
# ---------------------------------------------
if 'listed_in' in df.columns:
    st.subheader("🎭 Top 10 Genres on Netflix")
    genres = df['listed_in'].str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(10)
    fig4, ax4 = plt.subplots()
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax4)
    ax4.set_xlabel("Count")
    ax4.set_ylabel("Genre")
    st.pyplot(fig4)

# ---------------------------------------------
# 8️⃣ Visualization 5 - Ratings Distribution
# ---------------------------------------------
if 'rating' in df.columns:
    st.subheader("⭐ Distribution of Ratings")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index, palette='rocket', ax=ax5)
    ax5.set_xlabel("Count")
    ax5.set_ylabel("Rating")
    st.pyplot(fig5)

# ---------------------------------------------
# 9️⃣ Visualization 6 - Duration (Movies only)
# ---------------------------------------------
if 'duration' in df.columns and 'type' in df.columns:
    st.subheader("⏱ Distribution of Movie Durations")
    movies = df[df['type'].str.lower() == "movie"]
    movies['duration_mins'] = movies['duration'].str.replace(" min", "").replace("Unknown", "0").astype(int)
    fig6, ax6 = plt.subplots()
    sns.histplot(movies['duration_mins'], bins=20, kde=True, color='skyblue', ax=ax6)
    ax6.set_xlabel("Duration (minutes)")
    ax6.set_ylabel("Number of Movies")
    st.pyplot(fig6)

# ---------------------------------------------
# 🔍 Search Functionality
# ---------------------------------------------
st.header("🔍 Search Netflix Titles")
search_term = st.text_input("Enter a movie or show name:")
if search_term:
    results = df[df['title'].str.contains(search_term, case=False, na=False)]
    st.write(f"### Results for '{search_term}'")
    st.dataframe(results)
    st.success(f"Found {len(results)} matching titles.")

# ---------------------------------------------
# ℹ️ Dataset Info
# ---------------------------------------------
st.header("ℹ️ About This Dataset")
st.markdown("""
**Source:** Netflix Titles Dataset  
**Columns include:**
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
