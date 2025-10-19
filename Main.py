# ============================================================
# 🎬 NETFLIX DATA ANALYSIS PROJECT USING PYTHON
# ============================================================

# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set default styles
plt.style.use('seaborn-v0_8')
sns.set_palette('Set2')

# ============================================================
# 📂 2. Load Dataset
# ============================================================
# ============================================================
# 📂 2. Load Dataset
# ============================================================
df = pd.read_csv(r"C:\Users\adity\OneDrive\Documents\Netflix Dataset.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Rename alternative names to match expected schema
rename_map = {
    'category': 'type',
    'show_id': 'show_id',
    'title': 'title',
    'release year': 'release_year',
    'date added': 'date_added',
    'listed in': 'listed_in'
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

print("\n✅ Columns after normalization:")
print(df.columns.tolist())


# ============================================================
# 🧹 3. Data Cleaning
# ============================================================
# ============================================================
# 🧹 3. Data Cleaning (Robust Fix)
# ============================================================

# Show available columns to confirm structure
print("\nAvailable columns in dataset:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# ---- Safe handling of missing columns ----
# If a column doesn't exist, create it with default values

if 'director' not in df.columns:
    df['director'] = 'Unknown'
else:
    df['director'].fillna('Unknown', inplace=True)

if 'country' not in df.columns:
    df['country'] = 'Unknown'
else:
    df['country'].fillna('Unknown', inplace=True)

if 'rating' not in df.columns:
    df['rating'] = 'Unknown'
else:
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)

if 'date_added' not in df.columns:
    df['date_added'] = pd.NaT
else:
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract year_added for analysis (handle missing safely)
df['year_added'] = df['date_added'].dt.year

# Verify after cleaning
print("\n✅ Columns cleaned successfully!")
print(df.head())


# ============================================================
# 🔍 4. Exploratory Data Analysis (EDA)
# ============================================================

# 4.1 Movies vs TV Shows count
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='type', palette='coolwarm')
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# 4.2 Top 10 Countries with most titles
plt.figure(figsize=(8,6))
top_countries = df['country'].value_counts().head(10)
sns.barplot(y=top_countries.index, x=top_countries.values)
plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

# 4.3 Titles added per year
plt.figure(figsize=(10,5))
df['year_added'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Number of Titles Added per Year")
plt.xlabel("Year Added")
plt.ylabel("Count")
plt.show()

# 4.4 Top 10 genres
plt.figure(figsize=(10,6))
genres = df['listed_in'].dropna().str.split(',').explode().str.strip()
top_genres = genres.value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()

# 4.5 Ratings Distribution
plt.figure(figsize=(10,6))
sns.countplot(data=df, y='rating', order=df['rating'].value_counts().index)
plt.title("Distribution of Netflix Content Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()

# 4.6 Content Release Year Trend
plt.figure(figsize=(12,5))
sns.histplot(df['release_year'], bins=30, kde=True, color='coral')
plt.title("Content Release Year Distribution")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

# ============================================================
# 🎭 5. Deep Insights
# ============================================================

# 5.1 Most frequent directors
plt.figure(figsize=(10,6))
top_directors = df[df['director'] != 'Unknown']['director'].value_counts().head(10)
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title("Top 10 Directors with Most Netflix Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Director")
plt.show()

# 5.2 Most frequent actors
actors = df['cast'].dropna().str.split(',').explode().str.strip()
top_actors = actors.value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_actors.values, y=top_actors.index, color='lightgreen')
plt.title("Top 10 Most Frequent Actors on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Actor")
plt.show()

# 5.3 Movies vs TV Shows over the years
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='release_year', hue='type', palette='cool')
plt.title("Movies vs TV Shows Released Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.legend(title="Type")
plt.show()

# 5.4 Correlation between release year and added year
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='release_year', y='year_added', hue='type')
plt.title("Release Year vs Year Added to Netflix")
plt.xlabel("Release Year")
plt.ylabel("Year Added")
plt.show()

# ============================================================
# 📊 6. Interactive Visualizations (Plotly)
# ============================================================

# Content added by year
fig = px.histogram(df, x='year_added', color='type',
                   title="Content Added Each Year by Type",
                   barmode='group')
fig.show()

# Genre distribution interactive
genre_data = genres.value_counts().head(15)
fig2 = px.bar(x=genre_data.values, y=genre_data.index,
              title="Top 15 Genres on Netflix", orientation='h')
fig2.show()

# ============================================================
# 🧾 7. Key Insights Summary
# ============================================================

print("\n✅ PROJECT INSIGHTS SUMMARY ✅")
print("- Netflix has more Movies than TV Shows.")
print("- United States and India are top producers of Netflix content.")
print("- Peak content addition occurred between 2018–2020.")
print("- Most popular ratings are TV-MA and TV-14.")
print("- Top genres: Dramas, Comedies, and International Movies.")
print("- Directors like Raúl Campos and Marcus Raboy appear frequently.")
print("- Adam Sandler and Shah Rukh Khan are among the most featured actors.")
