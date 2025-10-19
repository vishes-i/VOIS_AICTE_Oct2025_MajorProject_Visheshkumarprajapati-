# Streamlit Netflix Data Analysis App
# Save this file as: netflix_app.py
# Run with: streamlit run netflix_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime

st.set_page_config(page_title="Netflix Data Analysis", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Custom CSS (make it attractive)
# ---------------------------
st.markdown("""
<style>
/* page background */
body {
    background: linear-gradient(135deg, #0f172a 0%, #0b1220 100%);
}
.section-box {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(2,6,23,0.6);
}
.nav-link {
    display:inline-block; padding:8px 14px; margin-right:8px; border-radius:10px; color:#cbd5e1; text-decoration:none;
}
.nav-link-active { background: linear-gradient(90deg,#ef4444,#f97316); color:white; }
.footer { color: #8b98a9; font-size:12px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------

def load_data(path_or_file="/mnt/data/Netflix Dataset.csv"):
    """
    Accepts a local path string or a file-like object (uploaded file).
    Returns DataFrame or None on failure.
    """
    try:
        df = pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Could not read the dataset at {path_or_file}. Error:\n{e}")
        return None
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trim column names
    df.columns = [c.strip() for c in df.columns]
    # Try parsing date columns heuristically
    for col in df.columns:
        if "date" in col.lower() or "added" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    # Normalize runtime/duration columns, create numeric duration if possible
    if 'duration' in df.columns:
        try:
            df['duration_num'] = df['duration'].astype(str).str.extract(r'(\d+)').astype(float)
        except Exception:
            # fallback: create NaN column
            df['duration_num'] = np.nan
    return df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Try to write an Excel file (openpyxl). If not possible, fallback to CSV bytes.
    Returns bytes.
    """
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='data')
        return output.getvalue()
    except Exception:
        # Fallback to CSV bytes
        output = BytesIO()
        output.write(df.to_csv(index=False).encode('utf-8'))
        return output.getvalue()


# ---------------------------
# Navigation (simple top nav simulated)
# ---------------------------
PAGES = ["Home", "Analysis", "Visuals", "Team", "Portfolio", "About", "Contact"]
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# top nav
cols = st.columns([1, 6, 1])
with cols[0]:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=120)
with cols[1]:
    # Use radio for navigation; works across Streamlit versions
    nav = st.radio("", PAGES, index=PAGES.index(st.session_state.page))
    st.session_state.page = nav
with cols[2]:
    # small utility buttons
    if st.button("🎧 Play demo audio"):
        st.audio("https://file-examples.com/storage/fe1a7ec2c3f6a8a2b5ef22a/2017/11/file_example_MP3_700KB.mp3")

# Sidebar for filters and dataset load
st.sidebar.title("Dataset & Filters")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
if uploaded:
    DATA_SOURCE = uploaded
else:
    DATA_SOURCE = "/mnt/data/Netflix Dataset.csv"

st.sidebar.markdown("---")
show_raw = st.sidebar.checkbox("Show raw dataset (first 200 rows)")

# Load dataset
with st.spinner('Loading data...'):
    df = load_data(DATA_SOURCE)

if df is None:
    st.stop()

# quick clean
df = clean_data(df)

# Sidebar filters auto-detected
st.sidebar.markdown("### Quick filters")
if 'type' in df.columns:
    types = df['type'].dropna().unique().tolist()
    default_types = types[:2] if len(types) >= 2 else types
    sel_type = st.sidebar.multiselect("Type", options=types, default=default_types)
    if sel_type:
        df = df[df['type'].isin(sel_type)]

if 'country' in df.columns:
    countries = df['country'].dropna().astype(str)
    if not countries.empty:
        top_countries = countries.value_counts().nlargest(20).index.tolist()
        default_countries = top_countries[:3] if len(top_countries) >= 3 else top_countries
        sel_country = st.sidebar.multiselect("Country (top 20)", options=top_countries, default=default_countries)
        if sel_country:
            df = df[df['country'].isin(sel_country)]

if 'release_year' in df.columns:
    # Guard against all-NaN or empty column
    yrs = df['release_year'].dropna().astype(int)
    if not yrs.empty:
        miny = int(yrs.min())
        maxy = int(yrs.max())
        yr = st.sidebar.slider("Release Year range", min_value=miny, max_value=maxy, value=(miny, maxy))
        df = df[(df['release_year'] >= yr[0]) & (df['release_year'] <= yr[1])]

st.sidebar.markdown("---")
if st.sidebar.button("Download filtered data"):
    towrite = to_excel_bytes(df)
    # Detect whether we wrote xlsx or csv by guessing header bytes (xlsx is binary with PK)
    if towrite[:2] == b'PK':
        b64 = base64.b64encode(towrite).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="netflix_filtered.xlsx">Download Excel file</a>'
    else:
        b64 = base64.b64encode(towrite).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="netflix_filtered.csv">Download CSV file</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# ---------------------------
# Page: Home
# ---------------------------
if st.session_state.page == 'Home':
    st.markdown("# Welcome to the Netflix Data Explorer :clapper:")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("""
        **Interactive analysis dashboard built with Streamlit.**

        Features:

        - Clean, filter and explore Netflix dataset.
        - Interactive charts (Plotly) — distributions, trends, and breakdowns.
        - Team, Portfolio, About & Contact pages for a full portfolio website feel.
        - Export filtered data, preview, and demo audio/video embeds.
        """)
        if show_raw:
            st.dataframe(df.head(200))
    with right:
        st.markdown("<div class='section-box'>\n**Quick stats**</div>", unsafe_allow_html=True)
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        if 'type' in df.columns:
            st.metric("Types", len(df['type'].dropna().unique()))
        st.video("https://file-examples.com/storage/fe1a7ec2c3f6a8a2b5ef22a/2017/04/file_example_MP4_480_1_5MG.mp4")

# ---------------------------
# Page: Analysis
# ---------------------------
elif st.session_state.page == 'Analysis':
    st.markdown("# Analysis & Data Cleaning")
    st.markdown("## Data sample and automated suggestions")
    st.dataframe(df.head(10))

    # Missing values summary
    st.markdown("### Missing values summary")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        miss_df = miss.reset_index()
        miss_df.columns = ['column', 'missing_ratio']
        fig = px.bar(miss_df, x='column', y='missing_ratio', title='Columns with missing data')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected (after loaded filters).")

    # Example transformation: create a binary column for 'is_movie'
    if 'type' in df.columns:
        try:
            df['is_movie'] = df['type'].astype(str).str.lower() == 'movie'
            st.markdown("Created column `is_movie` from `type`.")
        except Exception:
            pass

    # Top genres/countries
    st.markdown("### Top countries & directors")
    cols = st.columns(2)
    with cols[0]:
        if 'country' in df.columns and not df['country'].dropna().empty:
            top = df['country'].value_counts().nlargest(15).reset_index()
            top.columns = ['country', 'count']
            fig1 = px.bar(top, x='country', y='count', title='Top Countries')
            st.plotly_chart(fig1, use_container_width=True)
    with cols[1]:
        if 'director' in df.columns and not df['director'].dropna().empty:
            topd = df['director'].dropna().value_counts().nlargest(15).reset_index()
            topd.columns = ['director', 'count']
            fig2 = px.bar(topd, x='director', y='count', title='Top Directors')
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Page: Visuals
# ---------------------------
elif st.session_state.page == 'Visuals':
    st.markdown("# Visualizations")

    # Distribution of types
    if 'type' in df.columns and not df['type'].dropna().empty:
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        fig = px.pie(type_counts, names='type', values='count', title='Movies vs TV Shows')
        st.plotly_chart(fig, use_container_width=True)

    # Release year trend
    if 'release_year' in df.columns and not df['release_year'].dropna().empty:
        try:
            trend = df.groupby('release_year').size().reset_index(name='count')
            fig = px.line(trend, x='release_year', y='count', title='Content added per Release Year')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # Duration histogram
    if 'duration_num' in df.columns and not df['duration_num'].dropna().empty:
        fig = px.histogram(df, x='duration_num', nbins=40, title='Duration distribution (numeric)')
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: release_year vs duration
    if 'release_year' in df.columns and 'duration_num' in df.columns and \
       not df['release_year'].dropna().empty and not df['duration_num'].dropna().empty:
        hover_cols = ['title'] if 'title' in df.columns else None
        fig = px.scatter(df, x='release_year', y='duration_num', hover_data=hover_cols,
                         title='Duration vs Release Year')
        st.plotly_chart(fig, use_container_width=True)

    # Interactive filtered table
    st.markdown("### Interactive filtered table")
    sample_n = min(100, len(df))
    if sample_n > 0:
        st.dataframe(df.sample(sample_n))
    else:
        st.write("No rows to display.")

# ---------------------------
# Page: Team
# ---------------------------
elif st.session_state.page == 'Team':
    st.markdown("# Team & Contributors")
    st.markdown("Meet the team behind this project — use the 'Portfolio' page to link to each member's work.")
    team = [
        {"name": "Vishesh Kumar Prajapati", "role": "Data Scientist", "bio": "Engineer & ML enthusiast"},
        {"name": "Vishesh ", "role": "Frontend Dev", "bio": "Designs beautiful UI"},
        {"name": "Vishesh Kumar ", "role": "Data Engineer", "bio": "Pipelines & ETL"}
    ]
    cols = st.columns(3)
    for c, member in zip(cols, team):
        with c:
            # Use DiceBear initials avatar (works without URL encoding for most names)
            avatar_url = "https://avatars.dicebear.com/api/initials/" + member['name'].replace(' ', '') + ".svg"
            st.image(avatar_url, width=120)
            st.subheader(member['name'])
            st.caption(member['role'])
            st.write(member['bio'])
            if st.button(f"Contact {member['name']}"):
                st.info(f"Send an email to {member['name'].split()[0].lower()}@example.com")

# ---------------------------
# Page: Portfolio
# ---------------------------
elif st.session_state.page == 'Portfolio':
    st.markdown("# Portfolio & Deliverables")
    st.markdown("This section can include links to GitHub, deployed apps, videos, and downloadable reports.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Project Report")
        st.download_button("Download full report (example)", data=to_excel_bytes(df), file_name='netflix_report.xlsx')
    with col2:
        st.subheader("Demo Video")
        st.video("https://file-examples.com/storage/fe1a7ec2c3f6a8a2b5ef22a/2017/04/file_example_MP4_480_1_5MG.mp4")

# ---------------------------
# Page: About
# ---------------------------
elif st.session_state.page == 'About':
    st.markdown("# About this Project")
    st.markdown("This Netflix Data Analysis project was built as a showcase for exploratory data analysis, interactive visualizations, and an example portfolio website using Streamlit.")
    st.markdown("## Tech stack")
    st.markdown("- Python (pandas, plotly, streamlit)\n- Optional: scikit-learn for modeling, wordcloud for text visuals")

# ---------------------------
# Page: Contact
# ---------------------------
elif st.session_state.page == 'Contact':
    st.markdown("# Contact & Delivery")
    st.markdown("Use the form below to drop a note. (This demo writes to local session only.)")
    name = st.text_input("Your name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    if st.button("Send message"):
        if not name or not email or not message:
            st.warning("Please fill name, email, and message.")
        else:
            st.success("Message saved to local demo (not sent).")
            st.write({"name": name, "email": email, "message": message, "time": datetime.now().isoformat()})

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div class='footer'>Built with ❤️ using Streamlit — modify freely. Preview dataset path: <code>/mnt/data/Netflix Dataset.csv</code></div>", unsafe_allow_html=True)
