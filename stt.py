import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# === NEW FEATURE: MOOD MAPPING ===
# Simple mood mapping based on audio features (valence = happiness, energy = energy level)
MOOD_MAPPING = {
    "happy": {"valence_min": 0.6, "energy_min": 0.4},
    "sad": {"valence_max": 0.4, "energy_max": 0.5},
    "chill": {"valence_max": 0.6, "energy_max": 0.5, "danceability_min": 0.3},
    "energetic": {"energy_min": 0.7, "tempo_min": 120},
}

st.set_page_config(
    page_title="Spotify Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #000000; }
    section[data-testid="stSidebar"] .stRadio label { color: #B3B3B3; }
    section[data-testid="stSidebar"] .stRadio label:hover { color: #1DB954; }
    div[data-testid="stMetric"] { background: #181818; border-radius: 12px; padding: 16px; border: 1px solid #282828; }
    div[data-testid="stMetric"] label { color: #B3B3B3 !important; }
    div[data-testid="stMetric"] div { color: #FFFFFF !important; }
    .stButton > button { background-color: #1DB954; color: #FFFFFF; border-radius: 24px; border: none; font-weight: 700; padding: 8px 32px; }
    .stButton > button:hover { background-color: #1ed760; }
    .stDataFrame { background-color: #181818; }
    .stTextInput > div > div > input { background-color: #282828; color: #FFFFFF; border-radius: 24px; border: 1px solid #404040; padding: 8px 16px; }
    .stSelectbox > div > div > div { background-color: #282828; color: #FFFFFF; border-radius: 8px; }
    .stSlider > div > div > div > div { background-color: #1DB954; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #282828; color: #B3B3B3; border-radius: 8px; padding: 8px 24px; }
    .stTabs [aria-selected="true"] { background-color: #1DB954 !important; color: #FFFFFF !important; }
    .stExpander { border: 1px solid #282828; border-radius: 12px; }
    .stMultiSelect > div > div { background-color: #282828; border-radius: 8px; }
    .stAlert { background-color: #181818; border-radius: 8px; }
</style>""",
    unsafe_allow_html=True,
)

AUDIO_FEATURES = [
    "valence",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "popularity",
]
RADAR_FEATURES = [
    "valence",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
]
DARK = dict(paper_bgcolor="#181818", plot_bgcolor="#181818", font=dict(color="#B3B3B3"))


def dark_layout(fig, height=400, **kw):
    merged = {**DARK, **kw}
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **merged)
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    return fig


def build_radar_data(row):
    return {f: round(float(row[f]), 3) for f in RADAR_FEATURES}


def radar_chart(values_dict, title="Audio Profile"):
    cats = list(values_dict.keys())
    vals = list(values_dict.values())
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill="toself",
            line=dict(color="#1DB954", width=2),
            fillcolor="rgba(29,185,84,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#181818",
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="#333", linecolor="#333"
            ),
            angularaxis=dict(
                gridcolor="#333", linecolor="#333", tickfont=dict(color="#B3B3B3")
            ),
        ),
        **DARK,
        title=dict(text=title, font=dict(size=16)),
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig


def show_track_radar(row):
    return radar_chart(build_radar_data(row), title=f"{row['name']} – Audio Profile")


# === NEW FEATURE: AUDIO PLAYER HELPER ===
def get_audio_type(url_or_path):
    """Determine if the source is a YouTube link or audio file."""
    if pd.isna(url_or_path) or not url_or_path:
        return None
    if "youtube.com" in str(url_or_path) or "youtu.be" in str(url_or_path):
        return "youtube"
    elif url_or_path.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
        return "audio"
    return None


# === NEW FEATURE: ENHANCED DISPLAY WITH AUDIO PLAYER ===
def display_track_item(i, row, key_prefix, show_audio=True):
    c1, c2, c3 = st.columns([4, 2, 1])
    with c1:
        st.markdown(f"**{i + 1}. {row['name']}**")
        st.caption(
            f"{row['artists']}  |  {row['year']}  |  Pop: {int(row['popularity'])}"
        )
        # === NEW FEATURE: AUDIO PLAYER ===
        if show_audio and 'audio_url' in row.index:
            audio_type = get_audio_type(row.get('audio_url', ''))
            if audio_type == "youtube":
                with st.expander("🎬 Watch on YouTube"):
                    st.video(row['audio_url'])
            elif audio_type == "audio":
                with st.expander("🎧 Play Audio"):
                    st.audio(row['audio_url'])
    with c2:
        st.plotly_chart(
            radar_chart(build_radar_data(row), title=""),
            use_container_width=True,
            key=f"{key_prefix}_radar_{i}",
            height=200,
        )
    with c3:
        if st.button("➕", key=f"{key_prefix}_add_{i}", help="Add to playlist"):
            entry = {
                "name": row["name"],
                "artists": row["artists"],
                "year": int(row["year"]),
                "popularity": int(row["popularity"]),
            }
            if entry not in st.session_state.playlist:
                st.session_state.playlist.append(entry)
                st.success(f"✅ Added: {row['name']} ❤️")
            else:
                st.warning(f"⚠️ {row['name']} already in playlist!")


def get_song_recommendations(song_name, dataframe, scaled, n=10):
    matches = dataframe[dataframe["name"].str.contains(song_name, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame(), None
    idx = matches.index[0]
    features = scaled.iloc[idx].values.reshape(1, -1)
    sims = cosine_similarity(features, scaled)
    top_indices = sims.argsort()[0][::-1][1 : n + 1]
    return dataframe.iloc[top_indices], idx


def get_mood_recommendations(mood_dict, dataframe, scaled, features, sc, n=10):
    vec = pd.DataFrame([mood_dict], columns=features)
    vec_scaled = sc.transform(vec)
    knn = NearestNeighbors(n_neighbors=n + 1, algorithm="brute", metric="cosine")
    knn.fit(scaled)
    _, indices = knn.kneighbors(vec_scaled)
    return dataframe.iloc[indices[0][1 : n + 1]]


# === NEW FEATURE: MOOD FILTER FUNCTION ===
def filter_by_mood(dataframe, mood):
    """Filter songs based on mood using audio features."""
    if mood == "All":
        return dataframe
    
    filtered = dataframe.copy()
    
    if mood == "happy":
        filtered = filtered[(filtered['valence'] >= 0.6) & (filtered['energy'] >= 0.4)]
    elif mood == "sad":
        filtered = filtered[(filtered['valence'] <= 0.4) & (filtered['energy'] <= 0.5)]
    elif mood == "chill":
        filtered = filtered[(filtered['valence'] <= 0.6) & (filtered['energy'] <= 0.5) & (filtered['danceability'] >= 0.3)]
    elif mood == "energetic":
        filtered = filtered[(filtered['energy'] >= 0.7) & (filtered['tempo'] >= 120)]
    
    return filtered


# ── Data Loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("output.csv")
    df["release_date"] = pd.to_datetime(
        df["release_date"], errors="coerce", format="mixed"
    )
    df["year"] = df["release_date"].dt.year.fillna(0).astype(int)
    df.drop_duplicates(inplace=True)
    df["duration_min"] = (df["duration_ms"] / 60000).round(2)
    # === NEW FEATURE: Add audio_url column if not exists ===
    if "audio_url" not in df.columns:
        df["audio_url"] = None
    return df


@st.cache_resource
def build_scaler(df, audio_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[audio_features].values)
    return scaler, pd.DataFrame(X_scaled, columns=audio_features)


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "output.csv not found. Place your dataset in the same directory as this script."
    )
    st.stop()

scaler, X_scaled_df = build_scaler(df, AUDIO_FEATURES)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎵 Spotify Explorer\n---")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🔍 Search",
        "📊 Analytics",
        "📈 Trends",
        "🎯 Recommendations",
        "🎶 My Playlist",
    ],
)

if "playlist" not in st.session_state:
    st.session_state.playlist = []

st.sidebar.markdown("---")
# === NEW FEATURE: ENHANCED SIDEBAR INFO ===
st.sidebar.markdown("### 📋 Quick Stats")
st.sidebar.markdown(f"🎵 **Playlist:** {len(st.session_state.playlist)} tracks")
st.sidebar.markdown(f"📊 **Dataset:** {len(df):,} tracks")
st.sidebar.markdown("---")

# === NEW FEATURE: QUICK MOOD FILTER IN SIDEBAR ===
st.sidebar.markdown("### 🎭 Quick Mood Filter")
quick_mood = st.sidebar.selectbox(
    "Filter songs by mood",
    ["All", "😊 Happy", "😢 Sad", "😌 Chill", "⚡ Energetic"],
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎵 Spotify Music Explorer")
    st.markdown("### Explore, discover, and analyze music data interactively. ❤️")

    # === NEW FEATURE: MOOD FILTER ON HOME PAGE ===
    st.markdown("---")
    col_mood_filter = st.columns([1, 3])
    with col_mood_filter[0]:
        st.markdown("#### 🎭 Filter by Mood")
    with col_mood_filter[1]:
        selected_mood = st.selectbox(
            "Select your mood",
            ["All", "😊 Happy", "😢 Sad", "😌 Chill", "⚡ Energetic"],
            index=0,
            label_visibility="collapsed"
        )
    
    # Convert selection to simple mood name
    mood_filter = "All" if selected_mood == "All" else selected_mood.split()[1].lower()
    
    # Filter data based on mood
    if mood_filter != "All":
        filtered_df = filter_by_mood(df, mood_filter)
        st.info(f"🎭 Showing {len(filtered_df):,} **{mood_filter}** songs (filtered from {len(df):,} total)")
    else:
        filtered_df = df
    # ============================================================================

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tracks", f"{len(filtered_df):,}")
    c2.metric("Artists", f"{filtered_df['artists'].nunique():,}")
    yr = filtered_df[filtered_df["year"] > 0]["year"]
    c3.metric("Year Range", f"{int(yr.min())} – {int(yr.max())}" if len(yr) > 0 else "N/A")
    c4.metric("Avg Popularity", f"{filtered_df['popularity'].mean():.1f}")
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔥 Top 20 Most Popular")
        top = filtered_df.nlargest(20, "popularity")[
            ["name", "artists", "popularity", "year", "danceability", "energy"]
        ].reset_index(drop=True)
        top.index += 1
        st.dataframe(top, use_container_width=True, height=520)

    with col_r:
        st.markdown("### 🎤 Top Artists by Track Count")
        ac = filtered_df["artists"].value_counts().head(20).reset_index()
        ac.columns = ["Artist", "Tracks"]
        fig = px.bar(
            ac,
            x="Tracks",
            y="Artist",
            orientation="h",
            color="Tracks",
            color_continuous_scale="Greens",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            **DARK,
        )
        fig.update_xaxes(gridcolor="#333")
        fig.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---\n### 🎲 Random Discovery")
    if st.button("Shuffle 10 Random Tracks 🎲"):
        shuffle_df = filtered_df.sample(10)[
            [
                "name",
                "artists",
                "year",
                "popularity",
                "danceability",
                "energy",
                "valence",
            ]
        ].reset_index(drop=True)
        shuffle_df.index += 1
        st.dataframe(shuffle_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Search
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search":
    st.markdown("# 🔍 Search")
    
    # === NEW FEATURE: MOOD FILTER ON SEARCH ===
    col_search_mood = st.columns([1, 3])
    with col_search_mood[0]:
        st.markdown("#### 🎭 Mood")
    with col_search_mood[1]:
        search_mood = st.selectbox(
            "Filter by mood",
            ["All", "😊 Happy", "😢 Sad", "😌 Chill", "⚡ Energetic"],
            index=0,
            label_visibility="collapsed"
        )
    search_mood_filter = "All" if search_mood == "All" else search_mood.split()[1].lower()
    # ============================================================================
    
    search_type = st.radio("Search by", ["Song", "Artist"], horizontal=True)
    query = st.text_input("Enter name", placeholder="e.g. Bohemian Rhapsody or Queen")

    if query:
        if search_type == "Song":
            matches = df[df["name"].str.contains(query, case=False, na=False)]
            # === NEW FEATURE: APPLY MOOD FILTER ===
            if search_mood_filter != "All":
                matches = filter_by_mood(matches, search_mood_filter)
                if not matches.empty:
                    st.info(f"🎭 Filtered to **{search_mood_filter}** mood: {len(matches)} tracks found")
            # ============================================================================
            if matches.empty:
                st.warning(f"No song matching '{query}' found.")
            else:
                st.success(f"Found **{len(matches)}** track(s) matching **{query}** 🎵")
                for _, row in matches.head(10).iterrows():
                    with st.container():
                        st.markdown("---")
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.markdown(f"### 🎵 {row['name']}")
                            st.markdown(f"**Artist:** {row['artists']}")
                            st.markdown(
                                f"**Year:** {row['year']}  |  **Popularity:** {int(row['popularity'])}  |  **Duration:** {row['duration_min']} min"
                            )
                            st.markdown(
                                f"**Danceability:** {row['danceability']:.2f}  |  **Energy:** {row['energy']:.2f}  |  **Valence:** {row['valence']:.2f}  |  **Tempo:** {row['tempo']:.0f} BPM"
                            )
                            
                            # === NEW FEATURE: AUDIO URL INPUT & PLAYER ===
                            st.markdown("#### 🎧 Audio Source")
                            col_audio1, col_audio2 = st.columns([3, 1])
                            with col_audio1:
                                audio_url_input = st.text_input(
                                    "Enter audio URL (YouTube or MP3)",
                                    value=str(row.get('audio_url', '')) if pd.notna(row.get('audio_url')) else '',
                                    key=f"audio_url_{row.name}",
                                    placeholder="https://youtube.com/... or https://...mp3"
                                )
                            with col_audio2:
                                if audio_url_input:
                                    audio_type = get_audio_type(audio_url_input)
                                    if audio_type == "youtube":
                                        st.markdown("📺 YouTube detected")
                                        st.video(audio_url_input)
                                    elif audio_type == "audio":
                                        st.markdown("🎵 Audio detected")
                                        st.audio(audio_url_input)
                            # ============================================================================
                            
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("🎶 Get Similar Songs", key=f"sim_{row.name}"):
                                    recs, _ = get_song_recommendations(
                                        row["name"], df, X_scaled_df, 5
                                    )
                                    if not recs.empty:
                                        st.markdown("**Similar songs:**")
                                        for _, r in recs.iterrows():
                                            st.markdown(
                                                f"- **{r['name']}** by {r['artists']} (Pop: {int(r['popularity'])})"
                                            )
                            with col_btn2:
                                if st.button("➕ Add to Playlist", key=f"add_pl_{row.name}"):
                                    entry = {
                                        "name": row["name"],
                                        "artists": row["artists"],
                                        "year": int(row["year"]),
                                        "popularity": int(row["popularity"]),
                                    }
                                    if entry not in st.session_state.playlist:
                                        st.session_state.playlist.append(entry)
                                        st.success(f"✅ Added: {row['name']} ❤️")
                                    else:
                                        st.warning(f"⚠️ Already in playlist!")
                        with c2:
                            st.plotly_chart(
                                show_track_radar(row),
                                use_container_width=True,
                                key=f"radar_search_{row.name}",
                            )
        else:
            matches = df[df["artists"].apply(lambda x: query.lower() in str(x).lower())]
            # === NEW FEATURE: APPLY MOOD FILTER TO ARTIST SEARCH ===
            if search_mood_filter != "All":
                matches = filter_by_mood(matches, search_mood_filter)
                if not matches.empty:
                    st.info(f"🎭 Filtered to **{search_mood_filter}** mood: {len(matches)} tracks found")
            # ============================================================================
            if matches.empty:
                st.warning(f"No artist matching '{query}' found.")
            else:
                st.success(f"Found **{len(matches)}** track(s) for artist **{query}** 🎤")
                c1, c2, c3 = st.columns(3)
                valid_yr = matches[matches["year"] > 0]["year"]
                c1.metric("Total Tracks", len(matches))
                c2.metric("Avg Popularity", f"{matches['popularity'].mean():.1f}")
                c3.metric(
                    "Year Range",
                    f"{int(valid_yr.min())} – {int(matches['year'].max())}"
                    if len(valid_yr)
                    else "N/A",
                )

                st.markdown("### Average Audio Profile")
                avg_radar = {
                    f: round(float(matches[f].mean()), 3) for f in RADAR_FEATURES
                }
                st.plotly_chart(
                    radar_chart(avg_radar, title=f"{query} – Average Audio Profile"),
                    use_container_width=True,
                )

                st.markdown("### Tracks")
                adf = (
                    matches.sort_values("popularity", ascending=False)[
                        [
                            "name",
                            "year",
                            "popularity",
                            "danceability",
                            "energy",
                            "valence",
                        ]
                    ]
                    .head(20)
                    .reset_index(drop=True)
                )
                adf.index += 1
                st.dataframe(adf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Analytics
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("# 📊 Analytics")
    tab_dist, tab_corr, tab_explorer = st.tabs(
        ["Distributions", "Correlations", "Explorer"]
    )

    with tab_dist:
        feature = st.selectbox("Select feature", AUDIO_FEATURES, index=2)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df,
                x=feature,
                nbins=80,
                marginal="box",
                title=f"Distribution of {feature.title()}",
                color_discrete_sequence=["#1DB954"],
            )
            dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.violin(
                df,
                y=feature,
                box=True,
                points=False,
                title=f"Violin – {feature.title()}",
                color_discrete_sequence=["#1DB954"],
            )
            dark_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### All Features at a Glance")
        selected = st.multiselect(
            "Select features", AUDIO_FEATURES, default=AUDIO_FEATURES[:6]
        )
        if selected:
            cols = st.columns(min(len(selected), 3))
            for i, f in enumerate(selected):
                with cols[i % 3]:
                    fig = px.histogram(
                        df,
                        x=f,
                        nbins=50,
                        title=f.title(),
                        color_discrete_sequence=["#1DB954"],
                    )
                    dark_layout(fig, height=250, font=dict(color="#B3B3B3", size=11))
                    st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        corr = df.select_dtypes(include=["float64", "int64"]).corr()
        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix",
        )
        dark_layout(fig_heat, height=700)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### Feature Scatter Plot")
        sc1, sc2, sc3 = st.columns(3)
        x_feat = sc1.selectbox("X-axis", AUDIO_FEATURES, index=3)
        y_feat = sc2.selectbox("Y-axis", AUDIO_FEATURES, index=2)
        color_feat = sc3.selectbox(
            "Color by", ["popularity", "year", "energy", "danceability", "valence"]
        )

        sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig_sc = px.scatter(
            sample,
            x=x_feat,
            y=y_feat,
            color=color_feat,
            color_continuous_scale="Viridis",
            opacity=0.5,
            title=f"{x_feat.title()} vs {y_feat.title()}",
            hover_data=["name", "artists"],
        )
        dark_layout(fig_sc, height=550)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab_explorer:
        ex1, ex2, ex3 = st.columns(3)
        yr_min = int(df[df["year"] > 0]["year"].min())
        yr_max = int(df["year"].max())
        year_range = ex1.slider(
            "Year range", yr_min, yr_max, (yr_min, yr_max), key="ex_year"
        )
        pop_range = ex2.slider("Popularity range", 0, 100, (0, 100), key="ex_pop")
        explicit_filter = ex3.selectbox("Explicit", ["All", "Yes", "No"], key="ex_exp")

        filtered = df[
            (df["year"] >= year_range[0])
            & (df["year"] <= year_range[1])
            & (df["popularity"] >= pop_range[0])
            & (df["popularity"] <= pop_range[1])
        ]
        if explicit_filter == "Yes":
            filtered = filtered[filtered["explicit"] == 1]
        elif explicit_filter == "No":
            filtered = filtered[filtered["explicit"] == 0]

        st.write(f"Showing **{len(filtered):,}** tracks")
        st.dataframe(
            filtered.sort_values("popularity", ascending=False)[
                [
                    "name",
                    "artists",
                    "year",
                    "popularity",
                    "danceability",
                    "energy",
                    "valence",
                    "tempo",
                    "duration_min",
                ]
            ].head(100),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Trends
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown("# 📈 Music Trends Over Time")

    yearly = (
        df[df["year"] > 0]
        .groupby("year")
        .agg(
            count=("name", "count"),
            avg_popularity=("popularity", "mean"),
            avg_danceability=("danceability", "mean"),
            avg_energy=("energy", "mean"),
            avg_valence=("valence", "mean"),
            avg_acousticness=("acousticness", "mean"),
            avg_tempo=("tempo", "mean"),
            avg_loudness=("loudness", "mean"),
        )
        .reset_index()
    )

    st.markdown("### Tracks Released per Year")
    fig = px.bar(
        yearly,
        x="year",
        y="count",
        color="count",
        color_continuous_scale="Greens",
        labels={"count": "Tracks", "year": "Year"},
    )
    dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Audio Feature Trends")
    features_to_track = st.multiselect(
        "Select features to track",
        [
            "avg_danceability",
            "avg_energy",
            "avg_valence",
            "avg_acousticness",
            "avg_popularity",
        ],
        default=["avg_danceability", "avg_energy", "avg_valence"],
    )
    if features_to_track:
        fig2 = px.line(
            yearly,
            x="year",
            y=features_to_track,
            labels={"value": "Average", "year": "Year"},
        )
        dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### Average Tempo")
        fig3 = px.line(yearly, x="year", y="avg_tempo", labels={"avg_tempo": "BPM"})
        fig3.update_traces(line_color="#1DB954")
        dark_layout(fig3, height=350)
        st.plotly_chart(fig3, use_container_width=True)
    with col_t2:
        st.markdown("### Average Loudness")
        fig4 = px.line(
            yearly, x="year", y="avg_loudness", labels={"avg_loudness": "dB"}
        )
        fig4.update_traces(line_color="#E8115B")
        dark_layout(fig4, height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Popularity vs Year (Heatmap)")
    if len(df[df["year"] > 0]) > 0:
        fig5 = px.density_heatmap(
            df[df["year"] > 0],
            x="year",
            y="popularity",
            color_continuous_scale="Greens",
            nbinsx=40,
            nbinsy=30,
            title="Popularity Distribution Over Years",
        )
        dark_layout(fig5)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.markdown("# 🎯 Recommendations")
    tab_song, tab_mood = st.tabs(["Song-Based", "Mood-Based"])

    with tab_song:
        st.markdown("### Find Similar Songs")
        song_input = st.text_input(
            "Enter a song name", value="Bohemian Rhapsody", key="rec_song"
        )
        num_rec = st.slider("How many recommendations?", 3, 20, 5, key="song_n")

        if st.button("Get Recommendations", key="btn_song"):
            results, source_idx = get_song_recommendations(
                song_input, df, X_scaled_df, num_rec
            )
            if results.empty:
                st.warning(f"No match found for '{song_input}'.")
            else:
                st.success(
                    f"Songs similar to **{df.loc[source_idx, 'name']}** by {df.loc[source_idx, 'artists']}:"
                )

                col_src, col_radar = st.columns([2, 1])
                with col_src:
                    src = df.loc[source_idx]
                    st.markdown(f"#### 🎵 {src['name']}")
                    st.markdown(f"**Artist:** {src['artists']}")
                    st.markdown(
                        f"**Year:** {src['year']}  |  **Popularity:** {int(src['popularity'])}"
                    )
                with col_radar:
                    st.plotly_chart(
                        show_track_radar(src),
                        use_container_width=True,
                        key="radar_source",
                    )

                st.markdown("---\n### Recommended Tracks")
                for i, (_, row) in enumerate(results.iterrows()):
                    with st.container():
                        display_track_item(i, row, "song_rec")

    with tab_mood:
        st.markdown("### Find Songs by Mood")
        st.caption("Adjust the sliders to match your desired mood.")

        col_a, col_b = st.columns(2)
        with col_a:
            valence = st.slider("Valence (happiness)", 0.0, 1.0, 0.7, 0.05)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.05)
            energy = st.slider("Energy", 0.0, 1.0, 0.6, 0.05)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3, 0.05)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.05)
        with col_b:
            liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.05)
            loudness_norm = st.slider("Loudness (normalized)", 0.0, 1.0, 0.6, 0.05)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.05)
            tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0, 5.0)
            popularity = st.slider("Popularity", 0, 100, 50)

        mood_vals = dict(
            valence=valence,
            acousticness=acousticness,
            danceability=danceability,
            energy=energy,
            instrumentalness=instrumentalness,
            liveness=liveness,
            loudness=loudness_norm,
            speechiness=speechiness,
            tempo=tempo,
            popularity=popularity,
        )

        st.markdown("#### Your Mood Profile")
        mood_radar = {
            k: v
            for k, v in mood_vals.items()
            if k not in ("tempo", "loudness", "popularity")
        }
        st.plotly_chart(
            radar_chart(mood_radar, title="Mood Profile"), use_container_width=True
        )

        num_mood = st.slider("Number of recommendations", 3, 20, 5, key="mood_n")
        if st.button("Find Songs for This Mood"):
            results = get_mood_recommendations(
                mood_vals, df, X_scaled_df, AUDIO_FEATURES, scaler, num_mood
            )
            st.success(f"Found **{num_mood}** songs matching your mood:")
            for i, (_, row) in enumerate(results.iterrows()):
                with st.container():
                    display_track_item(i, row, "mood_rec")


# ══════════════════════════════════════════════════════════════════════════
# === NEW FEATURE: ENHANCED PLAYLIST PAGE ===
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎶 My Playlist":
    st.markdown("# 🎧 My Playlist ❤️")
    
    # Initialize playlist in session state if not exists
    if "playlist" not in st.session_state:
        st.session_state.playlist = []

    if not st.session_state.playlist:
        st.info("🎵 Your playlist is empty! Go to Search or Recommendations to add tracks.")
    else:
        # === NEW FEATURE: PLAYLIST METRICS ===
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        playlist_df = pd.DataFrame(st.session_state.playlist)
        col_stats1.metric("🎶 Total Tracks", len(st.session_state.playlist))
        col_stats2.metric("⭐ Avg Popularity", f"{playlist_df['popularity'].mean():.1f}")
        col_stats3.metric("🎤 Unique Artists", playlist_df['artists'].nunique())
        
        st.markdown("---")
        
        # === NEW FEATURE: INDIVIDUAL SONG CARDS WITH REMOVE BUTTON ===
        st.markdown("### 🎵 Your Songs")
        
        # Create columns for each song with remove button
        for i, track in enumerate(st.session_state.playlist):
            with st.container():
                col_track, col_action = st.columns([5, 1])
                with col_track:
                    st.markdown(f"**{i+1}. {track['name']}**")
                    st.caption(f"🎤 {track['artists']} | 📅 {track['year']} | ⭐ Pop: {track['popularity']}")
                with col_action:
                    if st.button("❌", key=f"remove_{i}", help=f"Remove {track['name']}"):
                        st.session_state.playlist.pop(i)
                        st.success(f"✅ Removed: {track['name']}")
                        st.rerun()
                st.markdown("---")
        
        # === NEW FEATURE: CLEAR ALL PLAYLIST BUTTON ===
        st.markdown("### ⚙️ Playlist Actions")
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🗑️ Clear Entire Playlist", type="secondary"):
                st.session_state.playlist = []
                st.success("✅ Playlist cleared!")
                st.rerun()
        with col_clear2:
            if st.download_button(
                "📥 Export Playlist (CSV)",
                data=playlist_df.to_csv(index=False),
                file_name="my_spotify_playlist.csv",
                mime="text/csv",
            ):
                st.success("✅ Playlist exported!")
        
        st.markdown("---")
        
        # === NEW FEATURE: PLAYLIST VISUALIZATION ===
        st.markdown("### 📊 Playlist Analytics")
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("#### ⭐ Popularity by Track")
            fig = px.bar(
                playlist_df,
                x="name",
                y="popularity",
                color="popularity",
                color_continuous_scale="Greens",
                labels={"name": "Track", "popularity": "Popularity"},
            )
            fig.update_layout(
                xaxis_tickangle=-45, 
                **DARK, 
                height=350, 
                margin=dict(l=10, r=10, t=30, b=10)
            )
            fig.update_xaxes(gridcolor="#333")
            fig.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_viz2:
            st.markdown("#### 🎤 Artist Distribution")
            artist_counts = playlist_df['artists'].value_counts().reset_index()
            artist_counts.columns = ["Artist", "Tracks"]
            fig2 = px.pie(
                artist_counts.head(8),
                values="Tracks",
                names="Artist",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig2.update_layout(**DARK, height=350)
            st.plotly_chart(fig2, use_container_width=True)
        
        # === NEW FEATURE: YEAR DISTRIBUTION ===
        st.markdown("#### 📅 Tracks by Year")
        year_counts = playlist_df['year'].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Tracks"]
        fig3 = px.bar(
            year_counts,
            x="Year",
            y="Tracks",
            color="Tracks",
            color_continuous_scale="Viridis",
        )
        fig3.update_layout(**DARK, height=300, margin=dict(l=10, r=10, t=30, b=10))
        fig3.update_xaxes(gridcolor="#333")
        fig3.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig3, use_container_width=True)
