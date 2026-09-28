"""
TED Talk Insights Explorer
A GenAI-powered Streamlit app for analyzing TED Talk descriptions.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_data, get_filtered_data, get_all_tags
from src.genai_analysis import (
    classify_tone_batch,
    answer_question_about_data,
    GenAIUnavailableError,
)

st.set_page_config(
    page_title="TED Talk Insights Explorer",
    page_icon="🎤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎤 TED Talk Insights Explorer")
st.caption(
    "Upload a dataset of TED Talks, run GenAI-powered tone analysis and "
    "keyword extraction, and explore trends in views and topics interactively."
)

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.header("1. Dataset")
default_path = "data/ted_talks_sample.csv"
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV of TED Talks", type=["csv"], help="Or leave empty to use the bundled sample dataset."
)

data_source = uploaded_file if uploaded_file is not None else default_path

try:
    df = load_data(data_source)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} talks.")

# ---------------------------------------------------------------------------
# Sidebar: filters
# ---------------------------------------------------------------------------
st.sidebar.header("2. Filters")

tag_options = get_all_tags(df)
selected_tags = st.sidebar.multiselect("Tag(s) / topic(s)", tag_options, default=[])

occupation_options = sorted(df["speaker_occupation"].dropna().unique().tolist())
selected_occupations = st.sidebar.multiselect("Speaker occupation(s)", occupation_options, default=[])

date_range = None
if df["published_date"].notna().any():
    min_date = df["published_date"].min().date()
    max_date = df["published_date"].max().date()
    date_range = st.sidebar.date_input("Publication date range", (min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) != 2:
        date_range = None

filtered_df = get_filtered_data(
    df,
    tags=selected_tags or None,
    occupations=selected_occupations or None,
    date_range=date_range,
)

st.sidebar.markdown(f"**{len(filtered_df):,}** talks match current filters.")

# ---------------------------------------------------------------------------
# Sidebar: run analysis
# ---------------------------------------------------------------------------
st.sidebar.header("3. GenAI Analysis")
analysis_mode_label = st.sidebar.radio(
    "Analysis mode",
    ["GenAI (Hugging Face API)", "Offline (free demo, no API)"],
    index=0,
    help=(
        "GenAI mode requires a Hugging Face access token (free to create). "
        "Offline mode uses a local keyword-based classifier so you can test "
        "the app with no token/setup at all."
    ),
)
analysis_mode = "genai" if analysis_mode_label.startswith("GenAI") else "offline"

n_available = len(filtered_df)

if n_available == 0:
    st.sidebar.warning("No talks match the current filters — adjust filters above to enable analysis.")
    sample_cap = 0
    run_analysis = False
    st.sidebar.button("Run Tone Analysis", type="primary", disabled=True)
elif n_available <= 20:
    st.sidebar.caption(f"Only {n_available} talk(s) match filters — analyzing all of them.")
    sample_cap = n_available
    run_analysis = st.sidebar.button("Run Tone Analysis", type="primary")
else:
    slider_max = min(500, n_available)
    default_val = min(150, slider_max)
    sample_cap = st.sidebar.slider(
        "Max talks to analyze (controls API cost)", 20, slider_max, default_val
    )
    run_analysis = st.sidebar.button("Run Tone Analysis", type="primary")

if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = analysis_mode

if run_analysis:
    if filtered_df.empty:
        st.warning("No talks match the current filters.")
    else:
        spinner_msg = (
            "Calling Hugging Face API to analyze talk descriptions..."
            if analysis_mode == "genai"
            else "Running free offline keyword-based analysis..."
        )
        with st.spinner(spinner_msg):
            work_df = filtered_df.sample(
                min(sample_cap, len(filtered_df)), random_state=1
            ).reset_index(drop=True)
            try:
                results = classify_tone_batch(
                    work_df["description"].tolist(), mode=analysis_mode
                )
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: switch 'Analysis mode' to 'Offline (free demo, no API)' "
                    "in the sidebar to keep testing without spending API credits."
                )
                st.stop()
            work_df["tone"] = [r.get("tone", "informative") for r in results]
            work_df["keywords"] = [", ".join(r.get("keywords", [])) for r in results]
            st.session_state.analyzed_df = work_df
            st.session_state.analysis_mode = analysis_mode
        mode_note = "GenAI" if analysis_mode == "genai" else "offline (free demo)"
        st.success(f"Analyzed {len(work_df):,} talks using {mode_note} mode.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
tab_overview, tab_viz, tab_chat = st.tabs(["📄 Data Preview", "📊 Visualizations", "💬 Ask the Data"])

with tab_overview:
    st.subheader("Filtered Talk Sample")
    st.dataframe(
        filtered_df[["title", "main_speaker", "speaker_occupation", "tags", "views", "published_date"]].head(50),
        use_container_width=True,
    )

with tab_viz:
    if st.session_state.analyzed_df is None:
        st.info("Run tone analysis from the sidebar to see visualizations here.")
    else:
        adf = st.session_state.analyzed_df
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tone Distribution")
            counts = adf["tone"].value_counts().reset_index()
            counts.columns = ["tone", "count"]
            fig = px.bar(
                counts, x="tone", y="count", color="tone",
                color_discrete_map={"uplifting": "#2ecc71", "cautionary": "#e74c3c", "informative": "#3498db"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Views by Tone")
            views_by_tone = adf.groupby("tone")["views"].mean().reset_index()
            fig2 = px.bar(
                views_by_tone, x="tone", y="views", color="tone",
                color_discrete_map={"uplifting": "#2ecc71", "cautionary": "#e74c3c", "informative": "#3498db"},
            )
            fig2.update_layout(yaxis_title="Average views")
            st.plotly_chart(fig2, use_container_width=True)

        if adf["published_date"].notna().any():
            st.subheader("Average Views Trend Over Time")
            trend = adf.dropna(subset=["published_date"]).copy()
            trend["year"] = trend["published_date"].dt.year
            trend_counts = trend.groupby(["year", "tone"])["views"].mean().reset_index()
            fig3 = px.line(
                trend_counts, x="year", y="views", color="tone", markers=True,
                color_discrete_map={"uplifting": "#2ecc71", "cautionary": "#e74c3c", "informative": "#3498db"},
            )
            fig3.update_layout(yaxis_title="Average views")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Top Extracted Keywords")
        kw_series = adf["keywords"].str.split(", ").explode().dropna()
        kw_series = kw_series[kw_series != ""]
        top_kw = kw_series.value_counts().head(15).reset_index()
        top_kw.columns = ["keyword", "count"]
        fig4 = px.bar(top_kw, x="count", y="keyword", orientation="h")
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("See analyzed talks with tone + keywords"):
            st.dataframe(
                adf[["title", "main_speaker", "tone", "keywords", "views"]],
                use_container_width=True,
            )

with tab_chat:
    st.subheader("Ask questions about the talks")
    st.caption(
        "Ask things like 'Which tags get the most coverage?' or "
        "'Who are the top speakers by views?' Answers use the currently "
        "analyzed/filtered talks as context."
    )
    context_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else filtered_df
    chat_mode = st.session_state.get("analysis_mode", analysis_mode)
    st.caption(f"Currently answering in **{'GenAI' if chat_mode == 'genai' else 'offline (free demo)'}** mode.")
    user_question = st.text_input("Your question")
    if st.button("Ask") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = answer_question_about_data(user_question, context_df, mode=chat_mode)
            except GenAIUnavailableError as e:
                st.error(str(e))
                st.info(
                    "Tip: switch 'Analysis mode' to 'Offline (free demo, no API)' "
                    "in the sidebar to keep testing without spending API credits."
                )
                st.stop()
        st.markdown(f"**Answer:** {answer}")

st.divider()
st.caption(
    "Built with Streamlit + Hugging Face for a GenAI dataset-analysis assignment. "
    "Swap in your own TED Talks CSV via the sidebar uploader."
)
