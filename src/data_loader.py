"""
data_loader.py
Loads and cleans the TED Talks dataset.
"""

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = ["description"]


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """
    Load a CSV of TED Talks into a cleaned DataFrame.

    Expected (flexible) columns:
        talk_id, title, main_speaker, speaker_occupation, event, tags,
        description, published_date, views

    The loader is forgiving: it will still work if some optional columns
    are missing, but 'description' must be present.
    """
    df = pd.read_csv(file)

    # Normalize column names (strip spaces, lowercase, underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows with empty description text
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() > 0]

    # Deduplicate identical descriptions (proxy for duplicate talks)
    df = df.drop_duplicates(subset=["description"])

    # Fill in optional columns if missing, so the rest of the app can rely on them
    if "title" not in df.columns:
        df["title"] = df["description"].str.slice(0, 60)
    if "main_speaker" not in df.columns:
        df["main_speaker"] = "Unknown"
    if "speaker_occupation" not in df.columns:
        df["speaker_occupation"] = "Unknown"
    if "event" not in df.columns:
        df["event"] = "Unknown"
    if "tags" not in df.columns:
        df["tags"] = ""

    # Parse dates if present
    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    else:
        df["published_date"] = pd.NaT

    # Coerce views to numeric
    if "views" in df.columns:
        df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    else:
        df["views"] = 0

    df["tags"] = df["tags"].fillna("").astype(str)

    df = df.reset_index(drop=True)
    return df


def get_all_tags(df: pd.DataFrame) -> list:
    """Flatten the comma-separated tags column into a sorted unique list."""
    exploded = (
        df["tags"].str.split(",").explode().str.strip().replace("", pd.NA).dropna()
    )
    return sorted(exploded.unique().tolist())


def get_filtered_data(
    df: pd.DataFrame,
    tags: list | None = None,
    occupations: list | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to the cleaned dataframe."""
    filtered = df.copy()

    if tags:
        pattern = "|".join(pd.Series(tags).str.replace(r"([.^$*+?{}\[\]\\|()])", r"\\\1", regex=True))
        filtered = filtered[filtered["tags"].str.contains(pattern, case=False, na=False)]

    if occupations:
        filtered = filtered[filtered["speaker_occupation"].isin(occupations)]

    if date_range and filtered["published_date"].notna().any():
        start, end = date_range
        filtered = filtered[
            (filtered["published_date"] >= pd.Timestamp(start))
            & (filtered["published_date"] <= pd.Timestamp(end))
        ]

    return filtered
