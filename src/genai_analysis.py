"""
genai_analysis.py
Handles tone/sentiment classification, keyword extraction, and Q&A about the
TED Talks dataset. Supports two modes:

  - "genai"   : calls the Hugging Face Inference API (free tier, token-based)
  - "offline" : a free, local, lexicon-based fallback with no API calls at
                all, so the app is fully testable/demoable with zero setup.

Hugging Face setup:
  1. Create a token at https://huggingface.co/settings/tokens (a "Read" token
     with "Inference" permission is enough for the free Inference Providers API).
  2. Add it to `.streamlit/secrets.toml` as HF_TOKEN = "hf_..." (locally) or
     in Streamlit Community Cloud's Secrets settings (when deployed).

Models used (swap these constants if a model becomes unavailable on your
Hugging Face plan/provider):
  - SENTIMENT_MODEL: a dedicated 3-class sentiment classifier
  - CHAT_MODEL: an instruction-tuned chat model for the "Ask the Data" tab
"""

import re
import time
from collections import Counter
from typing import List, Dict

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SENTIMENT_PROVIDER = "hf-inference"  # this classifier is hosted on HF's own infra
CHAT_PROVIDER = "auto"  # let HF route to whichever partner (Cerebras, Together, etc.) hosts this model

_LABEL_MAP = {
    "negative": "cautionary",
    "neutral": "informative",
    "positive": "uplifting",
    "label_0": "cautionary",
    "label_1": "informative",
    "label_2": "uplifting",
}

# ---------------------------------------------------------------------------
# Hugging Face client
# ---------------------------------------------------------------------------

def get_client(provider: str = SENTIMENT_PROVIDER) -> InferenceClient:
    hf_token = st.secrets.get("HF_TOKEN", None)
    if not hf_token:
        st.error(
            "No HF_TOKEN found in Streamlit secrets. Create a token at "
            "https://huggingface.co/settings/tokens and add it to "
            "`.streamlit/secrets.toml` (locally) or Streamlit Community "
            "Cloud's Secrets settings, or switch to 'Offline (free demo)' "
            "mode in the sidebar."
        )
        st.stop()
    return InferenceClient(provider=provider, api_key=hf_token)


class GenAIUnavailableError(Exception):
    """Raised when the Hugging Face API call fails (auth, quota, model down, etc)."""


# ---------------------------------------------------------------------------
# GenAI (Hugging Face) mode
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def classify_tone_batch_genai(descriptions: List[str]) -> List[Dict]:
    """
    Calls the Hugging Face Inference API's text-classification task once per
    talk description using a general-purpose sentiment model, remapped to
    talk-appropriate labels (uplifting / cautionary / informative). Keyword
    extraction is done locally with a lightweight frequency method regardless
    of mode, since the free-tier classifier only returns a label + score.
    """
    client = get_client(provider=SENTIMENT_PROVIDER)
    results = []

    for text in descriptions:
        try:
            raw = client.text_classification(text, model=SENTIMENT_MODEL)
        except HfHubHTTPError as e:
            raise GenAIUnavailableError(
                "Hugging Face rejected the request. This usually means: the "
                "token is invalid/missing the 'Inference' permission, you've "
                "hit the free-tier rate limit, or the model is temporarily "
                "unavailable. Check your token at "
                "https://huggingface.co/settings/tokens, or switch to "
                "'Offline (free demo)' mode in the sidebar. "
                f"Details: {e}"
            ) from e
        except Exception as e:
            raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

        top_label = max(raw, key=lambda r: r["score"])["label"].lower()
        tone = _LABEL_MAP.get(top_label, "informative")

        keywords = _extract_keywords_local(text)
        results.append({"tone": tone, "keywords": keywords})

        time.sleep(0.05)  # light courtesy delay for the free tier

    return results


def answer_question_about_data_genai(question: str, df: pd.DataFrame) -> str:
    """RAG-lite chatbot using a Hugging Face chat/instruct model."""
    client = get_client(provider=CHAT_PROVIDER)

    total = len(df)
    up = int((df.get("tone") == "uplifting").sum()) if "tone" in df else None
    caut = int((df.get("tone") == "cautionary").sum()) if "tone" in df else None
    sample_talks = df[["title", "main_speaker", "description"]].sample(
        min(20, total), random_state=1
    )
    sample_lines = [
        f"- \"{row.title}\" by {row.main_speaker}: {row.description}"
        for row in sample_talks.itertuples()
    ]

    context = f"""
Dataset summary:
- Total talks in current view: {total}
- Uplifting: {up if up is not None else 'N/A'}
- Cautionary: {caut if caut is not None else 'N/A'}

Sample of up to 20 talks from the current filtered view:
{chr(10).join(sample_lines)}
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful analyst answering questions about a "
                "dataset of TED Talks. Base your answer only on the summary "
                "and sample talks provided. Be concise and specific."
            ),
        },
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]

    try:
        response = client.chat_completion(messages, model=CHAT_MODEL, max_tokens=400)
    except HfHubHTTPError as e:
        raise GenAIUnavailableError(
            "Hugging Face rejected the chat request. Check your token's "
            "'Inference' permission, try again in a moment (free-tier rate "
            f"limit), or switch to 'Offline (free demo)' mode. Details: {e}"
        ) from e
    except Exception as e:
        raise GenAIUnavailableError(f"Hugging Face API error: {e}") from e

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Shared local keyword extraction (used in both genai and offline modes)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "but", "of",
    "to", "in", "on", "for", "with", "was", "were", "are", "i", "you", "my",
    "at", "as", "be", "if", "so", "not", "very", "really", "just", "its",
    "have", "has", "had", "can", "will", "would", "than", "then", "there",
    "their", "they", "them", "after", "about", "into", "over", "amid",
    "talk", "talks", "shares", "shows", "makes", "step", "general",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _extract_keywords_local(text: str, top_n: int = 3) -> List[str]:
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 3]
    return [w for w, _ in Counter(tokens).most_common(top_n)]


# ---------------------------------------------------------------------------
# Offline (free, no API) fallback mode
# ---------------------------------------------------------------------------

_UPLIFTING_WORDS = {
    "hopeful", "joyful", "warmth", "humor", "remarkable", "breakthroughs",
    "celebrates", "revolution", "inspired", "inspiring", "transform",
    "better", "quiet",
}
_CAUTIONARY_WORDS = {
    "sobering", "warning", "risks", "stark", "confronts", "uncomfortable",
    "urgent", "crises", "troubling", "consequences", "neglecting",
}


def classify_tone_batch_offline(descriptions: List[str]) -> List[Dict]:
    """
    Free, local, lexicon-based tone classifier + naive keyword extraction.
    No API calls, no token needed. Good enough for demoing the app's UI/UX,
    not a substitute for real model-quality analysis.
    """
    results = []
    for text in descriptions:
        tokens = _tokenize(text)
        up_hits = sum(1 for t in tokens if t in _UPLIFTING_WORDS)
        caut_hits = sum(1 for t in tokens if t in _CAUTIONARY_WORDS)

        if up_hits > caut_hits:
            tone = "uplifting"
        elif caut_hits > up_hits:
            tone = "cautionary"
        else:
            tone = "informative"

        results.append({"tone": tone, "keywords": _extract_keywords_local(text)})
    return results


def answer_question_about_data_offline(question: str, df: pd.DataFrame) -> str:
    """
    Free, local, rule-based 'chatbot' fallback: answers a small set of
    common question patterns using stats computed directly from the data,
    instead of calling any API.
    """
    total = len(df)
    if total == 0:
        return "There are no talks in the current view to analyze."

    q = question.lower()
    has_tone = "tone" in df.columns

    if has_tone:
        counts = df["tone"].value_counts()
        up = int(counts.get("uplifting", 0))
        caut = int(counts.get("cautionary", 0))
        info = int(counts.get("informative", 0))
    else:
        up = caut = info = None

    if any(w in q for w in ["speaker", "who talks", "most popular speaker"]):
        if "main_speaker" in df.columns and "views" in df.columns:
            top_speaker = df.groupby("main_speaker")["views"].sum().sort_values(ascending=False).head(3)
            listing = ", ".join(f"{s} ({v:,} views)" for s, v in top_speaker.items())
            return f"[Offline mode] Top speakers by total views in this view: {listing}."

    if any(w in q for w in ["cautionary", "warning", "risk", "concern"]):
        if "keywords" in df.columns:
            caut_kw = (
                df.loc[df.get("tone") == "cautionary", "keywords"]
                .str.split(", ")
                .explode()
                .dropna()
            )
            caut_kw = caut_kw[caut_kw != ""]
            if not caut_kw.empty:
                top = ", ".join(caut_kw.value_counts().head(5).index.tolist())
                return f"[Offline mode] Most common cautionary-talk themes: {top}."
        return "[Offline mode] Run tone analysis first so I can identify cautionary themes."

    if any(w in q for w in ["popular tag", "most covered", "top topic", "which tag"]):
        if "tags" in df.columns:
            top_tags = (
                df["tags"].str.split(",").explode().str.strip().replace("", pd.NA).dropna()
                .value_counts().head(5)
            )
            listing = ", ".join(f"{t} ({n})" for t, n in top_tags.items())
            return f"[Offline mode] Most common tags in this view: {listing}."

    if any(w in q for w in ["trend", "over time", "growing", "views"]):
        if "published_date" in df.columns and df["published_date"].notna().any() and "views" in df.columns:
            trend = df.dropna(subset=["published_date"]).copy()
            trend["year"] = trend["published_date"].dt.year
            by_year = trend.groupby("year")["views"].mean().round(0)
            latest_year = by_year.index.max()
            return (
                f"[Offline mode] Average views per talk in {int(latest_year)}: "
                f"{int(by_year.loc[latest_year]):,} (out of {total} talks in this view)."
            )

    if up is not None:
        return (
            f"[Offline mode - no API used] Of {total} talks in the current view: "
            f"{up} uplifting, {caut} cautionary, {info} informative. "
            f"For richer, natural-language answers, add an HF_TOKEN and "
            f"switch to GenAI mode in the sidebar."
        )

    return (
        "[Offline mode] Run tone analysis first (sidebar) so I have "
        "tone data to answer questions about."
    )


# ---------------------------------------------------------------------------
# Unified entry points used by app.py
# ---------------------------------------------------------------------------

def classify_tone_batch(descriptions: List[str], mode: str = "genai") -> List[Dict]:
    if mode == "offline":
        return classify_tone_batch_offline(descriptions)
    return classify_tone_batch_genai(descriptions)


def answer_question_about_data(question: str, df: pd.DataFrame, mode: str = "genai") -> str:
    if mode == "offline":
        return answer_question_about_data_offline(question, df)
    return answer_question_about_data_genai(question, df)
