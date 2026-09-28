# 🎤 TED Talk Insights Explorer

A GenAI-powered Streamlit app that analyzes TED Talk descriptions for tone
(uplifting / cautionary / informative), keyword/topic extraction, and view
trends — with a built-in chatbot for asking questions about the dataset.

## Project Structure

```
ted_talks_insights_app/
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Python dependencies
├── README.md
├── data/
│   └── ted_talks_sample.csv            # Bundled SAMPLE dataset (synthetic, for demo/testing)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Loading + cleaning logic
│   └── genai_analysis.py               # Hugging Face API calls: tone, keywords, Q&A
└── .streamlit/
    └── secrets.toml                    # Template for your API key (keep out of git)
```

## 📊 Dataset source

This app is built around the **Kaggle "TED Talks" dataset** by Rounak
Banik — 2,550+ real TED Talks (2006–2017) with title, main speaker,
speaker occupation, event, tags, description, publish date, and view
count:

- **Dataset page:** https://www.kaggle.com/datasets/rounakbanik/ted-talks
- Not movie reviews, social media posts, or survey responses — it's
  conference-talk metadata and descriptions, which keeps the analysis
  (tone, topic/keyword themes, view trends over time) meaningfully
  different from a review-style app.

## ⚠️ About the bundled sample

`data/ted_talks_sample.csv` is a **synthetically generated sample** (480
rows across 10 topic areas: technology, climate, health, education,
psychology, business, science, design, social justice, personal growth)
that mirrors the structure of the real dataset, so you can run and test the
app immediately without downloading anything. **Before your final
submission/deployment, swap in the real dataset:**

1. Download `ted_main.csv` from
   https://www.kaggle.com/datasets/rounakbanik/ted-talks
2. Either update `default_path` in `app.py` to point to it, or just use the
   app's sidebar **file uploader** to load it at runtime (no code change needed).

The loader in `src/data_loader.py` is flexible about column names — it only
strictly requires a `description` column. If your real dataset uses
different column names, rename them to match:

| Expected column       | Description                                  |
|------------------------|-----------------------------------------------|
| `description`          | Summary of the talk (required)                |
| `title`                 | Talk title                                    |
| `main_speaker`          | Speaker's name                                |
| `speaker_occupation`    | Speaker's stated occupation                   |
| `event`                 | TED event name (e.g. TED2019)                 |
| `tags`                  | Comma-separated topic tags                    |
| `published_date`        | Publication date (YYYY-MM-DD)                 |
| `views`                 | View count                                    |

(The real Kaggle file uses `film_date`/`published_date` as Unix
timestamps and `tags` as a Python-list string — a short conversion
snippet is in the Troubleshooting section below.)

## Setup (local)

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Hugging Face access token:
   - Open `.streamlit/secrets.toml` and replace the placeholder with your
     real token.
   - Get a free token at https://huggingface.co/settings/tokens — click "New
     token", choose type "Read" (or "Fine-grained" with the "Make calls to
     Inference Providers" permission enabled), and copy the value starting
     with `hf_`.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How it works

1. **Load & clean** — `src/data_loader.py` loads the CSV with Pandas, strips
   empty/duplicate descriptions, normalizes column names, and coerces
   `views`/`published_date` to numeric/date types.
2. **Filter** — sidebar widgets let you filter by tag/topic, speaker
   occupation, and publication date range before running analysis (keeps
   things fast and light on the free tier).
3. **GenAI analysis** — `src/genai_analysis.py` sends each talk description
   to the Hugging Face Inference API using the
   `cardiffnlp/twitter-roberta-base-sentiment-latest` model, remapping its
   positive/neutral/negative output to talk-appropriate labels
   (uplifting/informative/cautionary), and extracts keywords locally via
   simple word-frequency analysis. Results are cached with
   `st.cache_data` so re-running the app doesn't re-call the API on the
   same input. A **chat model** (`meta-llama/Llama-3.1-8B-Instruct` by
   default) powers the "Ask the Data" tab.
4. **Offline fallback** — if you don't have a token yet, or hit a free-tier
   rate limit, switch to "Offline (free demo, no API)" mode in the sidebar.
   It uses a local lexicon-based classifier with zero API calls, so you can
   still test the whole app end to end.
5. **Visualize** — Plotly charts show tone distribution, average views by
   tone, average-views trend over time, and top extracted keywords.
6. **Ask the Data** — a chatbot tab feeds a summary + sample of the
   currently filtered talks to the model so you can ask free-form
   questions ("Who are the top speakers by views?").

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (make sure `.streamlit/secrets.toml`
   is in your `.gitignore` — never commit real tokens).
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Point it at your repo, branch, and `app.py` as the entry point.
4. In **App settings → Secrets**, paste:
   ```toml
   HF_TOKEN = "hf_your-real-token"
   ```
5. Deploy. Your app will be live at a `*.streamlit.app` URL.

## Troubleshooting

- **"No HF_TOKEN found"** — you haven't replaced the placeholder in
  `.streamlit/secrets.toml` locally (or added the secret in Streamlit
  Cloud) yet. See Setup step 3.
- **401/403 errors** — your token may lack the "Inference" permission.
  Regenerate it at https://huggingface.co/settings/tokens and make sure
  "Make calls to Inference Providers" is checked.
- **429 / rate limited** — the free Inference API tier has usage limits per
  hour. Wait a bit, lower "Max talks to analyze" in the sidebar, or switch
  to Offline mode to keep working in the meantime.
- **Model unavailable / cold start errors** — hosted free models occasionally
  scale to zero and take a few seconds to "warm up" on first call. The chat
  model uses `provider="auto"` so Hugging Face routes to whichever partner
  currently hosts it — if `CHAT_MODEL` in `src/genai_analysis.py` ever stops
  working, swap it for another instruction-tuned chat model listed at
  https://huggingface.co/models?inference_provider=all&other=conversational&sort=trending.
- **Converting the real Kaggle file:**
  ```python
  import pandas as pd, ast
  raw = pd.read_csv("ted_main.csv")
  raw["published_date"] = pd.to_datetime(raw["published_date"], unit="s")
  raw["tags"] = raw["tags"].apply(lambda s: ", ".join(ast.literal_eval(s)))
  raw.to_csv("data/ted_talks.csv", index=False)
  ```

## Next Goals (per assignment)

- ✅ Filters by tag/topic and speaker occupation are already included in the sidebar.
- ✅ A basic "Ask the Data" chatbot tab is included.
- Ideas to extend further:
  - Add a speaker leaderboard ranking by average views per talk.
  - Cache GenAI results to disk/DB so re-deploys don't re-analyze from scratch.
  - Add support for OpenAI or Anthropic's Claude API as an alternative model provider.
  - Weight tone aggregation by view count to see what resonates most.
