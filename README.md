# Multi-Model LLM Tokenization Comparison System

A Streamlit dashboard that compares how **GPT-4o**, **Gemini**, and **Qwen 2.5**
tokenize the same input text — showing token counts, estimated cost, and
tokenization efficiency side by side.

```
INPUT       -> user enters text in Streamlit
PROCESSING  -> a Request Orchestrator checks a cache first
               - cache hit  : return the stored result (no API cost)
               - cache miss : call the three tokenizers and store the result
OUTPUT      -> Streamlit dashboard with cards, tables, and Altair charts
```

> **Model change note:** the original proposal used Claude, but Anthropic's
> API has no free tier, so the free open-weight **Qwen 2.5** tokenizer replaces
> it (runs 100% locally). The old `adapters/claude_tok.py` is kept, disabled,
> for reference.

---

## Project structure

```
tokencompare/
├── app.py                  # Streamlit UI (single input + batch mode)
├── components.py           # reusable cards / chips / charts / tables
├── styles.py               # centralised CSS + model brand colours
├── orchestrator.py         # request orchestrator + SQLite cache
├── adapters/               # tokenizer adapters (one per model)
│   ├── __init__.py
│   ├── openai_tok.py       # GPT-4o   (tiktoken, local)
│   ├── gemini_tok.py       # Gemini   (google-genai SDK)
│   ├── qwen_tok.py         # Qwen 2.5 (Hugging Face transformers, local)
│   └── claude_tok.py       # Claude   (disabled, kept for reference)
├── cost.py                 # cost estimation (reads pricing.yaml)
├── pricing.yaml            # per-model prices / metadata
├── samples/sample_texts.csv
├── tests/                  # pytest tests (offline)
├── requirements.txt
├── .streamlit/
│   ├── config.toml         # custom theme
│   └── secrets.toml.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Virtual environment + dependencies

```bash
# Windows
py -m venv venv
venv\Scripts\Activate.ps1        # PowerShell  (cmd: venv\Scripts\activate)
pip install -r requirements.txt

# macOS / Linux
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Gemini API key (only key needed)

GPT-4o and Qwen 2.5 run locally. Copy the example secrets file and add your
key:

```bash
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

```toml
GEMINI_API_KEY = "AIza..."   # from https://aistudio.google.com/apikey
```

### 3. Prices (see "Prices to fill in" below)

Edit `pricing.yaml`.

---

## Run

```bash
streamlit run app.py
# or, without activating the venv:
venv\Scripts\python.exe -m streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Run the tests

```bash
py -m pytest -v
```

All offline (Gemini and Qwen are mocked; GPT-4o uses local tiktoken).

---

## Prices to fill in (`pricing.yaml`)

| Model | What to set | Official source |
|-------|-------------|-----------------|
| `gpt-4o` | `input_per_million` + `price_date` | https://openai.com/api/pricing/ — the **gpt-4o** row, **"Input"** price (per 1M tokens) |
| `gemini-3.8-flash` | `input_per_million` + `price_date` | https://ai.google.dev/gemini-api/docs/pricing — the **Flash** model **"Input"** price (per 1M tokens) |
| `qwen2.5-0.5b` | leave as-is | open-source → the UI shows **"Free (self-hosted)"** |

Until you set them, GPT-4o and Gemini show **"Price not set"** (with a warning)
instead of a fake `$0.00`.

---

## How it works

- **Adapters** (`adapters/`) each return
  `{"model", "tokens", "pieces", "token_ids"}`. GPT-4o and Qwen 2.5 run locally
  and return token pieces + IDs (used for the token breakdown and chip
  tooltips); Gemini is count-only.
- **Orchestrator** (`orchestrator.py`) — SQLite cache keyed by
  `sha256(model | text)`, per-model `try/except`, and running hit/miss counters.
- **Cost** (`cost.py`) — `cost = tokens / 1_000_000 * input_per_million`, with
  an "open-source → free" special case and "price not set" detection.
- **UI** (`app.py` + `components.py` + `styles.py`) — branded metric cards,
  winner badges, token chips with tooltips, per-model-coloured Altair charts,
  a cost-basis selector (single / per 1k / per 1M requests), and a batch tab.
