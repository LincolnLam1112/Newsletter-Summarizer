# Newsletter Summarizer MVP

A Streamlit app that generates personalized news summaries based on user topics.

## Features

* Dynamic news sources (Google News RSS)
* Topic-based filtering (TF-IDF)
* AI summarization (BART or fallback)

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example

Enter topics like:

* AI startups
* California politics

Get a concise 5–10 minute summary.

## Tech Stack

* Python
* Streamlit
* HuggingFace Transformers
* Scikit-learn
