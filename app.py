"""
Newsletter Summarizer MVP

Run with:
    streamlit run app.py

This one-file prototype follows the roadmap MVP pipeline:
User Input -> Content Retrieval -> Preprocessing -> TF-IDF/Keyword Filtering
-> Content Selection -> Transformer Summarization -> Output Display
"""

from __future__ import annotations

import html
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import streamlit as st


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


@dataclass
class Article:
    """Small in-memory representation of a fetched RSS item."""

    title: str
    source: str
    link: str
    text: str
    score: float = 0.0


def clean_text(raw_text: str) -> str:
    """Remove HTML, entities, extra whitespace, and common RSS artifacts."""
    text = html.unescape(raw_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Simple sentence segmentation without requiring NLTK or spaCy."""
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


def parse_keywords(user_input: str) -> List[str]:
    """Turn comma/newline-separated user topics into normalized keywords."""
    chunks = re.split(r"[,\n]+", user_input.lower())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def build_topic_feed_urls(keywords: Sequence[str]) -> Tuple[str, ...]:
    """Create dynamic RSS search feeds from the user's requested topics."""
    feeds: List[str] = []

    # Fetch a combined query first, then each individual topic. This gives the
    # MVP topic-specific sources without API keys or a static feed catalog.
    queries = [" ".join(keywords), *keywords]
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            continue
        encoded_query = urllib.parse.quote_plus(normalized[:120])
        feed_url = GOOGLE_NEWS_RSS.format(query=encoded_query)
        if feed_url not in feeds:
            feeds.append(feed_url)

    return tuple(feeds)


def fetch_feed_with_feedparser(url: str) -> List[Article]:
    """Fetch RSS items with feedparser when it is installed."""
    import feedparser  # type: ignore

    parsed = feedparser.parse(url)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        raise ValueError(f"Could not parse feed: {url}")

    source = parsed.feed.get("title", url)
    articles: List[Article] = []
    for entry in parsed.entries[:20]:
        title = clean_text(entry.get("title", "Untitled"))
        summary = entry.get("summary", "") or entry.get("description", "")
        content = " ".join(part.get("value", "") for part in entry.get("content", []))
        text = clean_text(f"{title}. {summary} {content}")
        if text:
            articles.append(
                Article(
                    title=title,
                    source=source,
                    link=entry.get("link", ""),
                    text=text,
                )
            )
    return articles


def fetch_feed_with_stdlib(url: str) -> List[Article]:
    """Fetch RSS items using only the Python standard library."""
    request = urllib.request.Request(url, headers={"User-Agent": "NewsletterSummarizerMVP/1.0"})
    with urllib.request.urlopen(request, timeout=12) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    channel_title = root.findtext("./channel/title") or url
    items = root.findall(".//item")[:20]
    articles: List[Article] = []

    for item in items:
        title = clean_text(item.findtext("title") or "Untitled")
        description = clean_text(item.findtext("description") or "")
        link = clean_text(item.findtext("link") or "")
        text = clean_text(f"{title}. {description}")
        if text:
            articles.append(Article(title=title, source=channel_title, link=link, text=text))

    return articles


@st.cache_data(show_spinner=False, ttl=900)
def fetch_articles(feed_urls: Tuple[str, ...]) -> Tuple[List[Article], List[str]]:
    """Fetch articles from RSS feeds and collect readable fetch errors."""
    articles: List[Article] = []
    errors: List[str] = []

    for url in feed_urls:
        try:
            try:
                articles.extend(fetch_feed_with_feedparser(url))
            except ImportError:
                articles.extend(fetch_feed_with_stdlib(url))
        except (ET.ParseError, urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            errors.append(f"{url}: {exc}")

    return articles, errors


def keyword_score(text: str, keywords: Sequence[str]) -> float:
    """Fallback relevance score based on direct keyword matches."""
    lowered = text.lower()
    return float(sum(lowered.count(keyword) for keyword in keywords))


def rank_with_tfidf(articles: Sequence[Article], keywords: Sequence[str]) -> List[Article]:
    """Rank articles by cosine similarity between user topics and article text."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except ImportError:
        ranked = [
            Article(article.title, article.source, article.link, article.text, keyword_score(article.text, keywords))
            for article in articles
        ]
        return sorted(ranked, key=lambda article: article.score, reverse=True)

    query = " ".join(keywords)
    documents = [query] + [article.text for article in articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(documents)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = [
        Article(article.title, article.source, article.link, article.text, float(score))
        for article, score in zip(articles, scores)
    ]
    return sorted(ranked, key=lambda article: article.score, reverse=True)


def select_relevant_articles(
    articles: Sequence[Article], keywords: Sequence[str], max_articles: int
) -> List[Article]:
    """Filter and keep the most relevant articles for the summary context."""
    if not articles or not keywords:
        return []

    ranked = rank_with_tfidf(articles, keywords)
    relevant = [article for article in ranked if article.score > 0]

    # If TF-IDF finds no non-zero matches, keep a small fallback set so users get
    # a useful error-free demo from broad RSS content.
    if not relevant:
        relevant = ranked[:max_articles]

    return relevant[:max_articles]


def build_context(articles: Sequence[Article], max_chars: int = 7000) -> str:
    """Combine selected articles into a bounded summarization context."""
    chunks: List[str] = []
    total = 0
    for article in articles:
        chunk = f"{article.title}. {article.text}"
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load a small transformer summarizer if HuggingFace is available."""
    try:
        from transformers import pipeline  # type: ignore

        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def extractive_summary(text: str, keywords: Sequence[str], max_sentences: int = 8) -> str:
    """Deterministic fallback summary when transformers/model files are unavailable."""
    sentences = split_sentences(text)
    scored: List[Tuple[float, int, str]] = []

    for index, sentence in enumerate(sentences):
        score = keyword_score(sentence, keywords)
        # Add a small position bias because RSS summaries usually put key facts early.
        score += max(0.0, 1.0 - (index * 0.03))
        scored.append((score, index, sentence))

    best = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
    ordered = sorted(best, key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in ordered)


def summarize_context(text: str, keywords: Sequence[str], target_minutes: int) -> Tuple[str, str]:
    """Generate a concise summary and report which summarization method was used."""
    if not text.strip():
        return "", "No content"

    # Approximate 5-10 minute reading output. The transformer model has practical
    # token limits, so the MVP asks for a compact briefing rather than a long report.
    max_words = min(900, max(350, target_minutes * 130))
    summarizer = load_summarizer()

    if summarizer is None:
        fallback = extractive_summary(text, keywords, max_sentences=10)
        return textwrap.shorten(fallback, width=max_words * 6, placeholder="..."), "Extractive fallback"

    try:
        limited_text = text[:3500]
        result = summarizer(
            limited_text,
            max_length=min(450, max(180, target_minutes * 55)),
            min_length=90,
            do_sample=False,
        )
        return result[0]["summary_text"].strip(), "HuggingFace BART"
    except Exception:
        fallback = extractive_summary(text, keywords, max_sentences=10)
        return textwrap.shorten(fallback, width=max_words * 6, placeholder="..."), "Extractive fallback"


def render_article_list(articles: Iterable[Article]) -> None:
    """Display source articles used for transparency and citation."""
    for article in articles:
        with st.expander(f"{article.title} ({article.source})"):
            st.write(article.text[:800] + ("..." if len(article.text) > 800 else ""))
            if article.link:
                st.link_button("Open source", article.link)
            st.caption(f"Relevance score: {article.score:.3f}")


def main() -> None:
    st.set_page_config(page_title="Newsletter Summarizer MVP", page_icon="NEWS", layout="wide")

    st.title("Newsletter Summarizer")
    st.caption("MVP prototype: topic filtering plus AI-generated briefing from RSS/newsletter sources.")

    with st.sidebar:
        st.header("Sources")
        st.write("Sources are generated from your topics using Google News RSS search.")
        custom_feeds = st.text_area(
            "Optional extra RSS URLs",
            placeholder="https://example.com/feed.xml\nhttps://another-site.com/rss",
            height=100,
        )
        max_articles = st.slider("Articles to summarize", min_value=3, max_value=12, value=6)
        target_minutes = st.slider("Summary length target", min_value=5, max_value=10, value=7)

    topics = st.text_input(
        "Topics or keywords",
        placeholder="AI startups, California politics, energy markets",
    )

    if st.button("Generate summary", type="primary"):
        keywords = parse_keywords(topics)
        if not keywords:
            st.error("Enter at least one topic or keyword.")
            return

        feed_urls = list(build_topic_feed_urls(keywords))
        feed_urls.extend(url.strip() for url in custom_feeds.splitlines() if url.strip())

        if not feed_urls:
            st.error("Enter a searchable topic or add a custom RSS URL.")
            return

        with st.spinner("Fetching topic-specific newsletter/news items..."):
            articles, errors = fetch_articles(tuple(feed_urls))

        if errors:
            with st.expander("Some feeds could not be fetched"):
                for error in errors:
                    st.warning(error)

        if not articles:
            st.error("No articles were fetched. Try different sources or a custom RSS feed.")
            return

        relevant_articles = select_relevant_articles(articles, keywords, max_articles=max_articles)
        if not relevant_articles:
            st.error("No relevant articles were found for those topics.")
            return

        context = build_context(relevant_articles)

        with st.spinner("Generating summary..."):
            summary, method = summarize_context(context, keywords, target_minutes)

        if not summary:
            st.error("Relevant content was found, but the app could not generate a summary.")
            return

        st.subheader("Your briefing")
        st.write(summary)
        st.caption(f"Summary method: {method}")

        st.subheader("Articles used")
        render_article_list(relevant_articles)
    else:
        st.info("Enter topics, choose sources, and generate a concise newsletter briefing.")


if __name__ == "__main__":
    main()
