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


# Bing works better than Google News here because the RSS links expose publisher URLs
BING_NEWS_RSS = "https://www.bing.com/news/search?q={query}&format=rss"


@dataclass
class Article:
    """Small in-memory representation of a fetched RSS item."""

    # keeping article data together makes the pipeline easier to pass around
    title: str
    source: str
    link: str
    text: str
    score: float = 0.0


MIN_FULL_ARTICLE_WORDS = 80


# clean scraped/RSS text before filtering or summarizing it
def clean_text(raw_text: str) -> str:
    """Remove HTML, entities, extra whitespace, and common RSS artifacts."""
    text = html.unescape(raw_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# split text into rough sentences without adding another dependency
def split_sentences(text: str) -> List[str]:
    """Simple sentence segmentation without requiring NLTK or spaCy."""
    # this is not perfect NLP, but it is good enough for an MVP fallback
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


# normalize a sentence so we can spot repeated titles and near-duplicates
def normalize_for_similarity(text: str) -> set[str]:
    """Create a small word set for lightweight duplicate checks."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    # short words add noise, so keep the words that are more likely to matter
    return {word for word in words if len(word) > 3}


# count real words so short headlines do not get treated like full articles
def word_count(text: str) -> int:
    """Count words in cleaned text."""
    return len(re.findall(r"\b\w+\b", clean_text(text)))


# remove the article title from the body when RSS text already includes it
def remove_repeated_title(title: str, text: str) -> str:
    """Avoid feeding the same headline to the summarizer twice."""
    cleaned_title = clean_text(title)
    cleaned_text = clean_text(text)
    if not cleaned_title or not cleaned_text:
        return cleaned_text

    # RSS snippets often start with the exact headline, so strip that out
    title_pattern = re.escape(cleaned_title)
    without_title = re.sub(rf"^{title_pattern}\s*[.:-]?\s*", "", cleaned_text, flags=re.IGNORECASE)
    return without_title.strip()


# get the best article body we have without letting the headline dominate
def article_body_for_summary(article: Article) -> str:
    """Prepare article text for summarization, not headline matching."""
    body = remove_repeated_title(article.title, article.text)
    # remove source names that sometimes get tacked onto the end of snippets
    body = re.sub(r"\s+-\s+[A-Z][A-Za-z0-9 .&'-]{2,}$", "", body)
    return clean_text(body)


# try to pull the real article body from the article URL
def extract_full_article_text(url: str) -> str:
    """Use newspaper3k for full text, but fail quietly for blocked pages."""
    if not url:
        return ""

    try:
        from newspaper import Article as NewspaperArticle  # type: ignore
        from newspaper import Config  # type: ignore

        # newspaper3k is the main path for getting real article text
        config = Config()
        config.browser_user_agent = "Mozilla/5.0"
        config.request_timeout = 8
        article = NewspaperArticle(url, config=config)
        article.download()
        article.parse()
        text = clean_text(article.text)
        # if it only got a tiny snippet, treat it as a failed scrape
        if word_count(text) >= MIN_FULL_ARTICLE_WORDS:
            return text
    except Exception:
        # lots of news sites block scraping, so do not let one article kill the app
        pass

    try:
        from bs4 import BeautifulSoup  # type: ignore

        # backup scraper: grab paragraph text directly from the page
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            html_data = response.read()

        soup = BeautifulSoup(html_data, "html.parser")
        # remove page chrome so nav/footer text does not pollute the article
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paragraphs = [
            clean_text(paragraph.get_text(" "))
            for paragraph in soup.find_all("p")
        ]
        # only keep paragraphs that look like real content
        text = " ".join(paragraph for paragraph in paragraphs if word_count(paragraph) >= 8)
        text = clean_text(text)
        if word_count(text) >= MIN_FULL_ARTICLE_WORDS:
            return text
    except Exception:
        return ""

    return ""


# turn news-search redirect links into publisher URLs when the feed provides one
def resolve_article_url(url: str) -> str:
    """Pull the real publisher URL out of Bing News redirect links."""
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    # Bing RSS wraps links in an apiclick URL, with the real URL in ?url=
    if "url" in query and query["url"]:
        return query["url"][0]
    return url


# prefer full article text, but keep the RSS snippet as a fallback
def choose_article_text(link: str, rss_text: str) -> str:
    """Pick the richest text available for the article."""
    # full text gives much better summaries than RSS snippets
    full_text = extract_full_article_text(resolve_article_url(link))
    if full_text:
        return full_text
    # fallback in case the publisher blocks scraping
    return clean_text(rss_text)


# turn what the user typed into a clean list of topics
def parse_keywords(user_input: str) -> List[str]:
    """Turn comma/newline-separated user topics into normalized keywords."""
    chunks = re.split(r"[,\n]+", user_input.lower())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


# build dynamic news RSS URLs from user topics
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
        # encode spaces and special characters so the query works inside a URL
        encoded_query = urllib.parse.quote_plus(normalized[:120])
        feed_url = BING_NEWS_RSS.format(query=encoded_query)
        if feed_url not in feeds:
            feeds.append(feed_url)

    return tuple(feeds)


# use feedparser if it exists because it handles messy feeds better
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
        # some feeds put the main body in content instead of summary
        content = " ".join(part.get("value", "") for part in entry.get("content", []))
        # resolve before saving so links open on the real publisher site
        link = resolve_article_url(entry.get("link", ""))
        rss_text = clean_text(f"{summary} {content}") or title
        # try full article content first, then use the RSS snippet if needed
        text = choose_article_text(link, rss_text)
        if text:
            articles.append(
                Article(
                    title=title,
                    source=source,
                    link=link,
                    text=text,
                )
            )
    return articles


# fallback RSS parser using only the Python standard library
def fetch_feed_with_stdlib(url: str) -> List[Article]:
    """Fetch RSS items using only the Python standard library."""
    # a user-agent helps avoid basic blocks from some RSS providers
    request = urllib.request.Request(url, headers={"User-Agent": "NewsletterSummarizerMVP/1.0"})
    with urllib.request.urlopen(request, timeout=12) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    channel_title = root.findtext("./channel/title") or url
    # standard RSS uses item nodes for individual stories
    items = root.findall(".//item")[:20]
    articles: List[Article] = []

    for item in items:
        title = clean_text(item.findtext("title") or "Untitled")
        description = clean_text(item.findtext("description") or "")
        link = resolve_article_url(clean_text(item.findtext("link") or ""))
        rss_text = clean_text(description) or title
        # same full-text-first approach as the feedparser path
        text = choose_article_text(link, rss_text)
        if text:
            articles.append(Article(title=title, source=channel_title, link=link, text=text))

    return articles


# fetch all feeds and keep errors instead of crashing the app
@st.cache_data(show_spinner=False, ttl=900)
def fetch_articles(feed_urls: Tuple[str, ...]) -> Tuple[List[Article], List[str]]:
    """Fetch articles from RSS feeds and collect readable fetch errors."""
    articles: List[Article] = []
    errors: List[str] = []

    for url in feed_urls:
        try:
            try:
                # prefer feedparser because it handles RSS weirdness better
                articles.extend(fetch_feed_with_feedparser(url))
            except ImportError:
                # still works without feedparser, just less robust
                articles.extend(fetch_feed_with_stdlib(url))
        except (ET.ParseError, urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            # show feed failures in the UI instead of crashing everything
            errors.append(f"{url}: {exc}")

    return articles, errors


# simple keyword match when sklearn is not available
def keyword_score(text: str, keywords: Sequence[str]) -> float:
    """Fallback relevance score based on direct keyword matches."""
    lowered = text.lower()
    return float(sum(lowered.count(keyword) for keyword in keywords))


# rank articles against the user's topics with TF-IDF when possible
def rank_with_tfidf(articles: Sequence[Article], keywords: Sequence[str]) -> List[Article]:
    """Rank articles by cosine similarity between user topics and article text."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except ImportError:
        # keep the app usable even if sklearn was not installed
        ranked = [
            Article(
                article.title,
                article.source,
                article.link,
                article.text,
                keyword_score(f"{article.title} {article.text}", keywords),
            )
            for article in articles
        ]
        return sorted(ranked, key=lambda article: article.score, reverse=True)

    query = " ".join(keywords)
    # include the title for relevance ranking, but not as the main summary input
    documents = [query] + [f"{article.title}. {article.text}" for article in articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(documents)
    # compare the topic query vector against every article vector
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = [
        Article(article.title, article.source, article.link, article.text, float(score))
        for article, score in zip(articles, scores)
    ]
    return sorted(ranked, key=lambda article: article.score, reverse=True)


# keep only the articles that look most relevant to the user's topics
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


# combine selected article text into one chunk for the summarizer
def build_context(articles: Sequence[Article], max_chars: int = 12000) -> str:
    """Combine selected articles into a bounded summarization context."""
    chunks: List[str] = []
    short_chunks: List[str] = []
    total = 0
    for article in articles:
        chunk = article_body_for_summary(article)
        if word_count(chunk) < 8:
            # save tiny snippets only as a last-resort fallback
            if chunk:
                short_chunks.append(chunk)
            continue
        # keep the prompt small enough for local transformer models
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks or short_chunks[:5])


# load the HuggingFace summarizer once so it is not reloaded on every click
@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load a small transformer summarizer if HuggingFace is available."""
    try:
        from transformers import pipeline  # type: ignore

        # distilBART is smaller than full BART, so it is more realistic for local demos
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        # if transformers/model download fails, the app falls back to extractive summaries
        return None


# backup summarizer when transformers or the model are not available
def extractive_summary(text: str, keywords: Sequence[str], max_sentences: int = 8) -> str:
    """Deterministic fallback summary when transformers/model files are unavailable."""
    sentences = split_sentences(text)
    scored: List[Tuple[float, int, str]] = []
    keyword_words = normalize_for_similarity(" ".join(keywords))

    for index, sentence in enumerate(sentences):
        sentence_words = normalize_for_similarity(sentence)
        if len(sentence_words) < 5:
            # very short sentences are usually headlines or fragments
            continue
        score = keyword_score(sentence, keywords)
        # reward sentences that overlap with the user's topics
        score += len(sentence_words & keyword_words) * 1.5
        # slightly prefer sentences with enough substance
        score += min(len(sentence_words), 35) / 35
        if word_count(sentence) < 14:
            score -= 1.5
        # Add a small position bias because RSS summaries usually put key facts early.
        score += max(0.0, 0.6 - (index * 0.02))
        scored.append((score, index, sentence))

    chosen: List[Tuple[float, int, str]] = []
    seen_word_sets: List[set[str]] = []
    for score, index, sentence in sorted(scored, key=lambda item: item[0], reverse=True):
        sentence_words = normalize_for_similarity(sentence)
        if not sentence_words:
            continue
        # avoid repeating the same point in the fallback summary
        is_duplicate = any(
            len(sentence_words & seen_words) / max(len(sentence_words | seen_words), 1) > 0.55
            for seen_words in seen_word_sets
        )
        if is_duplicate:
            continue
        chosen.append((score, index, sentence))
        seen_word_sets.append(sentence_words)
        if len(chosen) >= max_sentences:
            break

    # choose the strongest non-repeating sentences, then put them back in reading order
    ordered = sorted(chosen, key=lambda item: item[1])
    summary = " ".join(sentence for _, _, sentence in ordered)
    return summary or textwrap.shorten(clean_text(text), width=900, placeholder="...")


# keep all summary length settings in one place so the slider has a clear effect
def summary_length_settings(target_minutes: int) -> Tuple[int, int, int, int, int]:
    """Map the 5-10 minute slider to model and fallback length budgets."""
    minute_scale = max(0, min(target_minutes - 5, 5))
    # these values are token-ish for BART, not exact reading-time math
    min_summary_length = 60 + (minute_scale * 32)
    max_summary_length = 140 + (minute_scale * 76)
    # fallback summaries also need to scale, or the slider feels broken
    fallback_sentence_count = 2 + (minute_scale * 2)
    fallback_char_limit = 800 + (minute_scale * 850)
    # longer summaries need more source text, not just bigger model settings
    context_char_limit = min(9000, 1500 + (target_minutes * 800))
    return min_summary_length, max_summary_length, fallback_sentence_count, fallback_char_limit, context_char_limit


# summarize one article for the "Articles used" section
def summarize_article(article: Article, keywords: Sequence[str]) -> str:
    """Create a short, useful article-level summary without repeating the title."""
    body = article_body_for_summary(article)
    if not body:
        return "No article summary available from the RSS snippet."

    if word_count(body) < 25:
        # too short for real summarization, so just show a cleaned version
        return textwrap.shorten(body, width=280, placeholder="...")

    # article cards stay short so the page is easy to scan
    summary = extractive_summary(body, keywords, max_sentences=2)
    return textwrap.shorten(summary, width=420, placeholder="...")


# generate the final briefing, using BART first and fallback logic if needed
def summarize_context(text: str, keywords: Sequence[str], target_minutes: int) -> Tuple[str, str]:
    """Generate a concise summary and report which summarization method was used."""
    if not text.strip():
        return "", "No content"

    # Scale every path, including fallback, so the slider is visible in the UI.
    (
        min_summary_length,
        max_summary_length,
        fallback_sentence_count,
        fallback_char_limit,
        context_char_limit,
    ) = summary_length_settings(target_minutes)

    if word_count(text) < 90 or len(split_sentences(text)) < 3:
        # do not force BART to summarize tiny RSS-like input
        cleaned = extractive_summary(text, keywords, max_sentences=fallback_sentence_count)
        return textwrap.shorten(cleaned, width=fallback_char_limit, placeholder="..."), "Cleaned RSS summary"

    summarizer = load_summarizer()

    if summarizer is None:
        # fallback if model loading fails so the app still returns something useful
        fallback = extractive_summary(text, keywords, max_sentences=fallback_sentence_count)
        return textwrap.shorten(fallback, width=fallback_char_limit, placeholder="..."), "Extractive fallback"

    try:
        # Longer targets get more source context, but we still stay under BART's limits.
        limited_text = text[:context_char_limit]
        result = summarizer(
            limited_text,
            max_length=max_summary_length,
            min_length=min_summary_length,
            do_sample=False,
            # beam search makes the output more stable than sampling for this use case
            num_beams=4,
            # these help reduce title/snippet repetition
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
            truncation=True,
        )
        return result[0]["summary_text"].strip(), "HuggingFace BART"
    except Exception:
        # fallback if the model errors during generation
        fallback = extractive_summary(text, keywords, max_sentences=fallback_sentence_count)
        return textwrap.shorten(fallback, width=fallback_char_limit, placeholder="..."), "Extractive fallback"


# show the source articles so the summary does not feel like a black box
def render_article_list(articles: Iterable[Article], keywords: Sequence[str]) -> None:
    """Display source articles used for transparency and citation."""
    for article in articles:
        # simple list reads better here than a bunch of empty-feeling dropdowns
        st.markdown(f"**{article.title}**")
        st.write(summarize_article(article, keywords))
        st.caption(f"{article.source} | Relevance score: {article.score:.3f}")
        if article.link:
            st.link_button("Open source", article.link)
        st.divider()


# main Streamlit UI and app flow
def main() -> None:
    # Streamlit handles both the UI and backend logic in this MVP
    st.set_page_config(page_title="Newsletter Summarizer MVP", page_icon="NEWS", layout="wide")

    st.title("Newsletter Summarizer")
    st.caption("MVP prototype: topic filtering plus AI-generated briefing from RSS/newsletter sources.")

    with st.sidebar:
        st.header("Sources")
        st.write("Sources are generated from your topics using news RSS search.")
        # custom feeds are useful for demos if a news source blocks scraping
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

        # create topic-based feeds first, then let users add their own RSS feeds too
        feed_urls = list(build_topic_feed_urls(keywords))
        feed_urls.extend(url.strip() for url in custom_feeds.splitlines() if url.strip())

        if not feed_urls:
            st.error("Enter a searchable topic or add a custom RSS URL.")
            return

        with st.spinner("Fetching topic-specific newsletter/news items..."):
            # cached for a bit so repeated clicks do not refetch every article
            articles, errors = fetch_articles(tuple(feed_urls))

        if errors:
            with st.expander("Some feeds could not be fetched"):
                for error in errors:
                    st.warning(error)

        if not articles:
            st.error("No articles were fetched. Try different sources or a custom RSS feed.")
            return

        # filtering still runs after dynamic fetching to remove weak matches
        relevant_articles = select_relevant_articles(articles, keywords, max_articles=max_articles)
        if not relevant_articles:
            st.error("No relevant articles were found for those topics.")
            return

        context = build_context(relevant_articles)

        with st.spinner("Generating summary..."):
            # one combined briefing across the selected relevant articles
            summary, method = summarize_context(context, keywords, target_minutes)

        if not summary:
            st.error("Relevant content was found, but the app could not generate a summary.")
            return

        st.subheader("Your briefing")
        st.write(summary)
        st.caption(f"Summary method: {method}")

        st.subheader("Articles used")
        render_article_list(relevant_articles, keywords)
    else:
        st.info("Enter topics, choose sources, and generate a concise newsletter briefing.")


if __name__ == "__main__":
    main()
