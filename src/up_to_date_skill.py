from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
from typing import List, Optional
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET

import requests


@dataclass
class WebSource:
    title: str
    url: str
    snippet: str


@dataclass
class SkillResult:
    active: bool
    generated_at_utc: str
    sources: List[WebSource]
    error: Optional[str] = None

    @property
    def web_context(self) -> str:
        if not self.sources:
            return ""
        lines = []
        for index, source in enumerate(self.sources, start=1):
            lines.append(
                f"{index}. {source.title}\nURL: {source.url}\nSnippet: {source.snippet}"
            )
        return "\n\n".join(lines)


class UpToDateAnswerSkill:
    """Fetches relevant and recent web sources for time-sensitive F1 questions."""

    LIVE_SIGNAL_TERMS = (
        "latest",
        "current",
        "recent",
        "today",
        "this week",
        "this month",
        "this year",
        "update",
        "updated",
        "new",
        "breaking",
        "news",
        "now",
        "as of",
    )

    DUCKDUCKGO_ENDPOINT = (
        "https://api.duckduckgo.com/?q={query}&format=json&no_html=1&no_redirect=1"
    )
    REQUEST_TIMEOUT_SECONDS = 8
    REQUEST_HEADERS = {"User-Agent": "F1RAGAgent/1.0"}
    MAX_SOURCE_AGE_DAYS = 180
    AUTHORITATIVE_FEEDS = (
        ("https://www.formula1.com/en/latest/all.xml", 8.0),
        ("https://www.fia.com/rss/news", 7.0),
        ("https://www.motorsport.com/rss/f1/news/", 5.0),
        ("https://www.autosport.com/rss/feed/f1", 4.0),
    )
    BLOCKED_DOMAINS = ("wikipedia.org",)
    OTHER_SERIES_TERMS = (
        "formula e",
        "wrc",
        "frec",
        "formula 2",
        "formula 3",
        "f2",
        "f3",
        "wec",
        "rally",
    )
    STOPWORDS = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "about",
        "latest",
        "current",
        "recent",
        "update",
        "updates",
        "news",
        "f1",
        "formula",
    }

    def should_activate(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in self.LIVE_SIGNAL_TERMS)

    def run(self, question: str, max_results: int = 5) -> SkillResult:
        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if not self.should_activate(question):
            return SkillResult(active=False, generated_at_utc=generated_at_utc, sources=[])

        sources = self._search_authoritative_feeds(question, max_results=max_results)
        if not sources:
            sources = self._search_duckduckgo_related(question, max_results=max_results)

        error = None
        if not sources:
            error = "No recent live sources were found from configured providers."
        return SkillResult(
            active=True,
            generated_at_utc=generated_at_utc,
            sources=sources,
            error=error,
        )

    def _search_authoritative_feeds(self, question: str, max_results: int) -> List[WebSource]:
        question_tokens = self._tokenize(question)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.MAX_SOURCE_AGE_DAYS)
        relevant_candidates: List[tuple[float, WebSource]] = []
        fallback_candidates: List[tuple[float, WebSource]] = []

        for feed_url, domain_weight in self.AUTHORITATIVE_FEEDS:
            try:
                response = requests.get(
                    feed_url,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                    headers=self.REQUEST_HEADERS,
                )
                response.raise_for_status()
            except requests.RequestException:
                continue

            for source in self._parse_rss_sources(response.text):
                if self._is_blocked_domain(source.url):
                    continue
                if not self._is_f1_article(source):
                    continue
                published_at = self._extract_published_at(source.snippet)
                if published_at and published_at < cutoff:
                    continue
                recency = self._recency_score(published_at, now)
                overlap = self._overlap_score(source, question_tokens)
                base_score = domain_weight + recency
                fallback_candidates.append((base_score, source))
                if self._is_relevant(source, question_tokens):
                    relevant_candidates.append((base_score + overlap, source))

        candidates = relevant_candidates if relevant_candidates else fallback_candidates
        candidates.sort(key=lambda item: item[0], reverse=True)
        deduped: List[WebSource] = []
        seen_urls = set()
        for _, source in candidates:
            normalized = source.url.split("?")[0]
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            deduped.append(source)
            if len(deduped) >= max_results:
                break
        return deduped

    def _search_duckduckgo_related(self, question: str, max_results: int) -> List[WebSource]:
        query = quote_plus(f"Formula 1 latest {question}")
        endpoint = self.DUCKDUCKGO_ENDPOINT.format(query=query)
        try:
            response = requests.get(
                endpoint,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        payload = response.json()
        related_topics = payload.get("RelatedTopics") or []
        question_tokens = self._tokenize(question)
        sources: List[WebSource] = []

        for topic in related_topics:
            if len(sources) >= max_results:
                break
            candidates = []
            if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                candidates.append(topic)
            nested = topic.get("Topics") if isinstance(topic, dict) else None
            if isinstance(nested, list):
                candidates.extend(item for item in nested if isinstance(item, dict))

            for item in candidates:
                if len(sources) >= max_results:
                    break
                text = str(item.get("Text") or "")
                url = str(item.get("FirstURL") or "")
                if not text or not url:
                    continue
                if self._is_blocked_domain(url):
                    continue
                source = WebSource(title=self._derive_title(url), url=url, snippet=text)
                if not self._is_f1_article(source):
                    continue
                if not self._is_relevant(source, question_tokens):
                    continue
                sources.append(source)
        return sources

    def _parse_rss_sources(self, xml_text: str) -> List[WebSource]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        items = root.findall(".//item")
        if not items:
            return []

        sources: List[WebSource] = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            description = (item.findtext("description") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            snippet = description
            if pub_date:
                snippet = f"{pub_date} | {description}" if description else pub_date
            sources.append(WebSource(title=title, url=link, snippet=snippet))
        return sources

    def _is_relevant(self, source: WebSource, question_tokens: set[str]) -> bool:
        haystack = f"{source.title} {source.snippet}".lower()
        if "formula e" in haystack:
            return False
        if not question_tokens:
            return True
        overlap = sum(1 for token in question_tokens if token in haystack)
        return overlap >= 1

    def _overlap_score(self, source: WebSource, question_tokens: set[str]) -> float:
        if not question_tokens:
            return 0.0
        haystack = f"{source.title} {source.snippet}".lower()
        overlap = sum(1 for token in question_tokens if token in haystack)
        return float(overlap * 2)

    def _recency_score(self, published_at: Optional[datetime], now: datetime) -> float:
        if not published_at:
            return 0.0
        age_days = max((now - published_at).days, 0)
        if age_days <= 7:
            return 4.0
        if age_days <= 30:
            return 2.0
        if age_days <= 90:
            return 1.0
        return 0.0

    def _extract_published_at(self, snippet: str) -> Optional[datetime]:
        prefix = snippet.split("|", 1)[0].strip()
        if not prefix:
            return None
        try:
            parsed = parsedate_to_datetime(prefix)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _tokenize(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
        return {token for token in tokens if token not in self.STOPWORDS}

    def _is_blocked_domain(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        return any(blocked in domain for blocked in self.BLOCKED_DOMAINS)

    def _is_f1_article(self, source: WebSource) -> bool:
        domain = urlparse(source.url).netloc.lower()
        if "formula1.com" in domain or "motorsport.com" in domain and "/f1/" in source.url:
            return True
        if "autosport.com" in domain and "/f1/" in source.url:
            return True
        haystack = f"{source.title} {source.snippet}".lower()
        if any(term in haystack for term in self.OTHER_SERIES_TERMS):
            return False
        return "f1" in haystack or "formula 1" in haystack or "grand prix" in haystack

    @staticmethod
    def _derive_title(url: str) -> str:
        leaf = url.rstrip("/").split("/")[-1]
        if not leaf:
            return "Web Source"
        return leaf.replace("_", " ")
