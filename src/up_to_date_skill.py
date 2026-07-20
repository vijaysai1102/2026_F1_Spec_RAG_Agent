from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import quote_plus

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
    """Adds lightweight web context for questions asking for current or latest information."""

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

    DUCKDUCKGO_ENDPOINT = "https://api.duckduckgo.com/?q={query}&format=json&no_html=1&no_redirect=1"
    WIKIPEDIA_OPENSEARCH_ENDPOINT = (
        "https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit={limit}&namespace=0&format=json"
    )
    WIKIPEDIA_SEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"
    REQUEST_TIMEOUT_SECONDS = 8
    REQUEST_HEADERS = {"User-Agent": "F1RAGAgent/1.0"}

    def should_activate(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in self.LIVE_SIGNAL_TERMS)

    def run(self, question: str, max_results: int = 5) -> SkillResult:
        generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if not self.should_activate(question):
            return SkillResult(active=False, generated_at_utc=generated_at_utc, sources=[])

        query = quote_plus(f"Formula 1 {question}")
        endpoint = self.DUCKDUCKGO_ENDPOINT.format(query=query)
        try:
            response = requests.get(
                endpoint,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            return SkillResult(
                active=True,
                generated_at_utc=generated_at_utc,
                sources=[],
                error=f"Live web lookup failed: {exc}",
            )

        payload = response.json()
        sources = self._extract_sources(payload, max_results=max_results)
        if not sources:
            sources = self._search_wikipedia(question, max_results=max_results)
        if not sources:
            sources = self._search_wikipedia_query(question, max_results=max_results)
        return SkillResult(active=True, generated_at_utc=generated_at_utc, sources=sources)

    def _extract_sources(self, payload: dict, max_results: int) -> List[WebSource]:
        sources: List[WebSource] = []

        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        abstract_source = payload.get("AbstractSource") or "DuckDuckGo Abstract"
        if abstract_text and abstract_url:
            sources.append(
                WebSource(
                    title=str(abstract_source),
                    url=str(abstract_url),
                    snippet=str(abstract_text),
                )
            )

        related_topics = payload.get("RelatedTopics") or []
        for topic in related_topics:
            if len(sources) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                sources.append(
                    WebSource(
                        title=self._derive_title(topic["FirstURL"]),
                        url=str(topic["FirstURL"]),
                        snippet=str(topic["Text"]),
                    )
                )
                continue
            nested_topics = topic.get("Topics") if isinstance(topic, dict) else None
            if not isinstance(nested_topics, list):
                continue
            for nested in nested_topics:
                if len(sources) >= max_results:
                    break
                if isinstance(nested, dict) and nested.get("Text") and nested.get("FirstURL"):
                    sources.append(
                        WebSource(
                            title=self._derive_title(nested["FirstURL"]),
                            url=str(nested["FirstURL"]),
                            snippet=str(nested["Text"]),
                        )
                    )

        return sources[:max_results]

    def _search_wikipedia(self, question: str, max_results: int) -> List[WebSource]:
        query = quote_plus(f"Formula 1 {question}")
        endpoint = self.WIKIPEDIA_OPENSEARCH_ENDPOINT.format(query=query, limit=max_results)
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
        if not isinstance(payload, list) or len(payload) < 4:
            return []

        titles = payload[1] if isinstance(payload[1], list) else []
        descriptions = payload[2] if isinstance(payload[2], list) else []
        urls = payload[3] if isinstance(payload[3], list) else []

        sources: List[WebSource] = []
        for title, description, url in zip(titles, descriptions, urls):
            if not title or not url:
                continue
            snippet = description or "Wikipedia summary result."
            sources.append(WebSource(title=str(title), url=str(url), snippet=str(snippet)))
            if len(sources) >= max_results:
                break
        return sources

    def _search_wikipedia_query(self, question: str, max_results: int) -> List[WebSource]:
        query = f"Formula 1 {question}"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        }
        try:
            response = requests.get(
                self.WIKIPEDIA_SEARCH_ENDPOINT,
                params=params,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            response.raise_for_status()
        except requests.RequestException:
            return []

        payload = response.json()
        query_payload = payload.get("query", {}) if isinstance(payload, dict) else {}
        search_results = query_payload.get("search", []) if isinstance(query_payload, dict) else []

        sources: List[WebSource] = []
        for result in search_results:
            if not isinstance(result, dict):
                continue
            title = result.get("title")
            page_id = result.get("pageid")
            if not title or not page_id:
                continue
            snippet_raw = str(result.get("snippet") or "Wikipedia search result.")
            snippet = snippet_raw.replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            url = f"https://en.wikipedia.org/?curid={page_id}"
            sources.append(WebSource(title=str(title), url=url, snippet=snippet))
            if len(sources) >= max_results:
                break
        return sources

    @staticmethod
    def _derive_title(url: str) -> str:
        leaf = url.rstrip("/").split("/")[-1]
        if not leaf:
            return "Web Source"
        return leaf.replace("_", " ")
