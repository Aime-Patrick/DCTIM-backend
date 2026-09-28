"""Web scraping helper: fetch a public HTTP(S) page and extract readable text.

The text extraction is implemented with the standard library only
(``html.parser`` + ``re``) so the backend does not pull in a new dependency.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

# Elements whose content is never part of the readable article text.
_SKIP_ELEMENTS = {"script", "style", "noscript", "template", "svg", "iframe"}
# Elements that imply a visual block / paragraph break when encountered.
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "li", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table",
}

_USER_AGENT = "Mozilla/5.0 (compatible; DC-TIM/0.1; +https://dc-tim.local/data-collector)"

_VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                  "link", "meta", "param", "source", "track", "wbr"}


class ScrapeError(ValueError):
    """Raised when a URL cannot be fetched or parsed into usable text."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: N802 (HTMLParser API)
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_ELEMENTS:
            self._skip_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: N802
        if tag in _SKIP_ELEMENTS:
            return
        if tag in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:  # noqa: N802
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP_ELEMENTS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            if (text := data.strip()):
                self._title_parts.append(text)
            return
        if self._skip_depth == 0 and (text := data.strip()):
            self._parts.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()

    def title(self) -> str | None:
        raw = " ".join(self._title_parts).strip()
        return raw or None


def extract_html_text(html: str) -> tuple[str, str | None]:
    """Return ``(visible_text, page_title)`` extracted from an HTML document.

    ``page_title`` is ``None`` when the document has no usable ``<title>``.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - malformed input is fed safely
        raise ScrapeError("The HTML document could not be parsed.") from exc
    return parser.text(), parser.title()


def scrape_url(
    url: str,
    *,
    timeout_seconds: float = 20.0,
    max_bytes: int = 5_000_000,
) -> tuple[str, str | None]:
    """Fetch ``url`` and return ``(visible_text, page_title)``.

    Raises :class:`ScrapeError` for unsupported URLs, network failures,
    non-HTML responses, and oversized pages.
    """
    target = url.strip()
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ScrapeError("Only absolute http(s) URLs are supported.")
    if parsed.username or parsed.password:
        raise ScrapeError("URLs with embedded credentials are not supported.")

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html, application/xhtml+xml",
    }
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
            response = client.get(target)
    except httpx.TimeoutException as exc:
        raise ScrapeError(
            f"The page took too long to respond ({timeout_seconds:g}s)."
        ) from exc
    except Exception as exc:  # pragma: no cover - network layer surface is broad
        raise ScrapeError("Could not reach the page.") from exc

    if response.status_code >= 400:
        raise ScrapeError(f"The page returned HTTP {response.status_code}.")
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower():
        raise ScrapeError("The URL did not return an HTML page.")

    data = response.content
    if not data:
        raise ScrapeError("The page returned no content.")
    if len(data) > max_bytes:
        raise ScrapeError("The page is larger than the size limit.")

    return extract_html_text(data.decode("utf-8", errors="replace"))