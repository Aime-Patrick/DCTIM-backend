"""Web scraper tests: HTML→text extraction and URL fetching safeguards."""
from __future__ import annotations

import pytest

from app.modules.rag.scraper import (
    ScrapeError,
    extract_html_text,
    scrape_url,
)


def test_extract_html_returns_title_and_readable_text() -> None:
    html = """
    <html><head><title>Education Policy Brief</title></head>
    <body>
      <h1>Teacher training</h1>
      <p>Improves learning outcomes across districts.</p>
      <ul><li>Boost enrolment</li><li>Fund schools</li></ul>
    </body></html>
    """
    text, title = extract_html_text(html)

    assert title == "Education Policy Brief"
    assert "Teacher training" in text
    assert "Improves learning outcomes" in text
    assert "Boost enrolment" in text


def test_extract_html_ignores_scripts_and_styles() -> None:
    html = """
    <html><body>
      <style>.x { color: red }</style>
      <script>const secret = "do-not-show";</script>
      <p>Visible only.</p>
    </body></html>
    """
    text, _ = extract_html_text(html)

    assert "Visible only" in text
    assert "do-not-show" not in text
    assert "color: red" not in text


def test_extract_html_decodes_entities_and_collapses_whitespace() -> None:
    text, title = extract_html_text(
        "<html><head><title>Q&amp;A</title></head><body>  Grand&nbsp;plan   <p>Next&nbsp;step</p>  </body></html>"
    )
    assert title == "Q&A"
    assert "Grand plan" in text
    assert "Next step" in text
    assert "\n" not in text


def test_scrape_url_rejects_non_http_schemes() -> None:
    with pytest.raises(ScrapeError):
        scrape_url("ftp://example.com/file")


def test_scrape_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ScrapeError):
        scrape_url("https://user:pass@example.com/page")


def test_scrape_url_fetches_and_extracts(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        content = b"<html><head><title>Demo</title></head><body><p>Hello world</p></body></html>"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.modules.rag.scraper.httpx.Client",
        FakeClient,
    )

    text, title = scrape_url("https://example.com/demo")

    assert title == "Demo"
    assert text == "Hello world"


def test_scrape_url_rejects_non_html_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"a": 1}'

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.modules.rag.scraper.httpx.Client",
        FakeClient,
    )

    with pytest.raises(ScrapeError):
        scrape_url("https://example.com/data")


def test_scrape_url_rejects_http_error_response(monkeypatch) -> None:
    class FakeResponse:
        status_code = 404
        headers = {"content-type": "text/html"}
        content = b"<html><body>gone</body></html>"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.modules.rag.scraper.httpx.Client",
        FakeClient,
    )

    with pytest.raises(ScrapeError, match="HTTP 404"):
        scrape_url("https://example.com/missing")


def test_scrape_url_rejects_oversized_page(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        content = b"<html><body><p>big</p></body></html>" * 1000

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "app.modules.rag.scraper.httpx.Client",
        FakeClient,
    )

    with pytest.raises(ScrapeError, match="larger"):
        scrape_url("https://example.com/huge", max_bytes=50)