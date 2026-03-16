"""Tests for read_paper source-first and keyword behavior."""

import json
from pathlib import Path
import pytest
from arxiv_mcp_server.tools.read_paper import handle_read_paper


def _mock_async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


@pytest.mark.asyncio
async def test_read_paper_prefers_latex(mocker):
    """Should return LaTeX content when source is available."""
    source_dir = Path("/tmp/mock-source")
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_latex_source",
        side_effect=_mock_async_return((source_dir, None, "Mock Paper")),
    )
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._collect_latex_contents",
        return_value=[
            {"file": "main.tex", "content": "Intro section"},
            {"file": "sections/method.tex", "content": "Method section"},
        ],
    )
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_markdown_content",
        side_effect=RuntimeError("Markdown fallback should not run"),
    )

    response = await handle_read_paper({"paper_id": "2401.12345"})
    payload = json.loads(response[0].text)

    assert payload["status"] == "success"
    assert payload["content_format"] == "latex"
    assert payload["source_dir"] == str(source_dir)
    assert "%% FILE: main.tex" in payload["content"]


@pytest.mark.asyncio
async def test_read_paper_falls_back_to_markdown(mocker):
    """Should return markdown when LaTeX source is unavailable."""
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_latex_source",
        side_effect=_mock_async_return((None, "No source files available", None)),
    )
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_markdown_content",
        side_effect=_mock_async_return(
            (Path("/tmp/mock.md"), "# Title\nMarkdown body", "Mock Paper")
        ),
    )

    response = await handle_read_paper({"paper_id": "2401.12345"})
    payload = json.loads(response[0].text)

    assert payload["status"] == "success"
    assert payload["content_format"] == "markdown"
    assert payload["content"] == "# Title\nMarkdown body"
    assert payload["fallback_reason"] == "No source files available"


@pytest.mark.asyncio
async def test_read_paper_keyword_search_on_latex(mocker):
    """Keyword mode should return snippet matches instead of full content."""
    source_dir = Path("/tmp/mock-source")
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_latex_source",
        side_effect=_mock_async_return((source_dir, None, "Mock Paper")),
    )
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._collect_latex_contents",
        return_value=[
            {
                "file": "main.tex",
                "content": "line one\nThis introduces transformer blocks\nline three",
            }
        ],
    )

    response = await handle_read_paper(
        {
            "paper_id": "2401.12345",
            "keyword": "transformer",
            "context_lines": 0,
        }
    )
    payload = json.loads(response[0].text)

    assert payload["status"] == "success"
    assert payload["content_format"] == "latex"
    assert payload["total_matches"] == 1
    assert payload["matches"][0]["file"] == "main.tex"
    assert "transformer" in payload["matches"][0]["snippet"].lower()
    assert "content" not in payload


@pytest.mark.asyncio
async def test_read_paper_keyword_search_on_markdown_fallback(mocker):
    """Keyword mode should search markdown when source is unavailable."""
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_latex_source",
        side_effect=_mock_async_return((None, "source unavailable", "Mock Paper")),
    )
    mocker.patch(
        "arxiv_mcp_server.tools.read_paper._ensure_markdown_content",
        side_effect=_mock_async_return(
            (
                Path("/tmp/mock.md"),
                "# Heading\nWe evaluate diffusion policies.\nDone.",
                "Mock Paper",
            )
        ),
    )

    response = await handle_read_paper(
        {
            "paper_id": "2401.12345",
            "keywords": ["diffusion", "policy"],
            "max_matches": 3,
        }
    )
    payload = json.loads(response[0].text)

    assert payload["status"] == "success"
    assert payload["content_format"] == "markdown"
    assert payload["total_matches"] >= 1
    assert payload["matches"][0]["file"] == "mock.md"
    assert payload["fallback_reason"] == "source unavailable"
