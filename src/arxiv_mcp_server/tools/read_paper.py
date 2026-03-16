"""Read functionality for the arXiv MCP server."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import arxiv
import mcp.types as types
import pymupdf4llm
from ..config import Settings
from .download import get_paper_path
from .download_source import (
    get_paper_info,
    handle_download_source,
    sanitize_filename,
)

settings = Settings()

LATEX_TEXT_EXTENSIONS = {
    ".tex",
    ".bib",
    ".sty",
    ".cls",
    ".txt",
    ".md",
}

read_tool = types.Tool(
    name="read_paper",
    description=(
        "Read a paper with automatic retrieval. "
        "Prefers LaTeX source (downloaded to ~/.arxiv-mcp-server) and falls "
        "back to PDF->Markdown. Supports keyword-based snippet search."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "The arXiv ID of the paper to read",
            },
            "keyword": {
                "type": "string",
                "description": "Optional keyword (or comma-separated keywords) to search within paper content",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional keyword list for snippet search",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of keyword matches to return (default: 20)",
                "default": 20,
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of surrounding lines per snippet (default: 2)",
                "default": 2,
            },
        },
        "required": ["paper_id"],
    },
)


def _json_response(payload: Dict[str, Any]) -> List[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(payload))]


def _source_base_dir() -> Path:
    base_dir = (Path.home() / ".arxiv-mcp-server").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _extract_keywords(arguments: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []
    single_keyword = arguments.get("keyword")
    if isinstance(single_keyword, str):
        keywords.extend([kw.strip() for kw in single_keyword.split(",") if kw.strip()])

    multi_keywords = arguments.get("keywords", [])
    if isinstance(multi_keywords, list):
        for keyword in multi_keywords:
            if isinstance(keyword, str):
                normalized = keyword.strip()
                if normalized:
                    keywords.append(normalized)

    # Preserve order while deduplicating.
    return list(dict.fromkeys(keywords))


def _find_existing_markdown_path(paper_id: str, paper_title: str | None = None) -> Path | None:
    if paper_title:
        title_path = get_paper_path(paper_id, ".md", paper_title=paper_title)
        if title_path.exists():
            return title_path

    id_path = get_paper_path(paper_id, ".md")
    if id_path.exists():
        return id_path
    return None


def _sync_download_pdf_to_markdown(
    paper_id: str, paper: arxiv.Result
) -> tuple[Path, str, str]:
    paper_title = paper.title
    pdf_path = get_paper_path(paper_id, ".pdf", paper_title=paper_title)
    md_path = get_paper_path(paper_id, ".md", paper_title=paper_title)

    paper.download_pdf(dirpath=pdf_path.parent, filename=pdf_path.name)
    markdown = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
    md_path.write_text(markdown, encoding="utf-8")

    if pdf_path.exists():
        pdf_path.unlink()

    return md_path, markdown, paper_title


def _parse_tool_response(response: List[types.TextContent]) -> Dict[str, Any]:
    if not response:
        return {}
    return json.loads(response[0].text)


def _has_latex_files(source_dir: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() == ".tex"
        for path in source_dir.rglob("*")
    )


async def _ensure_latex_source(
    paper_id: str,
) -> tuple[Path | None, str | None, str | None]:
    source_base = _source_base_dir()
    paper = get_paper_info(paper_id)
    paper_title = paper.title if paper else None

    if paper_title:
        expected_dir = source_base / sanitize_filename(paper_title)
        if expected_dir.exists() and _has_latex_files(expected_dir):
            return expected_dir, None, paper_title

    source_result = await handle_download_source(
        {"paper_id": paper_id, "download_dir": str(source_base)}
    )
    parsed = _parse_tool_response(source_result)

    if parsed.get("status") != "success":
        return None, parsed.get("message"), paper_title

    source_path_str = parsed.get("extract_path")
    if not source_path_str and parsed.get("file_path"):
        source_path_str = str(Path(parsed["file_path"]).parent)

    if not source_path_str:
        return None, "Source downloaded but no source path returned", paper_title

    source_dir = Path(source_path_str).resolve()
    if not source_dir.exists():
        return None, f"Source directory does not exist: {source_dir}", paper_title

    if not _has_latex_files(source_dir):
        return None, "Source directory does not contain .tex files", paper_title

    return source_dir, None, parsed.get("paper_title") or paper_title


async def _ensure_markdown_content(
    paper_id: str, paper_title_hint: str | None = None
) -> tuple[Path, str, str | None]:
    paper = get_paper_info(paper_id)
    paper_title = paper.title if paper else paper_title_hint

    existing_md_path = _find_existing_markdown_path(paper_id, paper_title)
    if existing_md_path:
        return (
            existing_md_path,
            existing_md_path.read_text(encoding="utf-8"),
            paper_title,
        )

    if not paper:
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        paper_title = paper.title

    md_path, markdown, paper_title = await asyncio.to_thread(
        _sync_download_pdf_to_markdown, paper_id, paper
    )
    return md_path, markdown, paper_title


def _collect_latex_contents(source_dir: Path) -> List[Dict[str, str]]:
    file_entries: List[Dict[str, str]] = []

    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in LATEX_TEXT_EXTENSIONS:
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            continue

        relative_path = path.relative_to(source_dir).as_posix()
        file_entries.append({"file": relative_path, "content": content})

    def _sort_key(entry: Dict[str, str]) -> tuple[int, int, str]:
        file_name = Path(entry["file"]).name.lower()
        root_priority = 0 if file_name in {"main.tex", "root.tex", "arxiv.tex"} else 1
        depth = entry["file"].count("/")
        return (root_priority, depth, entry["file"])

    file_entries.sort(key=_sort_key)
    return file_entries


def _build_latex_combined_content(file_entries: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for entry in file_entries:
        blocks.append(f"%% FILE: {entry['file']}\n{entry['content']}")
    return "\n\n".join(blocks)


def _search_keyword_snippets(
    file_entries: List[Dict[str, str]],
    keywords: List[str],
    max_matches: int,
    context_lines: int,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    keyword_lowers = [(keyword, keyword.lower()) for keyword in keywords]

    for entry in file_entries:
        lines = entry["content"].splitlines()
        for line_idx, line in enumerate(lines):
            lower_line = line.lower()
            for keyword, keyword_lower in keyword_lowers:
                if keyword_lower not in lower_line:
                    continue

                start = max(0, line_idx - context_lines)
                end = min(len(lines), line_idx + context_lines + 1)
                snippet = "\n".join(lines[start:end])
                matches.append(
                    {
                        "keyword": keyword,
                        "file": entry["file"],
                        "line": line_idx + 1,
                        "snippet": snippet,
                    }
                )
                if len(matches) >= max_matches:
                    return matches

    return matches


async def handle_read_paper(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Read paper content with source-first retrieval and keyword search."""
    try:
        paper_id = arguments["paper_id"]
        keywords = _extract_keywords(arguments)
        max_matches = max(1, int(arguments.get("max_matches", 20)))
        context_lines = max(0, int(arguments.get("context_lines", 2)))

        source_dir, source_error, paper_title = await _ensure_latex_source(paper_id)
        if source_dir:
            latex_entries = _collect_latex_contents(source_dir)
            if not latex_entries:
                source_error = "Source directory has no readable LaTeX/text files"
            else:
                if keywords:
                    matches = _search_keyword_snippets(
                        latex_entries, keywords, max_matches, context_lines
                    )
                    return _json_response(
                        {
                            "status": "success",
                            "paper_id": paper_id,
                            "paper_title": paper_title,
                            "content_format": "latex",
                            "source_dir": str(source_dir),
                            "keywords": keywords,
                            "total_matches": len(matches),
                            "matches": matches,
                        }
                    )

                return _json_response(
                    {
                        "status": "success",
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "content_format": "latex",
                        "source_dir": str(source_dir),
                        "source_files": [entry["file"] for entry in latex_entries],
                        "content": _build_latex_combined_content(latex_entries),
                    }
                )

        md_path, markdown_content, md_title = await _ensure_markdown_content(
            paper_id, paper_title_hint=paper_title
        )

        if keywords:
            markdown_entries = [{"file": md_path.name, "content": markdown_content}]
            matches = _search_keyword_snippets(
                markdown_entries, keywords, max_matches, context_lines
            )
            return _json_response(
                {
                    "status": "success",
                    "paper_id": paper_id,
                    "paper_title": md_title or paper_title,
                    "content_format": "markdown",
                    "keywords": keywords,
                    "total_matches": len(matches),
                    "matches": matches,
                    "fallback_reason": source_error,
                }
            )

        return _json_response(
            {
                "status": "success",
                "paper_id": paper_id,
                "paper_title": md_title or paper_title,
                "content_format": "markdown",
                "content": markdown_content,
                "fallback_reason": source_error,
            }
        )
    except Exception as exc:
        return _json_response(
            {
                "status": "error",
                "message": f"Error reading paper: {str(exc)}",
            }
        )
