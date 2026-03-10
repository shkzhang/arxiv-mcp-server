"""Tool definitions for the arXiv MCP server."""

from .search import search_tool, handle_search
from .download import download_tool, handle_download
from .list_papers import list_tool, handle_list_papers
from .read_paper import read_tool, handle_read_paper
from .download_source import (
    download_source_tool,
    handle_download_source,
    get_html_link_tool,
    handle_get_html_link,
)


__all__ = [
    "search_tool",
    "download_tool",
    "read_tool",
    "handle_search",
    "handle_download",
    "handle_read_paper",
    "list_tool",
    "handle_list_papers",
    "download_source_tool",
    "handle_download_source",
    "get_html_link_tool",
    "handle_get_html_link",
]
