"""Download functionality for the arXiv MCP server."""

import arxiv
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import mcp.types as types
from ..config import Settings
import pymupdf4llm
import logging

logger = logging.getLogger("arxiv-mcp-server")
settings = Settings()


def sanitize_filename(title: str) -> str:
    """Sanitize paper title to be used as a valid filename.
    
    Removes or replaces special characters that are not allowed in filenames.
    """
    # Replace common problematic characters
    sanitized = title.replace("/", "-").replace("\\", "-")
    sanitized = sanitized.replace(":", " -").replace("?", "")
    sanitized = sanitized.replace("*", "").replace('"', "'")
    sanitized = sanitized.replace("<", "").replace(">", "")
    sanitized = sanitized.replace("|", "-")
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip('.')
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

# Global dictionary to track conversion status
conversion_statuses: Dict[str, Any] = {}


@dataclass
class ConversionStatus:
    """Track the status of a PDF to Markdown conversion."""

    paper_id: str
    status: str  # 'downloading', 'converting', 'success', 'error'
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


download_tool = types.Tool(
    name="download_paper",
    description="Download a paper and create a resource for it",
    inputSchema={
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "The arXiv ID of the paper to download",
            },
            "check_status": {
                "type": "boolean",
                "description": "If true, only check conversion status without downloading",
                "default": False,
            },
            "download_dir": {
                "type": "string",
                "description": "Optional custom directory to download the paper to. If not specified, uses the default storage path.",
            },
        },
        "required": ["paper_id"],
    },
)


def get_paper_path(paper_id: str, suffix: str = ".md", download_dir: Optional[str] = None, paper_title: Optional[str] = None) -> Path:
    """Get the absolute file path for a paper with given suffix.
    
    Args:
        paper_id: The arXiv paper ID
        suffix: File extension (e.g., '.md', '.pdf')
        download_dir: Optional custom download directory
        paper_title: Optional paper title to use as filename (will be sanitized)
    """
    if download_dir:
        storage_path = Path(download_dir).resolve()
    else:
        storage_path = Path(settings.STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Use paper title as filename if provided, otherwise use paper_id
    if paper_title:
        filename = sanitize_filename(paper_title)
    else:
        filename = paper_id
    
    return storage_path / f"{filename}{suffix}"


def convert_pdf_to_markdown(paper_id: str, pdf_path: Path, download_dir: Optional[str] = None, paper_title: Optional[str] = None) -> None:
    """Convert PDF to Markdown in a separate thread."""
    try:
        logger.info(f"Starting conversion for {paper_id}")
        markdown = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        md_path = get_paper_path(paper_id, ".md", download_dir, paper_title)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        status = conversion_statuses.get(paper_id)
        if status:
            status.status = "success"
            status.completed_at = datetime.now()

        # Clean up PDF after successful conversion
        logger.info(f"Conversion completed for {paper_id}")

    except Exception as e:
        logger.error(f"Conversion failed for {paper_id}: {str(e)}")
        status = conversion_statuses.get(paper_id)
        if status:
            status.status = "error"
            status.completed_at = datetime.now()
            status.error = str(e)


async def handle_download(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle paper download and conversion requests."""
    try:
        paper_id = arguments["paper_id"]
        check_status = arguments.get("check_status", False)
        download_dir = arguments.get("download_dir")

        # If only checking status
        if check_status:
            status = conversion_statuses.get(paper_id)
            if not status:
                if get_paper_path(paper_id, ".md").exists():
                    return [
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "success",
                                    "message": "Paper is ready",
                                    "resource_uri": f"file://{get_paper_path(paper_id, '.md')}",
                                }
                            ),
                        )
                    ]
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "status": "unknown",
                                "message": "No download or conversion in progress",
                            }
                        ),
                    )
                ]

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": status.status,
                            "started_at": status.started_at.isoformat(),
                            "completed_at": (
                                status.completed_at.isoformat()
                                if status.completed_at
                                else None
                            ),
                            "error": status.error,
                            "message": f"Paper conversion {status.status}",
                        }
                    ),
                )
            ]

        # Get paper info first to get the title
        client = arxiv.Client()
        try:
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
            paper_title = paper.title
        except StopIteration:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "message": f"Paper {paper_id} not found on arXiv",
                        }
                    ),
                )
            ]

        # Check if paper is already converted (check both by ID and by title)
        md_path_by_id = get_paper_path(paper_id, ".md", download_dir)
        md_path_by_title = get_paper_path(paper_id, ".md", download_dir, paper_title)
        
        if md_path_by_title.exists():
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "success",
                            "message": "Paper already available",
                            "resource_uri": f"file://{md_path_by_title}",
                            "paper_title": paper_title,
                        }
                    ),
                )
            ]
        
        if md_path_by_id.exists():
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "success",
                            "message": "Paper already available",
                            "resource_uri": f"file://{md_path_by_id}",
                            "paper_title": paper_title,
                        }
                    ),
                )
            ]

        # Check if already in progress
        if paper_id in conversion_statuses:
            status = conversion_statuses[paper_id]
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": status.status,
                            "message": f"Paper conversion {status.status}",
                            "started_at": status.started_at.isoformat(),
                        }
                    ),
                )
            ]

        # Start new download and conversion
        pdf_path = get_paper_path(paper_id, ".pdf", download_dir, paper_title)

        # Initialize status
        conversion_statuses[paper_id] = ConversionStatus(
            paper_id=paper_id, status="downloading", started_at=datetime.now()
        )

        # Download PDF
        paper.download_pdf(dirpath=pdf_path.parent, filename=pdf_path.name)

        # Update status and start conversion
        status = conversion_statuses[paper_id]
        status.status = "converting"

        # Start conversion in thread
        asyncio.create_task(
            asyncio.to_thread(convert_pdf_to_markdown, paper_id, pdf_path, download_dir, paper_title)
        )

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "converting",
                        "message": "Paper downloaded, conversion started",
                        "started_at": status.started_at.isoformat(),
                        "paper_title": paper_title,
                        "download_path": str(pdf_path.parent),
                    }
                ),
            )
        ]

    except StopIteration:
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "error",
                        "message": f"Paper {paper_id} not found on arXiv",
                    }
                ),
            )
        ]
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"status": "error", "message": f"Error: {str(e)}"}),
            )
        ]
