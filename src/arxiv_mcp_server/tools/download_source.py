"""Download LaTeX source files and get HTML links for arXiv papers."""

import arxiv
import json
import asyncio
import tarfile
import re
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import mcp.types as types
from ..config import Settings
import logging

logger = logging.getLogger("arxiv-mcp-server")
settings = Settings()


def sanitize_filename(title: str) -> str:
    """Sanitize paper title to be used as a valid filename/folder name.
    
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


def get_paper_info(paper_id: str) -> Optional[arxiv.Result]:
    """Get paper information from arXiv API."""
    try:
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[paper_id])))
        return paper
    except StopIteration:
        return None
    except Exception as e:
        logger.error(f"Error fetching paper info: {e}")
        return None


# Tool definition for downloading LaTeX source
download_source_tool = types.Tool(
    name="download_source",
    description="Download the LaTeX source files of an arXiv paper. Downloads the tar.gz archive and extracts it to a folder named after the paper title.",
    inputSchema={
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "The arXiv ID of the paper (e.g., '2301.00001' or '2301.00001v1')",
            },
            "download_dir": {
                "type": "string",
                "description": "Optional custom directory to download the source files to. If not specified, uses the default storage path.",
            },
        },
        "required": ["paper_id"],
    },
)


# Tool definition for getting HTML link
get_html_link_tool = types.Tool(
    name="get_html_link",
    description="Get the HTML version link for an arXiv paper. Returns the URL to view the paper in HTML format.",
    inputSchema={
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": "The arXiv ID of the paper (e.g., '2301.00001' or '2301.00001v1')",
            },
        },
        "required": ["paper_id"],
    },
)


async def handle_download_source(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle LaTeX source download requests."""
    try:
        paper_id = arguments["paper_id"]
        custom_dir = arguments.get("download_dir")
        
        # Get paper info for title
        paper = get_paper_info(paper_id)
        if not paper:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": f"Paper {paper_id} not found on arXiv",
                    }),
                )
            ]
        
        # Sanitize title for folder name
        folder_name = sanitize_filename(paper.title)
        
        # Determine download directory
        if custom_dir:
            base_path = Path(custom_dir).resolve()
        else:
            base_path = Path(settings.STORAGE_PATH)
        
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create folder for extracted files
        extract_path = base_path / folder_name
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Download source tar.gz
        # Clean paper_id (remove version if present for source URL)
        clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
        source_url = f"https://arxiv.org/src/{paper_id}"
        
        tar_path = base_path / f"{folder_name}.tar.gz"
        
        logger.info(f"Downloading source from {source_url}")
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(source_url)
            
            if response.status_code != 200:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "error",
                            "message": f"Failed to download source: HTTP {response.status_code}. Note: Not all papers have source files available.",
                        }),
                    )
                ]
            
            # Save tar.gz file
            with open(tar_path, "wb") as f:
                f.write(response.content)
        
        # Extract tar.gz
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            
            # Remove tar.gz after successful extraction
            tar_path.unlink()
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": "LaTeX source files downloaded and extracted successfully",
                        "paper_id": paper_id,
                        "paper_title": paper.title,
                        "extract_path": str(extract_path),
                    }),
                )
            ]
        except tarfile.TarError as e:
            # Some papers might be single files, not tar archives
            # Try to handle as a single file
            logger.warning(f"Not a valid tar.gz, might be a single file: {e}")
            
            # Rename the file to .tex if it's likely a TeX file
            single_file_path = extract_path / f"{folder_name}.tex"
            tar_path.rename(single_file_path)
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": "Source file downloaded (single file, not archive)",
                        "paper_id": paper_id,
                        "paper_title": paper.title,
                        "file_path": str(single_file_path),
                    }),
                )
            ]
            
    except Exception as e:
        logger.error(f"Error downloading source: {e}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Error downloading source: {str(e)}",
                }),
            )
        ]


async def handle_get_html_link(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle HTML link requests."""
    try:
        paper_id = arguments["paper_id"]
        
        # Get paper info to verify it exists and get title
        paper = get_paper_info(paper_id)
        if not paper:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": f"Paper {paper_id} not found on arXiv",
                    }),
                )
            ]
        
        # Generate HTML link
        html_url = f"https://arxiv.org/html/{paper_id}"
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "paper_id": paper_id,
                    "paper_title": paper.title,
                    "html_url": html_url,
                    "note": "Note: HTML version may not be available for all papers. Older papers or those with complex formatting might not have HTML versions.",
                }),
            )
        ]
        
    except Exception as e:
        logger.error(f"Error getting HTML link: {e}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Error: {str(e)}",
                }),
            )
        ]
