"""
OpenAlex API adapter for the AI Research and Learning Copilot.

All HTTP calls and response parsing for OpenAlex (https://openalex.org)
live here, so the MCP tool functions in research_mcp_server.py stay thin.

OpenAlex requires no API key. Adding `mailto` to every request opts into
OpenAlex's "polite pool", which gets faster and more reliable rate limits
than anonymous requests - not a secret, just contact info, configured via
the OPENALEX_MAILTO env var (see app.yaml).

Docs: https://docs.openalex.org/api-entities/works
"""

import os

import requests

OPENALEX_BASE_URL = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "surajgoel1997@gmail.com")
REQUEST_TIMEOUT_SECONDS = 15


class OpenAlexError(Exception):
    """Raised for any user-facing OpenAlex lookup failure (bad ID, API
    outage, etc). Callers should show str(exc) to the user, not a stack
    trace."""


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex stores abstracts as an inverted index (word -> list of word
    positions) instead of plain text, to work around publisher copyright
    restrictions on redistributing abstract text. Rebuild the plain-text
    abstract from it.
    """
    if not inverted_index:
        return None
    max_position = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_position + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words).strip() or None


def _parse_work(work: dict) -> dict:
    """Parse a single OpenAlex `work` object into this app's paper + authors shape."""
    authorships = work.get("authorships") or []
    authors = [
        {
            "id": (a.get("author") or {}).get("id", "").rsplit("/", 1)[-1],
            "display_name": (a.get("author") or {}).get("display_name"),
            "orcid": (a.get("author") or {}).get("orcid"),
            "author_position": i,
        }
        for i, a in enumerate(authorships)
        if (a.get("author") or {}).get("id")
    ]

    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = work.get("open_access") or {}
    concepts = sorted(work.get("concepts") or [], key=lambda c: c.get("score", 0), reverse=True)

    return {
        "id": work.get("id", "").rsplit("/", 1)[-1],
        "title": work.get("title") or work.get("display_name") or "Untitled",
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "doi": work.get("doi"),
        "cited_by_count": work.get("cited_by_count", 0),
        "open_access_url": open_access.get("oa_url"),
        "concepts": [c["display_name"] for c in concepts[:5] if c.get("display_name")],
        "authors": authors,
    }


def search_works(query: str, limit: int = 10) -> list[dict]:
    """
    Search OpenAlex for works (papers) matching a free-text query.

    Args:
        query: Free-text search, e.g. "retrieval augmented generation".
        limit: Max number of results (1-50).

    Returns:
        A list of parsed paper dicts (see _parse_work), each with an
        "authors" list embedded.

    Raises:
        OpenAlexError if the query is empty or the API call fails.
    """
    if not query or not query.strip():
        raise OpenAlexError("A search query is required.")

    limit = max(1, min(int(limit), 50))

    try:
        resp = requests.get(
            f"{OPENALEX_BASE_URL}/works",
            params={"search": query.strip(), "per_page": limit, "mailto": MAILTO},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise OpenAlexError(f"Could not reach OpenAlex: {e}") from e

    return [_parse_work(w) for w in data.get("results", [])]


def get_work(work_id: str) -> dict:
    """
    Fetch a single OpenAlex work by ID (e.g. "W2741809807" or the full
    OpenAlex URL).

    Raises:
        OpenAlexError if the ID can't be found or the API call fails.
    """
    if not work_id or not work_id.strip():
        raise OpenAlexError("A paper ID is required.")

    work_id = work_id.strip().rsplit("/", 1)[-1]

    try:
        resp = requests.get(
            f"{OPENALEX_BASE_URL}/works/{work_id}",
            params={"mailto": MAILTO},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise OpenAlexError(f"Could not reach OpenAlex: {e}") from e

    if resp.status_code == 404:
        raise OpenAlexError(f"No OpenAlex work found with ID {work_id!r}.")
    resp.raise_for_status()

    return _parse_work(resp.json())
