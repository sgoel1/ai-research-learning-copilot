"""
AI Research and Learning Copilot MCP server.

Exposes research/learning tools over MCP (Model Context Protocol) so a
Databricks Agent Bricks agent can call them like any other tool:
    - create_learning_goal(user_email, title, description, topic_keywords)
    - find_papers_for_goal(topic_keywords, limit)
    - search_papers_semantic(query, limit, collection_id)
    - get_paper_details(paper_ids)
    - create_collection(user_email, name, learning_goal_id)
    - add_paper_to_collection(collection_id, paper_id)
    - generate_reading_plan(collection_id)
    - get_collection(collection_id)
    - update_reading_progress(user_email, paper_id, status)
    - get_next_paper(user_email, collection_id)
    - add_note(user_email, body, paper_id)

Backed by the OpenAlex API (no key required - see openalex_broker.py) for
paper discovery, and Lakebase (Postgres + pgvector) for all persistence
and semantic search. Every tool here is a thin one-line delegation to
research_logic.py, which holds the actual logic and is shared with the
dashboard app so neither app re-derives the same behavior.

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern as the Day 3 reference app's mcp_server/alpaca_mcp_server.py), so
an Agent Bricks agent can register its URL as an external MCP server.

Run locally:
    python research_mcp_server.py
"""

import logging
import os

from fastmcp import FastMCP

import research_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-mcp-server")

mcp = FastMCP("ai-research-learning-copilot")


@mcp.tool
def create_learning_goal(user_email: str, title: str, description: str, topic_keywords: str) -> dict:
    """
    Create a new learning goal for a user.

    Args:
        user_email: The user's email (their identity in this app).
        title: Short title, e.g. "Understand retrieval-augmented generation".
        description: Longer free-text description of what the user wants to learn.
        topic_keywords: Comma-separated search terms to use when finding
            papers for this goal, e.g. "retrieval augmented generation, vector search".

    Returns:
        On success: {"status": "success", "learning_goal_id": int, ...}.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.create_learning_goal(user_email, title, description, topic_keywords)


@mcp.tool
def find_papers_for_goal(topic_keywords: str, limit: int = 10) -> dict:
    """
    Search OpenAlex live for papers matching a learning goal's topic keywords.
    Does NOT persist results - review candidates, then call
    add_paper_to_collection for the ones worth keeping.

    Args:
        topic_keywords: Free-text search, e.g. "retrieval augmented generation".
        limit: Max number of candidate papers to return (1-50, default 10).

    Returns:
        On success: {"status": "success", "papers": [...]}, each with id,
        title, abstract, publication_year, venue, doi, cited_by_count,
        open_access_url, concepts, and authors.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.find_papers_for_goal(topic_keywords, limit)


@mcp.tool
def search_papers_semantic(query: str, limit: int = 8, collection_id: int | None = None) -> dict:
    """
    Semantic search over already-known paper abstracts (cached from OpenAlex
    into Lakebase, either by the offline ingestion pipeline or by previous
    add_paper_to_collection/get_paper_details calls), using pgvector cosine
    similarity - so an agent can retrieve the most relevant evidence across
    many papers instead of sending an entire collection's abstracts to the model.

    Args:
        query: Free-text semantic query, e.g. "hallucination mitigation techniques".
        limit: Max number of results (default 8).
        collection_id: If given, restrict results to papers already in this collection.

    Returns:
        On success: {"status": "success", "results": [...]}, each with
        source_id (paper ID), text (the abstract), and similarity (0-1).
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.search_papers_semantic(query, limit, collection_id)


@mcp.tool
def get_paper_details(paper_ids: list[str]) -> dict:
    """
    Get full details for one or more papers - use this to summarize a
    single paper or compare several. Any paper not already cached in
    Lakebase is fetched live from OpenAlex and cached for next time.

    Args:
        paper_ids: One or more OpenAlex work IDs, e.g. ["W4389984066"].

    Returns:
        On success: {"status": "success", "papers": [...]}, each with id,
        title, abstract, publication_year, venue, doi, cited_by_count,
        open_access_url, concepts, authors, and a formatted `citation` string.
        On failure (any ID not found): {"status": "error", "message": "..."}.
    """
    return research_logic.get_paper_details(paper_ids)


@mcp.tool
def create_collection(user_email: str, name: str, learning_goal_id: int | None = None) -> dict:
    """
    Create a new paper collection for a user, optionally tied to a learning goal.

    Args:
        user_email: The user's email.
        name: Collection name, e.g. "RAG survey papers".
        learning_goal_id: Optional learning_goals.id this collection supports.

    Returns:
        On success: {"status": "success", "collection_id": int, ...}.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.create_collection(user_email, name, learning_goal_id)


@mcp.tool
def add_paper_to_collection(collection_id: int, paper_id: str) -> dict:
    """
    Add a paper to a collection, fetching and caching it from OpenAlex
    first if it isn't already known to this app.

    Args:
        collection_id: The target collections.id.
        paper_id: OpenAlex work ID, e.g. "W4389984066".

    Returns:
        On success: {"status": "success", "collection_id": ..., "paper": {...}}.
        On failure (bad paper ID, bad collection ID): {"status": "error", "message": "..."}.
    """
    return research_logic.add_paper_to_collection(collection_id, paper_id)


@mcp.tool
def generate_reading_plan(collection_id: int) -> dict:
    """
    Generate a sequenced reading order for every paper in a collection, and
    persist it as collection_papers.sequence_order.

    Not a passthrough - applies a simple pedagogical heuristic: read older,
    foundational work before newer work that builds on it, so papers are
    ordered by publication_year ascending; papers from the same year are
    ordered by cited_by_count descending (more-cited work in that year is
    more likely to be foundational within it). Papers with no publication
    year are placed last.

    Args:
        collection_id: The collections.id to plan.

    Returns:
        On success: {"status": "success", "collection_id": ..., "reading_plan": [...],
        "ordering_rationale": "..."}, where reading_plan is ordered oldest/most
        foundational first.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.generate_reading_plan(collection_id)


@mcp.tool
def get_collection(collection_id: int) -> dict:
    """
    Get a collection's metadata plus every paper in it, in reading-plan
    order (if generated) with each paper's reading progress for the
    collection's owner.

    Args:
        collection_id: The collections.id to look up.

    Returns:
        On success: {"status": "success", "collection": {...}, "papers": [...]}.
        On failure (unknown collection_id): {"status": "error", "message": "..."}.
    """
    return research_logic.get_collection(collection_id)


@mcp.tool
def update_reading_progress(user_email: str, paper_id: str, status: str) -> dict:
    """
    Record a user's reading progress on a paper.

    Args:
        user_email: The user's email.
        paper_id: OpenAlex work ID.
        status: One of "not_started", "in_progress", "completed".

    Returns:
        On success: {"status": "success", "paper_id": ..., "reading_status": ...}.
        On failure (invalid status): {"status": "error", "message": "..."}.
    """
    return research_logic.update_reading_progress(user_email, paper_id, status)


@mcp.tool
def get_next_paper(user_email: str, collection_id: int) -> dict:
    """
    Recommend the next paper to read in a collection: the first paper (in
    reading-plan order, or add order if no plan was generated) that isn't
    already marked "completed" for this user.

    Args:
        user_email: The user's email.
        collection_id: The collections.id to check.

    Returns:
        On success with a paper remaining: {"status": "success", "next_paper": {...}}.
        On success with nothing left: {"status": "success", "next_paper": None, "message": "..."}.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.get_next_paper(user_email, collection_id)


@mcp.tool
def add_note(user_email: str, body: str, paper_id: str | None = None) -> dict:
    """
    Save a study note for a user, optionally tied to a specific paper.
    Embeds the note body so it's part of the same semantically-searchable
    corpus as papers (source_type="note" in content_embeddings).

    Args:
        user_email: The user's email.
        body: The note text.
        paper_id: Optional OpenAlex work ID this note is about.

    Returns:
        On success: {"status": "success", "note_id": int}.
        On failure: {"status": "error", "message": "..."}.
    """
    return research_logic.add_note(user_email, body, paper_id)


if __name__ == "__main__":
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects.
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
