"""
Shared business logic for the AI Research and Learning Copilot.

Both research_mcp_server.py (the agent-facing MCP tools) and the
dashboard's Flask routes need the exact same operations - create a goal,
find papers, generate a reading plan, etc. This module holds the actual
logic once; both call into it instead of duplicating it, even though the
module itself is duplicated across the two app folders (see the note in
papers_repo.py/lakebase.py - each Databricks App deploys independently
from its own folder with no shared package install step).

Every function here returns {"status": "success", ...} or
{"status": "error", "message": "..."} - never raises - so callers in
either app can jsonify/return the result directly.
"""

import logging

import embeddings
import lakebase
import openalex_broker
import papers_repo

logger = logging.getLogger("research-logic")

VALID_PROGRESS_STATUSES = ("not_started", "in_progress", "completed")


def _ensure_user(user_email: str) -> None:
    lakebase.run_write(
        "INSERT INTO users (email) VALUES (%s) ON CONFLICT (email) DO NOTHING",
        (user_email,),
    )


def _format_citation(paper: dict) -> str:
    names = [a.get("display_name") for a in (paper.get("authors") or []) if a.get("display_name")]
    if not names:
        author_str = "Unknown author"
    elif len(names) == 1:
        author_str = names[0]
    elif len(names) == 2:
        author_str = f"{names[0]} & {names[1]}"
    else:
        author_str = f"{names[0]} et al."

    year = paper.get("publication_year") or "n.d."
    citation = f"{author_str} ({year}). {paper['title']}."
    if paper.get("venue"):
        citation += f" {paper['venue']}."
    if paper.get("doi"):
        citation += f" {paper['doi']}"
    return citation.strip()


def create_learning_goal(user_email: str, title: str, description: str, topic_keywords: str) -> dict:
    try:
        _ensure_user(user_email)
        row = lakebase.run_write_returning(
            """
            INSERT INTO learning_goals (user_email, title, description, topic_keywords)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (user_email, title, description, topic_keywords),
        )
        goal_id = row["id"]
        if description:
            embeddings.upsert_embedding("learning_goal", str(goal_id), description, user_email=user_email)
        return {"status": "success", "learning_goal_id": goal_id, "title": title, "topic_keywords": topic_keywords}
    except Exception as e:
        logger.exception("create_learning_goal failed")
        return {"status": "error", "message": f"Could not create learning goal: {e}"}


def list_learning_goals(user_email: str) -> dict:
    try:
        rows = lakebase.run_query(
            "SELECT * FROM learning_goals WHERE user_email = %s ORDER BY created_at DESC", (user_email,)
        )
        return {"status": "success", "learning_goals": rows}
    except Exception as e:
        logger.exception("list_learning_goals failed")
        return {"status": "error", "message": f"Could not list learning goals: {e}"}


def find_papers_for_goal(topic_keywords: str, limit: int = 10) -> dict:
    try:
        papers = openalex_broker.search_works(topic_keywords, limit)
        return {"status": "success", "papers": papers}
    except openalex_broker.OpenAlexError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("find_papers_for_goal failed")
        return {"status": "error", "message": f"Paper search failed: {e}"}


def search_papers_semantic(query: str, limit: int = 8, collection_id: int | None = None) -> dict:
    try:
        paper_ids = None
        if collection_id is not None:
            rows = lakebase.run_query(
                "SELECT paper_id FROM collection_papers WHERE collection_id = %s", (collection_id,)
            )
            paper_ids = [r["paper_id"] for r in rows]
            if not paper_ids:
                return {"status": "success", "results": []}

        results = embeddings.search_similar(query, source_type="paper", limit=limit, paper_ids=paper_ids)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("search_papers_semantic failed")
        return {"status": "error", "message": f"Semantic search failed: {e}"}


def get_paper_details(paper_ids: list[str]) -> dict:
    try:
        papers = []
        for paper_id in paper_ids:
            paper = papers_repo.get_or_fetch_paper(paper_id)
            paper["citation"] = _format_citation(paper)
            papers.append(paper)
        return {"status": "success", "papers": papers}
    except openalex_broker.OpenAlexError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("get_paper_details failed")
        return {"status": "error", "message": f"Could not get paper details: {e}"}


def create_collection(user_email: str, name: str, learning_goal_id: int | None = None) -> dict:
    try:
        _ensure_user(user_email)
        row = lakebase.run_write_returning(
            """
            INSERT INTO collections (user_email, learning_goal_id, name)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (user_email, learning_goal_id, name),
        )
        return {"status": "success", "collection_id": row["id"], "name": name}
    except Exception as e:
        logger.exception("create_collection failed")
        return {"status": "error", "message": f"Could not create collection: {e}"}


def list_collections(user_email: str) -> dict:
    try:
        rows = lakebase.run_query(
            "SELECT * FROM collections WHERE user_email = %s ORDER BY created_at DESC", (user_email,)
        )
        return {"status": "success", "collections": rows}
    except Exception as e:
        logger.exception("list_collections failed")
        return {"status": "error", "message": f"Could not list collections: {e}"}


def add_paper_to_collection(collection_id: int, paper_id: str) -> dict:
    try:
        paper = papers_repo.get_or_fetch_paper(paper_id)
        lakebase.run_write(
            """
            INSERT INTO collection_papers (collection_id, paper_id)
            VALUES (%s, %s)
            ON CONFLICT (collection_id, paper_id) DO NOTHING
            """,
            (collection_id, paper_id),
        )
        return {"status": "success", "collection_id": collection_id, "paper": {"id": paper["id"], "title": paper["title"]}}
    except openalex_broker.OpenAlexError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("add_paper_to_collection failed")
        return {"status": "error", "message": f"Could not add paper to collection: {e}"}


def generate_reading_plan(collection_id: int) -> dict:
    try:
        rows = lakebase.run_query(
            """
            SELECT p.id, p.title, p.publication_year, p.cited_by_count
            FROM collection_papers cp
            JOIN papers p ON p.id = cp.paper_id
            WHERE cp.collection_id = %s
            ORDER BY (p.publication_year IS NULL), p.publication_year ASC, p.cited_by_count DESC
            """,
            (collection_id,),
        )

        for i, row in enumerate(rows, start=1):
            lakebase.run_write(
                "UPDATE collection_papers SET sequence_order = %s WHERE collection_id = %s AND paper_id = %s",
                (i, collection_id, row["id"]),
            )

        reading_plan = [{"sequence_order": i, **row} for i, row in enumerate(rows, start=1)]
        return {
            "status": "success",
            "collection_id": collection_id,
            "reading_plan": reading_plan,
            "ordering_rationale": "Ordered oldest-to-newest by publication year (foundational work first), "
            "tie-broken by citation count within the same year; undated papers are placed last.",
        }
    except Exception as e:
        logger.exception("generate_reading_plan failed")
        return {"status": "error", "message": f"Could not generate reading plan: {e}"}


def get_collection(collection_id: int) -> dict:
    try:
        collections = lakebase.run_query("SELECT * FROM collections WHERE id = %s", (collection_id,))
        if not collections:
            return {"status": "error", "message": f"No collection found with id {collection_id}."}
        collection = collections[0]

        papers = lakebase.run_query(
            """
            SELECT p.id, p.title, p.publication_year, p.cited_by_count, cp.sequence_order,
                   COALESCE(rp.status, 'not_started') AS reading_status
            FROM collection_papers cp
            JOIN papers p ON p.id = cp.paper_id
            LEFT JOIN reading_progress rp ON rp.paper_id = p.id AND rp.user_email = %s
            WHERE cp.collection_id = %s
            ORDER BY (cp.sequence_order IS NULL), cp.sequence_order ASC, cp.added_at ASC
            """,
            (collection["user_email"], collection_id),
        )
        return {"status": "success", "collection": collection, "papers": papers}
    except Exception as e:
        logger.exception("get_collection failed")
        return {"status": "error", "message": f"Could not get collection: {e}"}


def update_reading_progress(user_email: str, paper_id: str, status: str) -> dict:
    if status not in VALID_PROGRESS_STATUSES:
        return {"status": "error", "message": f"status must be one of {VALID_PROGRESS_STATUSES}, got {status!r}."}

    try:
        _ensure_user(user_email)
        lakebase.run_write(
            """
            INSERT INTO reading_progress (user_email, paper_id, status, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (user_email, paper_id) DO UPDATE SET status = EXCLUDED.status, updated_at = now()
            """,
            (user_email, paper_id, status),
        )
        return {"status": "success", "paper_id": paper_id, "reading_status": status}
    except Exception as e:
        logger.exception("update_reading_progress failed")
        return {"status": "error", "message": f"Could not update reading progress: {e}"}


def get_next_paper(user_email: str, collection_id: int) -> dict:
    try:
        rows = lakebase.run_query(
            """
            SELECT p.id, p.title, p.publication_year, cp.sequence_order,
                   COALESCE(rp.status, 'not_started') AS reading_status
            FROM collection_papers cp
            JOIN papers p ON p.id = cp.paper_id
            LEFT JOIN reading_progress rp ON rp.paper_id = p.id AND rp.user_email = %s
            WHERE cp.collection_id = %s AND COALESCE(rp.status, 'not_started') != 'completed'
            ORDER BY (cp.sequence_order IS NULL), cp.sequence_order ASC, cp.added_at ASC
            LIMIT 1
            """,
            (user_email, collection_id),
        )
        if not rows:
            return {"status": "success", "next_paper": None, "message": "Every paper in this collection is marked completed."}
        return {"status": "success", "next_paper": rows[0]}
    except Exception as e:
        logger.exception("get_next_paper failed")
        return {"status": "error", "message": f"Could not get next paper: {e}"}


def add_note(user_email: str, body: str, paper_id: str | None = None) -> dict:
    try:
        _ensure_user(user_email)
        row = lakebase.run_write_returning(
            "INSERT INTO notes (user_email, paper_id, body) VALUES (%s, %s, %s) RETURNING id",
            (user_email, paper_id, body),
        )
        note_id = row["id"]
        embeddings.upsert_embedding("note", str(note_id), body, user_email=user_email)
        return {"status": "success", "note_id": note_id}
    except Exception as e:
        logger.exception("add_note failed")
        return {"status": "error", "message": f"Could not save note: {e}"}


def list_notes(user_email: str, paper_id: str | None = None) -> dict:
    try:
        if paper_id:
            rows = lakebase.run_query(
                "SELECT * FROM notes WHERE user_email = %s AND paper_id = %s ORDER BY created_at DESC",
                (user_email, paper_id),
            )
        else:
            rows = lakebase.run_query(
                "SELECT * FROM notes WHERE user_email = %s ORDER BY created_at DESC", (user_email,)
            )
        return {"status": "success", "notes": rows}
    except Exception as e:
        logger.exception("list_notes failed")
        return {"status": "error", "message": f"Could not list notes: {e}"}
