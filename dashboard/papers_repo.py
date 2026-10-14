"""
Paper cache/repo layer: bridges the live OpenAlex API (openalex_broker.py)
and the Lakebase `papers`/`authors`/`paper_authors` tables.

The `papers` table is a shared, growing cache - populated in bulk by the
offline Spark ingestion notebook (notebooks/ingest_papers_pipeline.py) and
incrementally by the agent itself whenever it looks up or adds a paper
that isn't cached yet. This module is what makes that incremental caching
work, so MCP tool functions in research_mcp_server.py stay thin.
"""

import json

import embeddings
import lakebase
import openalex_broker


def get_cached_paper(paper_id: str) -> dict | None:
    """Read a paper + its authors from Lakebase, or None if not cached yet."""
    rows = lakebase.run_query("SELECT * FROM papers WHERE id = %s", (paper_id,))
    if not rows:
        return None

    paper = dict(rows[0])
    authors = lakebase.run_query(
        """
        SELECT a.id, a.display_name, a.orcid, pa.author_position
        FROM paper_authors pa
        JOIN authors a ON a.id = pa.author_id
        WHERE pa.paper_id = %s
        ORDER BY pa.author_position
        """,
        (paper_id,),
    )
    paper["authors"] = authors
    return paper


def cache_paper(paper: dict) -> None:
    """
    Upsert a parsed OpenAlex paper (as returned by openalex_broker) into
    papers/authors/paper_authors, and embed its abstract into
    content_embeddings if it has one.
    """
    lakebase.run_write(
        """
        INSERT INTO papers (id, title, abstract, publication_year, venue, doi,
                             cited_by_count, open_access_url, concepts, payload, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            cited_by_count = EXCLUDED.cited_by_count,
            fetched_at = now()
        """,
        (
            paper["id"],
            paper["title"],
            paper.get("abstract"),
            paper.get("publication_year"),
            paper.get("venue"),
            paper.get("doi"),
            paper.get("cited_by_count", 0),
            paper.get("open_access_url"),
            ", ".join(paper.get("concepts") or []),
            json.dumps(paper),
        ),
    )

    for author in paper.get("authors") or []:
        if not author.get("id"):
            continue
        lakebase.run_write(
            """
            INSERT INTO authors (id, display_name, orcid, works_count, cited_by_count)
            VALUES (%s, %s, %s, NULL, NULL)
            ON CONFLICT (id) DO UPDATE SET display_name = EXCLUDED.display_name
            """,
            (author["id"], author.get("display_name"), author.get("orcid")),
        )
        lakebase.run_write(
            """
            INSERT INTO paper_authors (paper_id, author_id, author_position)
            VALUES (%s, %s, %s)
            ON CONFLICT (paper_id, author_id) DO NOTHING
            """,
            (paper["id"], author["id"], author.get("author_position")),
        )

    if paper.get("abstract"):
        embeddings.upsert_embedding("paper", paper["id"], paper["abstract"])


def get_or_fetch_paper(paper_id: str) -> dict:
    """
    Get a paper by ID from the Lakebase cache, fetching and caching it
    from OpenAlex first if it isn't already there.

    Raises:
        openalex_broker.OpenAlexError if the paper isn't cached and can't
        be found on OpenAlex either.
    """
    cached = get_cached_paper(paper_id)
    if cached is not None:
        return cached

    paper = openalex_broker.get_work(paper_id)
    cache_paper(paper)
    return paper
