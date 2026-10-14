"""
Real-time embedding helper for the AI Research and Learning Copilot.

The bulk of paper abstracts are embedded offline by the Spark ingestion
notebook (notebooks/ingest_papers_pipeline.py). This module handles the
other side: embedding a single piece of text (a new note, a new learning
goal, or a paper fetched on-demand via add_paper_to_collection) at
request time, using the same model and the same content_embeddings
table, so newly-created content is searchable immediately rather than
waiting for the next batch run.
"""

import os

import lakebase

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None


def get_model():
    """Lazy-load the embedding model (expensive - only load on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Compute a single embedding vector for `text`."""
    return get_model().encode(text).tolist()


def upsert_embedding(source_type: str, source_id: str, text: str, user_email: str | None = None) -> None:
    """
    Embed `text` and upsert it into content_embeddings, keyed by
    "<source_type>_<source_id>" (e.g. "paper_W123", "note_42").

    Args:
        source_type: "paper", "note", or "learning_goal".
        source_id: The primary key of the source row (papers.id, notes.id,
            or learning_goals.id).
        text: The exact text to embed (abstract, note body, or goal description).
        user_email: Owning user's email; None for shared content (papers).
    """
    embedding = embed_text(text)
    lakebase.run_write(
        """
        INSERT INTO content_embeddings (id, source_type, source_id, user_email, text, embedding, model_name, embedded_at)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            embedded_at = now()
        """,
        (f"{source_type}_{source_id}", source_type, str(source_id), user_email, text, str(embedding), EMBEDDING_MODEL_NAME),
    )


def search_similar(query: str, source_type: str | None = None, limit: int = 8, paper_ids: list[str] | None = None) -> list[dict]:
    """
    Semantic search over content_embeddings via pgvector cosine similarity.

    Args:
        query: Free-text search query.
        source_type: Restrict to "paper", "note", or "learning_goal"; None searches all.
        limit: Max number of results.
        paper_ids: If given (and source_type="paper"), restrict results to
            this set of paper IDs - e.g. searching only within one collection
            instead of the whole corpus.

    Returns:
        A list of dicts with source_type, source_id, text, and similarity
        (0-1, higher is closer), most similar first.
    """
    embedding = embed_text(query)

    sql = """
        SELECT source_type, source_id, text, 1 - (embedding <=> %s::vector) AS similarity
        FROM content_embeddings
        WHERE 1 = 1
    """
    params: list = [str(embedding)]

    if source_type:
        sql += " AND source_type = %s"
        params.append(source_type)

    if paper_ids:
        sql += " AND source_id = ANY(%s)"
        params.append(paper_ids)

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([str(embedding), limit])

    return lakebase.run_query(sql, tuple(params))
