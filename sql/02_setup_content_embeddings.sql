-- Vector store for all embedded unstructured text in this app: paper
-- abstracts, user notes, and learning goal descriptions. One shared table
-- (distinguished by source_type) instead of three near-identical tables,
-- so semantic search can span content types when useful.
--
-- Run this manually in your Lakebase Postgres database.
-- Replace {{EMBEDDING_DIM}} with your model's output dimension (384 for
-- the default sentence-transformers/all-MiniLM-L6-v2).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS content_embeddings (
    id TEXT PRIMARY KEY,             -- "paper_<paper_id>" | "note_<note_id>" | "goal_<goal_id>"
    source_type TEXT NOT NULL,       -- 'paper' | 'note' | 'learning_goal'
    source_id TEXT NOT NULL,         -- papers.id | notes.id | learning_goals.id
    user_email TEXT,                 -- NULL for papers (shared/global content)
    text TEXT NOT NULL,               -- the exact text that was embedded
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_content_embeddings_source_type ON content_embeddings(source_type);

-- HNSW index for fast cosine similarity search.
CREATE INDEX IF NOT EXISTS idx_content_embeddings_embedding
ON content_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Verify the table was created
SELECT table_name, column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'content_embeddings'
ORDER BY ordinal_position;
