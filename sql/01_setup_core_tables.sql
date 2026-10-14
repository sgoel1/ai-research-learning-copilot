-- Core relational schema for the AI Research and Learning Copilot.
-- Run this once against your Lakebase Postgres database before running the
-- ingestion notebook or deploying either app.
--
-- Identity convention (matches the Day 3 reference apps in this repo):
-- users are identified by email, taken from the Databricks-injected
-- X-Forwarded-Email header in production or a local default in dev.

CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_goals (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES users(email),
    title TEXT NOT NULL,
    description TEXT,
    -- Comma-separated search terms used to find papers on OpenAlex for this
    -- goal, e.g. "retrieval augmented generation, vector databases".
    topic_keywords TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- active | completed | archived
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_learning_goals_user ON learning_goals(user_email);

-- One row per OpenAlex "work" (paper). Shared/global - not owned by a user.
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,            -- OpenAlex work ID, e.g. "W2741809807"
    title TEXT NOT NULL,
    abstract TEXT,
    publication_year INT,
    venue TEXT,
    doi TEXT,
    cited_by_count INT DEFAULT 0,
    open_access_url TEXT,
    concepts TEXT,                  -- comma-separated top OpenAlex concept names
    payload JSONB,                  -- raw OpenAlex work object, for reference
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_papers_publication_year ON papers(publication_year);

CREATE TABLE IF NOT EXISTS authors (
    id TEXT PRIMARY KEY,            -- OpenAlex author ID, e.g. "A5023888391"
    display_name TEXT NOT NULL,
    orcid TEXT,
    works_count INT,
    cited_by_count INT
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id TEXT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    author_position INT,            -- 0-indexed position in the paper's author list
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS collections (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES users(email),
    learning_goal_id INT REFERENCES learning_goals(id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_email);

CREATE TABLE IF NOT EXISTS collection_papers (
    collection_id INT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    -- Position in the agent-generated sequenced reading plan; NULL until
    -- generate_reading_plan has been run for this collection.
    sequence_order INT,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, paper_id)
);

CREATE TABLE IF NOT EXISTS reading_progress (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES users(email),
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'not_started',  -- not_started | in_progress | completed
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_email, paper_id)
);

CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES users(email),
    paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,  -- nullable: a note can be general
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_email);
CREATE INDEX IF NOT EXISTS idx_notes_paper ON notes(paper_id);
