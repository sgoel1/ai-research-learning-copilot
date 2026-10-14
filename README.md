# AI Research and Learning Copilot

Capstone project for the Databricks AI Bootcamp, built from
[idea 3](https://github.com/EcZachly/databricks-ai-bootcamp-capstone#3-ai-research-and-learning-copilot)
of the capstone ideas list, following the same architectural pattern as this repo's Day 3
reference app (`Day3/databricks-lakebase-app-day-3`): **Lakebase** for relational + vector
storage, a **Spark data pipeline** for bulk ingestion, an **MCP server** exposing agent tools,
and a **Databricks App frontend**.

A user sets a learning goal (e.g. "understand retrieval-augmented generation"), discovers papers
for it, organizes them into collections, and asks an agent to summarize/compare research, build a
sequenced reading plan, and track their progress - with every answer grounded in real paper data
and citations, never hallucinated.

## Capstone requirements checklist

- ✅ **Data pipeline in Spark** - `notebooks/ingest_papers_pipeline.py` searches OpenAlex for
  papers matching active learning goals and computes abstract embeddings via a distributed
  `mapInPandas` stage (see [Spark pipeline](#spark-pipeline) below).
- ✅ **Third-party API** - [OpenAlex](https://docs.openalex.org/) (papers, authors, abstracts,
  citations). No API key required.
- ✅ **Unstructured data processing** - paper abstracts, user notes, and learning goal
  descriptions are all embedded into a shared pgvector table (`content_embeddings`) for semantic
  retrieval.
- ✅ **Databricks App with a frontend** - `dashboard/` (Flask UI: goals, collections, paper
  discovery, semantic search, reading plans, progress, notes).
- ✅ **AI agent with tools that read AND write** - `mcp_server/research_mcp_server.py` exposes 11
  tools to a Databricks Agent Bricks agent; 6 are pure reads/searches, 5 write to Lakebase.

## Architecture

```
Agent Bricks agent  --(MCP tool calls)-->        mcp_server/research_mcp_server.py
dashboard/app.py    --(direct calls, same logic)--> research_logic.py

research_logic.py is the only file either entry point imports; everything below is reached
through it, never bypassed:
  research_logic.py --(get / add paper)-->         papers_repo.py
  research_logic.py --(embed + search)-->          embeddings.py
  research_logic.py --(direct CRUD)-->             lakebase.py
  research_logic.py --(live search, uncached)-->   openalex_broker.py --(REST)--> OpenAlex API

papers_repo.py --(fetch by id)-->     openalex_broker.py --(REST)--> OpenAlex API
papers_repo.py --(embed abstract)-->  embeddings.py
papers_repo.py --(cache paper)-->     lakebase.py

embeddings.py --(vector I/O)-->        lakebase.py
lakebase.py   --(psycopg2 + pgvector)--> Lakebase (Postgres + pgvector)

notebooks/ingest_papers_pipeline.py  (Spark, scheduled Job)
  Lakebase (read learning_goals topics)
    --(REST)--> OpenAlex API
    --> Spark DataFrames (normalize & deduplicate)
    --> mapInPandas (distributed embeddings)
    --> Lakebase (write papers + embeddings)
```

- `mcp_server/` and `dashboard/` are **two separate Databricks Apps**, same split as the Day 3
  reference app - one serves MCP tool calls to the agent, the other serves the human-facing UI.
  `research_logic.py` is duplicated into each folder, since each Databricks App deploys
  independently with no shared package install step, so the agent and the UI can never drift
  into different behavior for the same operation.
- `research_logic.py` holds all the actual business logic; `research_mcp_server.py`'s
  `@mcp.tool` functions and `dashboard/app.py`'s Flask routes are both thin wrappers around it.
- `papers_repo.py` bridges OpenAlex and Lakebase: it caches any paper looked up or added into the
  `papers`/`authors`/`paper_authors` tables and embeds its abstract, so the corpus grows
  incrementally from agent/UI activity, not just from the offline Spark job.
- `embeddings.py` is the pgvector adapter (upsert + cosine-similarity search) shared by both apps
  and the offline pipeline's insert step.

## Lakebase schema

| Table | Purpose |
|---|---|
| `users` | Identity, keyed by email (same convention as the Day 3 reference apps). |
| `learning_goals` | A user's stated learning objective + topic keywords used to search OpenAlex. |
| `papers` | Shared cache of OpenAlex works (title, abstract, year, venue, DOI, citations, raw payload). |
| `authors` | Shared cache of OpenAlex authors. |
| `paper_authors` | Paper ↔ author join table, with author order preserved. |
| `collections` | A user's named group of papers, optionally tied to a learning goal. |
| `collection_papers` | Collection ↔ paper join table; carries the generated reading-plan `sequence_order`. |
| `reading_progress` | Per-user, per-paper status: `not_started` / `in_progress` / `completed`. |
| `notes` | Free-text study notes, optionally tied to a paper. |
| `content_embeddings` | pgvector store for all embedded text (papers, notes, learning goals), one shared table distinguished by `source_type`. |

See `sql/01_setup_core_tables.sql` and `sql/02_setup_content_embeddings.sql`.

## Spark pipeline

`notebooks/ingest_papers_pipeline.py`:
1. Reads distinct `topic_keywords` across every active learning goal (falls back to a seed topic
   list on a fresh install with no goals yet).
2. Searches OpenAlex for each topic and builds `papers`/`authors`/`paper_authors` **Spark
   DataFrames** (`spark.createDataFrame`).
3. Computes an embedding for every paper's abstract in a **distributed `mapInPandas` stage** -
   each partition loads its own copy of the sentence-transformers model and encodes its slice of
   papers in parallel with every other partition (not a driver-only loop).
4. Upserts papers/authors/links via batched `psycopg2.executemany`, and abstract embeddings into
   `content_embeddings` (Spark's JDBC writer doesn't support `ON CONFLICT` upserts or the
   `vector` column type, so the final write step uses psycopg2, same as this repo's Day 2
   reference notebook).

Scheduled as a Databricks Workflow via `databricks.yml` + `resources/ingest_papers_pipeline_job.yml`
(Databricks Asset Bundle) - deploy with `databricks bundle deploy -t dev`, or create the
equivalent job by hand in the Workflows UI (see [Day 3's README](../../Day3/databricks-lakebase-app-day-3/README.md)
for the no-CLI walkthrough of that same pattern).

## MCP tools

| Tool | Type | Purpose |
|---|---|---|
| `create_learning_goal` | write | Create a learning goal (embeds its description). |
| `find_papers_for_goal` | read (live API) | Search OpenAlex for papers matching topic keywords. |
| `search_papers_semantic` | read (pgvector) | Semantic search over cached paper abstracts, optionally scoped to a collection - retrieves relevant evidence instead of sending an entire collection to the model. |
| `get_paper_details` | read + caches | Summarize one paper or compare several; returns a formatted citation for each. Caches any paper not already known. |
| `create_collection` | write | Create a paper collection, optionally tied to a goal. |
| `add_paper_to_collection` | write | Add a paper to a collection (fetches + caches it from OpenAlex first if needed). |
| `generate_reading_plan` | write (derived) | Order a collection's papers oldest/most-foundational first (see docstring for the exact heuristic) and persist the order. |
| `get_collection` | read | Get a collection plus every paper's reading-plan position and progress. |
| `update_reading_progress` | write | Mark a paper `not_started` / `in_progress` / `completed` for a user. |
| `get_next_paper` | read (derived) | Recommend the next unfinished paper in a collection, in plan order. |
| `add_note` | write + embeds | Save a study note, embedding it into the same searchable corpus as papers. |

All 11 tools were smoke-tested locally (tool registration, MCP schema round-trips, and control
flow against a stubbed database) before deployment.

## Setup

### 1. Create a Lakebase instance and run the SQL setup scripts

Follow [Day 2's Lakebase setup](../../Day2/databricks-lakebase-app-day-2/README.md#2-create-a-lakebase-instance-and-a-native-password-role)
to create an instance and a native-password role, then run, in order, against it:
1. `sql/01_setup_core_tables.sql`
2. `sql/02_setup_content_embeddings.sql` (replace `{{EMBEDDING_DIM}}` with `384` for the default
   `sentence-transformers/all-MiniLM-L6-v2` model)

### 2. Store the Lakebase secret

From a Databricks notebook: `%sh python setup_secrets.py` (prompts for your Lakebase connection
URL via `getpass`, stores it as `database/lakebase-url`). OpenAlex needs no key, so this is the
only secret required.

### 3. Configure environment variables (local dev)

```bash
cp .env.example .env
```

### 4. Install dependencies and run both apps locally

```bash
cd mcp_server && pip install -r requirements.txt && python research_mcp_server.py   # serves MCP on :8000
```

In a second terminal:

```bash
cd dashboard && pip install -r requirements.txt && python app.py                    # serves UI on :8001
```

Open `http://localhost:8001`, create a learning goal, then use "Discover Papers" to search
OpenAlex and add results to a new collection.

### 5. Run the Spark ingestion pipeline

Import `notebooks/ingest_papers_pipeline.py` into a Databricks workspace and run it on a cluster
(needs the `spark`/`dbutils` globals a notebook runtime provides), or deploy it as a scheduled
Workflow: `databricks bundle deploy -t dev` then `databricks bundle run ingest_papers_pipeline_job -t dev`.

### 6. Deploy both apps to Databricks Apps

Same pattern as [Day 3's deploy steps](../../Day3/databricks-lakebase-app-day-3/README.md#5-deploy-both-apps-to-databricks-apps):
create a Git folder for this repo, then create two Databricks Apps pointed at this project's
`mcp_server/` and `dashboard/` subfolders respectively.

### 7. Register the MCP server as an external MCP

Follow [Connect agents to external MCPs and tools](https://docs.databricks.com/aws/en/agents/mcp-tools/connect-external):
**AI Gateway** > **MCPs** > **Add MCP**, paste the MCP app's URL, and it will list all 11 tools.

### 8. Build the Agent Bricks agent

**Agents** > **Agent Bricks** > **Create agent** > **Custom LLM** agent type. Add the registered
MCP server under **Tools** (all 11 tools). System prompt:

> You are a research and learning copilot. You help users set learning goals, discover and
> organize academic papers, and follow a structured reading plan. Ground every factual claim
> about a paper (its findings, authors, year, or citation count) in a tool call - never invent
> paper details, abstracts, or citations. Use `find_papers_for_goal` to discover new candidates
> from OpenAlex, and `search_papers_semantic` to retrieve relevant evidence from papers already
> in the system before answering questions about topics the user has already collected. Use
> `get_paper_details` to summarize a single paper or compare several, and always include the
> citation it returns when referencing a paper. Use `generate_reading_plan` (not your own
> ordering) when asked to sequence a collection, and explain the rationale it returns. Use
> `get_next_paper` for "what should I read next" questions rather than guessing. Call
> `update_reading_progress` whenever the user says they've started or finished a paper. If a
> tool call fails (unresolvable paper ID, OpenAlex outage, etc.), tell the user what went wrong
> and ask them to clarify or retry, rather than making up an answer.

Evaluate with sample prompts (e.g. "Find me papers on retrieval-augmented generation and start a
collection", "Compare these two papers", "What should I read next?"), then deploy and chat with
the agent.

## Demo: agent in action

*(Paste 3+ example questions and the agent's tool-calling + final answers here after deploying,
per the submission checklist - e.g. "I want to learn about retrieval-augmented generation, find
me some papers", "Compare the top two results on their approach", "Build me a reading plan and
tell me what to read first".)*

## Notes

- `research_logic.py`, `lakebase.py`, `embeddings.py`, `openalex_broker.py`, and `papers_repo.py`
  are intentionally duplicated between `mcp_server/` and `dashboard/`, for the same reason as the
  Day 3 reference app: each Databricks App deploys independently from its own folder with its own
  `app.yaml`/`requirements.txt`.
- `generate_reading_plan`'s ordering heuristic (oldest-and-most-cited-first) is a simple,
  explainable pedagogical default, not definitive research pedagogy - documented in the
  function's docstring in `research_logic.py`.
- OpenAlex's abstracts are stored as an inverted index rather than plain text (to route around
  publisher copyright restrictions on redistributing abstract text); `openalex_broker.py`
  reconstructs plain text from it before embedding or displaying it.
- `papers`, `authors`, and `content_embeddings` (for papers) are shared/global, not
  per-user - once one user's learning goal pulls a paper into the corpus, it's immediately
  available for every other user's semantic search too.
