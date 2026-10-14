# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Papers from OpenAlex -> Lakebase (Spark pipeline)
# MAGIC
# MAGIC This is the Spark data pipeline for the **AI Research and Learning
# MAGIC Copilot** capstone. It:
# MAGIC
# MAGIC 1. Reads distinct `topic_keywords` from the `learning_goals` table in
# MAGIC    Lakebase (falling back to a widget-configurable seed list of topics
# MAGIC    so the corpus isn't empty before any user has created a goal).
# MAGIC 2. Searches the OpenAlex API for papers matching each topic (no API key
# MAGIC    required) and builds Spark DataFrames for `papers`, `authors`, and
# MAGIC    `paper_authors`.
# MAGIC 3. Computes an embedding for each paper's abstract **distributed across
# MAGIC    the Spark cluster** via `mapInPandas` (each partition loads the
# MAGIC    sentence-transformers model once and encodes its slice of papers in
# MAGIC    parallel with every other partition).
# MAGIC 4. Upserts everything into Lakebase: `papers`/`authors`/`paper_authors`
# MAGIC    via batched psycopg2 `executemany`, and abstract embeddings into the
# MAGIC    pgvector-backed `content_embeddings` table.
# MAGIC
# MAGIC This is the same "read raw data as Spark DataFrames, do the heavy
# MAGIC compute in Spark, upsert into Postgres via psycopg2" shape as Day 2's
# MAGIC `ingest_ticker_news_embeddings.py`, but the embedding step here is a
# MAGIC genuine distributed `mapInPandas` stage rather than a driver-only loop.
# MAGIC
# MAGIC It reuses the same Lakebase secret (scope `database`, key
# MAGIC `lakebase-url`) that `lakebase.py` uses in both apps - no extra secrets
# MAGIC needed for this notebook, since OpenAlex requires no API key.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip install -q 'databricks-sdk>=0.30.0' sentence-transformers requests pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override table names, the embedding model, and the
# MAGIC seed topics without editing the notebook - useful when running this as
# MAGIC a scheduled Databricks Job (see `resources/ingest_papers_pipeline_job.yml`).

# COMMAND ----------

dbutils.widgets.text("learning_goals_table_name", "learning_goals", "Source table (learning goal topics)")
dbutils.widgets.text("papers_table_name", "papers", "Destination table (papers)")
dbutils.widgets.text("authors_table_name", "authors", "Destination table (authors)")
dbutils.widgets.text("paper_authors_table_name", "paper_authors", "Destination table (paper<->author)")
dbutils.widgets.text("content_embeddings_table_name", "content_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text(
    "seed_topics",
    "large language models, retrieval augmented generation, transformer architecture, vector databases",
    "Seed topics (comma-separated, used if learning_goals is empty)",
)
dbutils.widgets.text("papers_per_topic", "15", "Max papers to fetch per topic")
dbutils.widgets.text("openalex_mailto", "surajgoel1997@gmail.com", "Contact email for OpenAlex's polite pool")
dbutils.widgets.text("embedding_partitions", "4", "Spark partitions for the distributed embedding step")

LEARNING_GOALS_TABLE_NAME = dbutils.widgets.get("learning_goals_table_name")
PAPERS_TABLE_NAME = dbutils.widgets.get("papers_table_name")
AUTHORS_TABLE_NAME = dbutils.widgets.get("authors_table_name")
PAPER_AUTHORS_TABLE_NAME = dbutils.widgets.get("paper_authors_table_name")
CONTENT_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("content_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
SEED_TOPICS = dbutils.widgets.get("seed_topics")
PAPERS_PER_TOPIC = int(dbutils.widgets.get("papers_per_topic"))
OPENALEX_MAILTO = dbutils.widgets.get("openalex_mailto")
EMBEDDING_PARTITIONS = int(dbutils.widgets.get("embedding_partitions"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly - see sql/02_setup_content_embeddings.sql.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py` in both apps: a
# MAGIC single base64-encoded Postgres URL stored in a Databricks secret scope.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase connection info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


parsed = urlparse(get_lakebase_url())
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

print(f"Connection details: {db_user}@{db_host}:{db_port}/{db_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database setup instructions
# MAGIC
# MAGIC Before running this notebook, run these once against your Lakebase
# MAGIC Postgres database:
# MAGIC 1. `sql/01_setup_core_tables.sql`
# MAGIC 2. `sql/02_setup_content_embeddings.sql` (replace `{{EMBEDDING_DIM}}` with the value printed above)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick topics to search for
# MAGIC
# MAGIC Reads distinct `topic_keywords` from every active learning goal across
# MAGIC all users, so the corpus grows to cover whatever people are actually
# MAGIC studying. Falls back to `seed_topics` on a fresh install where no
# MAGIC learning goal exists yet.

# COMMAND ----------

# DBTITLE 1,Collect search topics
import psycopg2


def get_connection():
    return psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password, sslmode="require"
    )


def get_topics_from_learning_goals() -> list[str]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT topic_keywords FROM {LEARNING_GOALS_TABLE_NAME} WHERE status = 'active'")
        rows = cursor.fetchall()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"Could not read {LEARNING_GOALS_TABLE_NAME} (does it exist yet?): {e}")
        return []
    finally:
        conn.close()


goal_topic_strings = get_topics_from_learning_goals()
topics = set()
for topic_string in goal_topic_strings or [SEED_TOPICS]:
    topics.update(t.strip() for t in topic_string.split(",") if t.strip())
topics = sorted(topics)

print(f"Searching OpenAlex for {len(topics)} topics: {topics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch papers from OpenAlex
# MAGIC
# MAGIC One search request per topic (serial, since OpenAlex's free tier is
# MAGIC generous - ~100k requests/day - and this notebook only issues a
# MAGIC handful per run). Duplicate of the abstract-reconstruction and
# MAGIC work-parsing logic in `mcp_server/openalex_broker.py` - notebooks in
# MAGIC this repo don't import app code, since each Databricks App/Job deploys
# MAGIC independently from its own folder.

# COMMAND ----------

# DBTITLE 1,Search OpenAlex per topic
import requests

OPENALEX_BASE_URL = "https://api.openalex.org"


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    max_position = max(pos for positions in inverted_index.values() for pos in positions)
    out_words = [""] * (max_position + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            out_words[pos] = word
    return " ".join(out_words).strip() or None


def parse_work(work: dict) -> dict:
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
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_year": work.get("publication_year"),
        "venue": source.get("display_name"),
        "doi": work.get("doi"),
        "cited_by_count": work.get("cited_by_count", 0),
        "open_access_url": open_access.get("oa_url"),
        "concepts": [c["display_name"] for c in concepts[:5] if c.get("display_name")],
        "authors": authors,
        "raw": work,
    }


def search_openalex(topic: str, limit: int) -> list[dict]:
    resp = requests.get(
        f"{OPENALEX_BASE_URL}/works",
        params={"search": topic, "per_page": limit, "mailto": OPENALEX_MAILTO},
        timeout=30,
    )
    resp.raise_for_status()
    return [parse_work(w) for w in resp.json().get("results", [])]


all_papers = {}  # dedup by paper id across topics
for topic in topics:
    try:
        for paper in search_openalex(topic, PAPERS_PER_TOPIC):
            all_papers[paper["id"]] = paper
    except Exception as e:
        print(f"Skipping topic {topic!r}: {e}")

papers_list = list(all_papers.values())
print(f"Fetched {len(papers_list)} distinct papers across {len(topics)} topics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Spark DataFrames
# MAGIC
# MAGIC Flatten the fetched papers into `papers`, `authors`, and
# MAGIC `paper_authors` Spark DataFrames.

# COMMAND ----------

# DBTITLE 1,Build papers/authors/paper_authors DataFrames
import json

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

papers_rows = [
    (
        p["id"],
        p["title"],
        p["abstract"],
        p["publication_year"],
        p["venue"],
        p["doi"],
        p["cited_by_count"],
        p["open_access_url"],
        ", ".join(p["concepts"]),
        json.dumps(p["raw"]),
    )
    for p in papers_list
]
papers_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("abstract", StringType(), True),
        StructField("publication_year", IntegerType(), True),
        StructField("venue", StringType(), True),
        StructField("doi", StringType(), True),
        StructField("cited_by_count", IntegerType(), True),
        StructField("open_access_url", StringType(), True),
        StructField("concepts", StringType(), True),
        StructField("payload_json", StringType(), True),
    ]
)
papers_df = spark.createDataFrame(papers_rows, schema=papers_schema)

authors_rows = [
    (a["id"], a["display_name"], a["orcid"])
    for p in papers_list
    for a in p["authors"]
]
authors_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("display_name", StringType(), True),
        StructField("orcid", StringType(), True),
    ]
)
authors_df = spark.createDataFrame(authors_rows, schema=authors_schema).dropDuplicates(["id"])

paper_authors_rows = [
    (p["id"], a["id"], a["author_position"])
    for p in papers_list
    for a in p["authors"]
]
paper_authors_schema = StructType(
    [
        StructField("paper_id", StringType(), False),
        StructField("author_id", StringType(), False),
        StructField("author_position", IntegerType(), True),
    ]
)
paper_authors_df = spark.createDataFrame(paper_authors_rows, schema=paper_authors_schema)

print(f"papers_df: {papers_df.count()} rows, authors_df: {authors_df.count()} rows, paper_authors_df: {paper_authors_df.count()} rows")
display(papers_df.select("id", "title", "publication_year", "cited_by_count").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute abstract embeddings (distributed across the Spark cluster)
# MAGIC
# MAGIC `mapInPandas` runs the enclosed function once per partition, on
# MAGIC whichever executor owns that partition - each executor loads its own
# MAGIC copy of the sentence-transformers model and encodes only its slice of
# MAGIC papers, in parallel with every other partition. Repartitioning first
# MAGIC controls how many executors actually get work (useful to see
# MAGIC parallelism even on a small demo cluster/corpus).

# COMMAND ----------

# DBTITLE 1,Distributed embedding computation via mapInPandas
from pyspark.sql.types import ArrayType, DoubleType

papers_with_abstract_df = (
    papers_df.select("id", "title", "abstract")
    .filter("abstract IS NOT NULL AND abstract != ''")
    .repartition(EMBEDDING_PARTITIONS)
)


def encode_partition(pdf_iterator):
    import os

    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
    for pdf in pdf_iterator:
        vectors = model.encode(pdf["abstract"].tolist(), show_progress_bar=False)
        pdf["embedding"] = [v.tolist() for v in vectors]
        yield pdf


embeddings_schema = papers_with_abstract_df.schema.add("embedding", ArrayType(DoubleType()))
embeddings_df = papers_with_abstract_df.mapInPandas(encode_partition, schema=embeddings_schema)

embeddings_pdf = embeddings_df.toPandas()
print(f"Computed {len(embeddings_pdf)} abstract embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert papers, authors, and paper_authors into Lakebase
# MAGIC
# MAGIC Written via psycopg2 `executemany` with `ON CONFLICT` upserts - Spark's
# MAGIC JDBC writer doesn't support upserts or the `vector` column type
# MAGIC directly, so (matching Day 2's `ingest_ticker_news_embeddings.py`
# MAGIC convention) the Spark-built DataFrames are collected to the driver and
# MAGIC written with psycopg2 for this final step.

# COMMAND ----------

# DBTITLE 1,Upsert papers/authors/paper_authors
papers_pdf = papers_df.toPandas()
authors_pdf = authors_df.toPandas()
paper_authors_pdf = paper_authors_df.toPandas()

conn = get_connection()
try:
    cursor = conn.cursor()

    cursor.executemany(
        f"""
        INSERT INTO {PAPERS_TABLE_NAME}
            (id, title, abstract, publication_year, venue, doi, cited_by_count, open_access_url, concepts, payload, fetched_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title, abstract = EXCLUDED.abstract,
            cited_by_count = EXCLUDED.cited_by_count, fetched_at = now()
        """,
        list(papers_pdf.itertuples(index=False, name=None)),
    )
    print(f"Upserted {len(papers_pdf)} papers")

    cursor.executemany(
        f"""
        INSERT INTO {AUTHORS_TABLE_NAME} (id, display_name, orcid, works_count, cited_by_count)
        VALUES (%s, %s, %s, NULL, NULL)
        ON CONFLICT (id) DO UPDATE SET display_name = EXCLUDED.display_name
        """,
        list(authors_pdf.itertuples(index=False, name=None)),
    )
    print(f"Upserted {len(authors_pdf)} authors")

    cursor.executemany(
        f"""
        INSERT INTO {PAPER_AUTHORS_TABLE_NAME} (paper_id, author_id, author_position)
        VALUES (%s, %s, %s)
        ON CONFLICT (paper_id, author_id) DO NOTHING
        """,
        list(paper_authors_pdf.itertuples(index=False, name=None)),
    )
    print(f"Upserted {len(paper_authors_pdf)} paper-author links")

    conn.commit()
finally:
    cursor.close()
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert abstract embeddings into content_embeddings

# COMMAND ----------

# DBTITLE 1,Upsert abstract embeddings
conn = get_connection()
try:
    cursor = conn.cursor()

    rows = [
        (
            f"paper_{row.id}",
            "paper",
            row.id,
            None,  # user_email: NULL for shared/global paper content
            row.abstract,
            "[" + ",".join(str(float(x)) for x in row.embedding) + "]",
            EMBEDDING_MODEL_NAME,
        )
        for row in embeddings_pdf.itertuples(index=False)
    ]

    cursor.executemany(
        f"""
        INSERT INTO {CONTENT_EMBEDDINGS_TABLE_NAME}
            (id, source_type, source_id, user_email, text, embedding, model_name, embedded_at)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text, embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name, embedded_at = now()
        """,
        rows,
    )
    conn.commit()
    print(f"Upserted {len(rows)} paper abstract embeddings into {CONTENT_EMBEDDINGS_TABLE_NAME}")
finally:
    cursor.close()
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC The `papers`/`authors`/`paper_authors` tables now have fresh OpenAlex
# MAGIC data for every active learning goal's topics, and `content_embeddings`
# MAGIC has an up-to-date vector for every paper with an abstract - ready for
# MAGIC `search_papers_semantic` in the MCP server and the dashboard's search
# MAGIC page.
