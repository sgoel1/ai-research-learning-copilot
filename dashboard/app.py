"""
AI Research and Learning Copilot dashboard: the human-facing Databricks
App frontend. Lets a user create learning goals, discover and collect
papers, generate a reading plan, track progress, and jot notes - the same
actions the Agent Bricks agent can take via research_mcp_server.py, but
through a UI instead of natural language.

Calls research_logic.py directly (not through MCP) - same broker/logic
modules, duplicated into this folder because each Databricks App deploys
independently from its own folder with no shared package install step
(see the note in papers_repo.py).

Run locally:
    python app.py
"""

import os

from flask import Flask, jsonify, render_template, request

import research_logic

app = Flask(__name__)

DEFAULT_USER_EMAIL = os.environ.get("DEFAULT_USER_EMAIL", "student@example.com")


def _user_email() -> str:
    """
    Resolve the current user's email: prefer the Databricks-injected
    X-Forwarded-Email header (real end-user identity in production), then
    a query/body param (useful for local dev/testing), then a default.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    body = request.get_json(silent=True) or {}
    return request.args.get("user_email") or body.get("user_email") or DEFAULT_USER_EMAIL


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page)."""
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"status": "error", "message": str(err)}), status_code


@app.route("/")
def index():
    return render_template("index.html", default_user_email=DEFAULT_USER_EMAIL)


@app.route("/api/goals", methods=["GET", "POST"])
def api_goals():
    user_email = _user_email()
    if request.method == "POST":
        body = request.get_json(force=True)
        result = research_logic.create_learning_goal(
            user_email, body["title"], body.get("description", ""), body["topic_keywords"]
        )
        return jsonify(result)
    return jsonify(research_logic.list_learning_goals(user_email))


@app.route("/api/collections", methods=["GET", "POST"])
def api_collections():
    user_email = _user_email()
    if request.method == "POST":
        body = request.get_json(force=True)
        result = research_logic.create_collection(user_email, body["name"], body.get("learning_goal_id"))
        return jsonify(result)
    return jsonify(research_logic.list_collections(user_email))


@app.route("/api/collections/<int:collection_id>")
def api_collection_detail(collection_id):
    return jsonify(research_logic.get_collection(collection_id))


@app.route("/api/collections/<int:collection_id>/papers", methods=["POST"])
def api_add_paper(collection_id):
    body = request.get_json(force=True)
    return jsonify(research_logic.add_paper_to_collection(collection_id, body["paper_id"]))


@app.route("/api/collections/<int:collection_id>/reading_plan", methods=["POST"])
def api_reading_plan(collection_id):
    return jsonify(research_logic.generate_reading_plan(collection_id))


@app.route("/api/collections/<int:collection_id>/next_paper")
def api_next_paper(collection_id):
    return jsonify(research_logic.get_next_paper(_user_email(), collection_id))


@app.route("/api/progress", methods=["POST"])
def api_progress():
    body = request.get_json(force=True)
    return jsonify(research_logic.update_reading_progress(_user_email(), body["paper_id"], body["status"]))


@app.route("/api/discover")
def api_discover():
    topic = request.args.get("topic", "")
    limit = int(request.args.get("limit", 10))
    if not topic:
        return jsonify({"status": "error", "message": "topic query param is required"}), 400
    return jsonify(research_logic.find_papers_for_goal(topic, limit))


@app.route("/api/search")
def api_search():
    query = request.args.get("query", "")
    limit = int(request.args.get("limit", 8))
    collection_id = request.args.get("collection_id", type=int)
    if not query:
        return jsonify({"status": "error", "message": "query param is required"}), 400
    return jsonify(research_logic.search_papers_semantic(query, limit, collection_id))


@app.route("/api/papers/<paper_id>")
def api_paper_details(paper_id):
    return jsonify(research_logic.get_paper_details([paper_id]))


@app.route("/api/notes", methods=["GET", "POST"])
def api_notes():
    user_email = _user_email()
    if request.method == "POST":
        body = request.get_json(force=True)
        result = research_logic.add_note(user_email, body["body"], body.get("paper_id"))
        return jsonify(result)
    return jsonify(research_logic.list_notes(user_email, request.args.get("paper_id")))


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8001))
    app.run(debug=True, host=host, port=port)
