"""Flask API for DocuMind: upload documents, ask questions, get cited answers."""
import os
import sqlite3

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from .rag import Index, chunk_text, generate_answer

MAX_BYTES = 1_000_000
ALLOWED = {".txt", ".md"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  pos INTEGER NOT NULL, text TEXT NOT NULL);
"""


def anthropic_llm():
    """Optional generator: used only if ANTHROPIC_API_KEY is set and the SDK is installed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic()
    model = os.environ.get("DOCUMIND_MODEL", "claude-haiku-5-5")

    def call(prompt: str) -> str:
        msg = client.messages.create(model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    return call


def create_app(db_path: str = "documind.db", llm=None) -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config["DB_PATH"] = db_path
    app.config["LLM"] = llm if llm is not None else anthropic_llm()
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

    def db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DB_PATH"])
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys=ON")
        return g.db

    @app.teardown_appcontext
    def close(_):
        d = g.pop("db", None)
        if d:
            d.close()

    with sqlite3.connect(db_path) as c:
        c.executescript(SCHEMA)

    def load_index() -> Index:
        rows = db().execute(
            "SELECT c.id, c.doc_id, d.name AS doc_name, c.text FROM chunks c "
            "JOIN documents d ON d.id=c.doc_id ORDER BY c.id"
        ).fetchall()
        return Index([dict(r) for r in rows])

    def err(msg, code=400):
        return jsonify(error=msg), code

    @app.errorhandler(413)
    def too_big(_):
        return err("file too large (max 1 MB)", 413)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", generator="llm" if app.config["LLM"] else "extractive")

    @app.get("/api/documents")
    def list_docs():
        rows = db().execute(
            "SELECT d.id, d.name, d.created, COUNT(c.id) AS chunks FROM documents d "
            "LEFT JOIN chunks c ON c.doc_id=d.id GROUP BY d.id ORDER BY d.id"
        ).fetchall()
        return jsonify([dict(r) for r in rows])

    @app.post("/api/documents")
    def add_doc():
        if "file" in request.files:
            f = request.files["file"]
            name = f.filename or ""
            if os.path.splitext(name)[1].lower() not in ALLOWED:
                return err("only .txt and .md files are supported")
            try:
                text = f.read().decode("utf-8")
            except UnicodeDecodeError:
                return err("file must be UTF-8 text")
        else:
            body = request.get_json(silent=True) or {}
            name, text = (body.get("name") or "").strip(), body.get("text") or ""
            if not name:
                return err("name is required")
        chunks = chunk_text(text)
        if not chunks:
            return err("document has no text")
        cur = db().execute("INSERT INTO documents(name) VALUES (?)", (name,))
        db().executemany(
            "INSERT INTO chunks(doc_id,pos,text) VALUES (?,?,?)",
            [(cur.lastrowid, i, c) for i, c in enumerate(chunks)],
        )
        db().commit()
        return jsonify(id=cur.lastrowid, name=name, chunks=len(chunks)), 201

    @app.delete("/api/documents/<int:doc_id>")
    def del_doc(doc_id):
        cur = db().execute("DELETE FROM documents WHERE id=?", (doc_id,))
        db().commit()
        return ("", 204) if cur.rowcount else err("not found", 404)

    @app.post("/api/ask")
    def ask():
        body = request.get_json(silent=True) or {}
        q = (body.get("question") or "").strip()
        if not q:
            return err("question is required")
        hits = load_index().search(q, k=int(body.get("k", 3)))
        answer, mode = generate_answer(q, hits, app.config["LLM"])
        return jsonify(
            answer=answer,
            mode=mode,
            sources=[
                {"n": i + 1, "doc": h["doc_name"], "score": h["score"], "text": h["text"]}
                for i, h in enumerate(hits)
            ],
        )

    return app
