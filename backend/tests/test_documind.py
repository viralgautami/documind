import io
import pytest

from documind.app import create_app
from documind.rag import Index, chunk_text, tokenize, build_prompt

DOC = (
    "The Eiffel Tower is in Paris and was completed in 1889. It is 330 metres tall. "
    "Mount Fuji is the highest mountain in Japan. It last erupted in 1707. "
    "Python is a programming language created by Guido van Rossum. Flask is a Python web framework."
)


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "t.db"), llm=None)
    app.config["LLM"] = None
    return app.test_client()


def add(client, name="facts.txt", text=DOC):
    return client.post("/api/documents", json={"name": name, "text": text})


# ---- core RAG units
def test_chunking_respects_size_and_covers_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=30)
    assert len(chunks) > 3
    assert all(len(c) <= 260 for c in chunks)
    assert "Sentence number 59" in chunks[-1]


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_tokenize_drops_stopwords():
    assert tokenize("The Tower of Paris") == ["tower", "paris"]


def test_search_ranks_relevant_chunk_first():
    idx = Index([
        {"doc_id": 1, "doc_name": "a", "text": "Cats are small domestic animals that purr."},
        {"doc_id": 1, "doc_name": "a", "text": "Kafka is a distributed event streaming platform."},
        {"doc_id": 1, "doc_name": "a", "text": "Bread is baked from flour and water."},
    ])
    hit = idx.search("what is kafka streaming?", k=1)[0]
    assert "Kafka" in hit["text"]


def test_search_empty_index_and_no_match():
    assert Index([]).search("anything") == []
    idx = Index([{"doc_id": 1, "doc_name": "a", "text": "Bread is baked from flour."}])
    assert idx.search("quantum chromodynamics") == []


def test_prompt_contains_numbered_sources():
    p = build_prompt("q?", [{"doc_name": "d", "text": "alpha"}, {"doc_name": "e", "text": "beta"}])
    assert "[1] (d) alpha" in p and "[2] (e) beta" in p and "ONLY" in p


# ---- API
def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok", "generator": "extractive"}


def test_upload_list_delete(client):
    r = add(client)
    assert r.status_code == 201 and r.get_json()["chunks"] >= 1
    docs = client.get("/api/documents").get_json()
    assert [d["name"] for d in docs] == ["facts.txt"]
    assert client.delete(f"/api/documents/{docs[0]['id']}").status_code == 204
    assert client.get("/api/documents").get_json() == []
    assert client.delete("/api/documents/999").status_code == 404


def test_file_upload_and_validation(client):
    ok = client.post("/api/documents", data={"file": (io.BytesIO(DOC.encode()), "n.md")},
                     content_type="multipart/form-data")
    assert ok.status_code == 201
    bad = client.post("/api/documents", data={"file": (io.BytesIO(b"x"), "n.exe")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    notutf = client.post("/api/documents", data={"file": (io.BytesIO(b"\xff\xfe\x00"), "n.txt")},
                         content_type="multipart/form-data")
    assert notutf.status_code == 400
    assert add(client, name="", text="x").status_code == 400
    assert add(client, text="   ").status_code == 400


def test_oversized_upload_rejected(client):
    big = io.BytesIO(b"a. " * 400_000)
    r = client.post("/api/documents", data={"file": (big, "big.txt")}, content_type="multipart/form-data")
    assert r.status_code == 413


def test_ask_returns_cited_correct_answer(client):
    add(client)
    r = client.post("/api/ask", json={"question": "When was the Eiffel Tower completed?"}).get_json()
    assert "1889" in r["answer"] and "[1]" in r["answer"]
    assert r["mode"] == "extractive"
    assert r["sources"][0]["doc"] == "facts.txt" and "Eiffel" in r["sources"][0]["text"]


def test_ask_picks_right_passage_among_topics(client):
    add(client)
    r = client.post("/api/ask", json={"question": "Who created Python?"}).get_json()
    assert "Guido" in r["answer"]


def test_ask_refuses_when_nothing_relevant(client):
    add(client)
    r = client.post("/api/ask", json={"question": "quantum chromodynamics gluon"}).get_json()
    assert r["mode"] == "no-context" and "don't know" in r["answer"] and r["sources"] == []


def test_ask_without_documents_and_without_question(client):
    assert client.post("/api/ask", json={"question": "hello?"}).get_json()["mode"] == "no-context"
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", data="notjson").status_code == 400


def test_deleted_document_no_longer_retrieved(client):
    doc_id = add(client).get_json()["id"]
    client.delete(f"/api/documents/{doc_id}")
    r = client.post("/api/ask", json={"question": "Eiffel Tower"}).get_json()
    assert r["sources"] == []


def test_llm_generator_gets_grounded_prompt(tmp_path):
    seen = {}

    def fake(prompt):
        seen["p"] = prompt
        return "Completed in 1889 [1]."

    c = create_app(str(tmp_path / "l.db"), llm=fake).test_client()
    add(c)
    r = c.post("/api/ask", json={"question": "When was the Eiffel Tower completed?"}).get_json()
    assert r["mode"] == "llm" and r["answer"].startswith("Completed")
    assert "Eiffel" in seen["p"] and "ONLY" in seen["p"]
    # llm must not be called when nothing is retrieved (no hallucination path)
    seen.clear()
    r2 = c.post("/api/ask", json={"question": "quantum gluon"}).get_json()
    assert r2["mode"] == "no-context" and "p" not in seen
