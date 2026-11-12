# DocuMind: ask questions of your documents (RAG)

Upload `.txt` / `.md` files, ask a question, get an answer that cites the passages it came from.

**Stack:** Python, Flask, SQLite, NumPy · React, TypeScript, Vite · pytest

## How it works

1. **Ingest:** a document is split into overlapping, sentence-aware chunks and stored in SQLite.
2. **Retrieve:** each chunk is embedded as an L2-normalised TF-IDF vector; the question is embedded the same way and the top-k chunks are found by cosine similarity. Chunks below a minimum score are dropped.
3. **Generate:** the retrieved chunks go into a prompt that tells the model to answer only from the numbered context and cite `[n]`.
   - With `ANTHROPIC_API_KEY` set (and `pip install anthropic`), the answer is written by an LLM.
   - Without a key, an offline extractive fallback returns the best-matching sentence, cited.
4. **No context, no answer:** if nothing relevant is retrieved, the API replies "I don't know" and the LLM is never called, so it cannot invent an answer.

Retrieval is lexical (TF-IDF), not neural. The `Index`/embedder boundary in `rag.py` is small, so a sentence-embedding model can replace it without changing the API.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status and active generator |
| GET / POST | `/api/documents` | list; upload (multipart `file`, or JSON `name` + `text`) |
| DELETE | `/api/documents/<id>` | remove a document and its chunks |
| POST | `/api/ask` | `{question}` returns `{answer, mode, sources[]}` |

Uploads are limited to 1 MB, `.txt` / `.md`, UTF-8.

## Run it

```bash
# backend
cd backend
pip install -r requirements.txt
python run.py                  # http://127.0.0.1:5000
python -m pytest -q            # 16 tests

# frontend (second terminal)
cd frontend
npm install
npm run dev                    # proxies /api to the Flask server
```
