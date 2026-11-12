"""Retrieval evaluation: does the right passage come back for each question?

Corpus and questions are small and hand-written (see data below), so treat the
numbers as a regression benchmark, not a claim about general-purpose accuracy.
Run:  python -m eval.run_eval   (from the backend folder)
"""
import sys

from documind.rag import Index, chunk_text

DOCS = {
    "kafka.md": (
        "Apache Kafka is a distributed event streaming platform. Topics are split into partitions, "
        "and each partition is an ordered log. Producers append records to a partition. "
        "Consumers in a consumer group share the partitions of a topic, so each record is processed "
        "by one consumer in the group. Replication copies each partition to several brokers for fault tolerance."
    ),
    "fhir.md": (
        "FHIR is a standard for exchanging healthcare data over REST APIs. Data is modelled as resources "
        "such as Patient, Observation and Encounter. Each resource is exchanged as JSON or XML. "
        "A Bundle groups several resources into one request. Search parameters let clients filter resources, "
        "for example finding all Observations for one Patient."
    ),
    "jwt.md": (
        "A JSON Web Token has three parts: header, payload and signature. The server signs the token with a secret "
        "or private key. Clients send the token in the Authorization header. A token must be rejected if it is expired "
        "or if its signature does not match. Never accept tokens that use the none algorithm."
    ),
    "flask.md": (
        "Flask is a lightweight Python web framework. A route maps a URL to a view function. "
        "Blueprints organise a large application into modules. The application factory pattern creates the app "
        "inside a function, which makes testing with different configurations easy. Flask ships with a development server "
        "that should not be used in production."
    ),
    "react.md": (
        "React builds user interfaces from components. State is data that changes over time and is stored with the useState hook. "
        "The useEffect hook runs side effects such as fetching data after render. Keys help React identify which list items changed. "
        "Lifting state up means moving shared state to the closest common parent."
    ),
    "rag.md": (
        "Retrieval-augmented generation answers questions using passages retrieved from a document store. "
        "Documents are split into chunks and each chunk is converted to an embedding vector. "
        "At query time the question is embedded and the nearest chunks are retrieved by cosine similarity. "
        "The retrieved chunks are placed in the prompt so the model can ground its answer and cite sources. "
        "If nothing relevant is retrieved the system should say it does not know."
    ),
}

# (question, expected document)
QA = [
    ("How are Kafka topics divided?", "kafka.md"),
    ("What do consumers in a group share?", "kafka.md"),
    ("How does Kafka tolerate broker failure?", "kafka.md"),
    ("Which healthcare resources does FHIR define?", "fhir.md"),
    ("How do I group several FHIR resources into one request?", "fhir.md"),
    ("How can clients filter observations for a patient?", "fhir.md"),
    ("What are the three parts of a JWT?", "jwt.md"),
    ("When must a token be rejected?", "jwt.md"),
    ("Where does the client send the token?", "jwt.md"),
    ("What does a Flask route do?", "flask.md"),
    ("Why use the application factory pattern?", "flask.md"),
    ("Should the Flask development server be used in production?", "flask.md"),
    ("Which hook stores state in React?", "react.md"),
    ("What is useEffect for?", "react.md"),
    ("Why do list items need keys?", "react.md"),
    ("What does lifting state up mean?", "react.md"),
    ("How are chunks converted for retrieval?", "rag.md"),
    ("How does retrieval find the nearest chunks?", "rag.md"),
    ("What should the system do if nothing relevant is found?", "rag.md"),
    ("How does the prompt let the model cite sources?", "rag.md"),
]

# Harder set: paraphrased, little word overlap with the source text.
PARAPHRASED = [
    ("What keeps a message stream available if a server machine dies?", "kafka.md"),
    ("How do several workers split up the incoming messages?", "kafka.md"),
    ("Which standard lets hospitals share patient records through web services?", "fhir.md"),
    ("How do I fetch many medical records in a single call?", "fhir.md"),
    ("Why would a login credential be refused by the server?", "jwt.md"),
    ("How is a login credential protected against tampering?", "jwt.md"),
    ("How do I structure a big Python web app into pieces?", "flask.md"),
    ("How do I make a screen update when data changes over time?", "react.md"),
    ("How does an AI assistant avoid making things up when answering from files?", "rag.md"),
    ("How are text passages turned into numbers for similarity search?", "rag.md"),
]

OUT_OF_SCOPE = [
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "Explain quantum chromodynamics gluon confinement",
    "Who won the football world cup?",
]


def build() -> Index:
    chunks = []
    for name, text in DOCS.items():
        for c in chunk_text(text, size=220, overlap=40):
            chunks.append({"doc_id": 0, "doc_name": name, "text": c})
    return Index(chunks)


def score(idx: Index, qa, label: str) -> None:
    hit1 = hit3 = 0
    rr = 0.0
    misses = []
    for q, want in qa:
        docs = [h["doc_name"] for h in idx.search(q, k=3)]
        if docs[:1] == [want]:
            hit1 += 1
        if want in docs:
            hit3 += 1
            rr += 1 / (docs.index(want) + 1)
        else:
            misses.append((q, want, docs))
    n = len(qa)
    print(f"[{label}] n={n}  hit@1={hit1}/{n} ({hit1 / n:.0%})  hit@3={hit3}/{n} ({hit3 / n:.0%})  MRR={rr / n:.3f}")
    for m in misses:
        print("   MISS:", m)


def main() -> int:
    idx = build()
    print(f"chunks indexed: {len(idx.chunks)}")
    score(idx, QA, "keyword-style")
    score(idx, PARAPHRASED, "paraphrased  ")
    refused = sum(1 for q in OUT_OF_SCOPE if not idx.search(q, k=3))
    print(f"out-of-scope refused: {refused}/{len(OUT_OF_SCOPE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
