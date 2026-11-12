// In-browser demo backend: a TypeScript port of backend/documind/rag.py (same chunking,
// TF-IDF vectors, cosine search and refusal rule), so the GitHub Pages demo needs no server.
import type { Answer, Doc } from './api'

const STOP = new Set('a an the and or of to in on for with is are was were be by as at it this that from'.split(' '))
const tokenize = (t: string) => (t.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => !STOP.has(w))

export function chunkText(text: string, size = 400, overlap = 60): string[] {
  const sentences = text.split(/(?<=[.!?])\s+|\n{2,}/).map((s) => s.trim()).filter(Boolean)
  const chunks: string[] = []
  let cur = ''
  for (const s of sentences) {
    if (cur && cur.length + s.length + 1 > size) {
      chunks.push(cur)
      const tail = cur.slice(-overlap)
      const i = tail.indexOf(' ')
      cur = `${i >= 0 ? tail.slice(i + 1) : tail} ${s}`
    } else cur = `${cur} ${s}`.trim()
  }
  if (cur) chunks.push(cur)
  return chunks
}

type Chunk = { docId: number; doc: string; text: string }
let docs: { id: number; name: string }[] = []
let chunks: Chunk[] = []
let nextId = 1

function vectors(texts: string[], queryText: string) {
  const df = new Map<string, number>()
  for (const t of texts) for (const w of new Set(tokenize(t))) df.set(w, (df.get(w) ?? 0) + 1)
  const n = texts.length
  const idf = (w: string) => Math.log((1 + n) / (1 + (df.get(w) ?? 0))) + 1
  const embed = (t: string) => {
    const tf = new Map<string, number>()
    for (const w of tokenize(t)) if (df.has(w)) tf.set(w, (tf.get(w) ?? 0) + 1)
    const v = new Map<string, number>()
    let norm = 0
    for (const [w, c] of tf) {
      const x = (1 + Math.log(c)) * idf(w)
      v.set(w, x)
      norm += x * x
    }
    norm = Math.sqrt(norm)
    if (norm) for (const [w, x] of v) v.set(w, x / norm)
    return v
  }
  return { docVecs: texts.map(embed), q: embed(queryText) }
}

function search(question: string, k = 3, minScore = 0.05) {
  if (!chunks.length) return []
  const { docVecs, q } = vectors(chunks.map((c) => c.text), question)
  return docVecs
    .map((v, i) => {
      let s = 0
      for (const [w, x] of q) s += x * (v.get(w) ?? 0)
      return { i, s }
    })
    .sort((a, b) => b.s - a.s)
    .slice(0, k)
    .filter((r) => r.s >= minScore)
    .map((r) => ({ ...chunks[r.i], score: Math.round(r.s * 1e4) / 1e4 }))
}

const IDK = "I don't know: no relevant passage found in the uploaded documents."

export async function addText(name: string, text: string): Promise<Doc> {
  const cs = chunkText(text)
  if (!name.trim()) throw new Error('name is required')
  if (!cs.length) throw new Error('document has no text')
  const id = nextId++
  docs.push({ id, name })
  cs.forEach((c) => chunks.push({ docId: id, doc: name, text: c }))
  return { id, name, chunks: cs.length }
}

export async function uploadDoc(file: File): Promise<Doc> {
  if (!/\.(txt|md)$/i.test(file.name)) throw new Error('only .txt and .md files are supported')
  if (file.size > 1_000_000) throw new Error('file too large (max 1 MB)')
  return addText(file.name, await file.text())
}

export const listDocs = async (): Promise<Doc[]> =>
  docs.map((d) => ({ ...d, chunks: chunks.filter((c) => c.docId === d.id).length }))

export async function deleteDoc(id: number): Promise<void> {
  docs = docs.filter((d) => d.id !== id)
  chunks = chunks.filter((c) => c.docId !== id)
}

export async function ask(question: string): Promise<Answer> {
  const hits = search(question)
  if (!hits.length) return { answer: IDK, mode: 'no-context', sources: [] }
  const q = new Set(tokenize(question))
  let best = hits[0].text
  let bestO = -1
  for (const s of hits[0].text.split(/(?<=[.!?])\s+/)) {
    const o = tokenize(s).filter((w) => q.has(w)).length
    if (o > bestO) [best, bestO] = [s, o]
  }
  return {
    answer: `${best} [1]`,
    mode: 'extractive',
    sources: hits.map((h, i) => ({ n: i + 1, doc: h.doc, score: h.score, text: h.text })),
  }
}

export const SAMPLES: Record<string, string> = {
  'kafka.md':
    'Apache Kafka is a distributed event streaming platform. Topics are split into partitions, and each partition is an ordered log. Producers append records to a partition. Consumers in a consumer group share the partitions of a topic, so each record is processed by one consumer in the group. Replication copies each partition to several brokers for fault tolerance.',
  'fhir.md':
    'FHIR is a standard for exchanging healthcare data over REST APIs. Data is modelled as resources such as Patient, Observation and Encounter. Each resource is exchanged as JSON or XML. A Bundle groups several resources into one request. Search parameters let clients filter resources, for example finding all Observations for one Patient.',
  'jwt.md':
    'A JSON Web Token has three parts: header, payload and signature. The server signs the token with a secret or private key. Clients send the token in the Authorization header. A token must be rejected if it is expired or if its signature does not match. Never accept tokens that use the none algorithm.',
}

export async function loadSamples() {
  for (const [n, t] of Object.entries(SAMPLES)) await addText(n, t)
}
