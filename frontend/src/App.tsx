import { useEffect, useRef, useState } from 'react'
import { DEMO, ask, deleteDoc, listDocs, loadSamples, uploadDoc } from './api'
import type { Answer, Doc } from './api'

type Turn = { q: string; a?: Answer; error?: string }

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([])
  const [turns, setTurns] = useState<Turn[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = () => listDocs().then(setDocs).catch((e) => setNotice(e.message))
  useEffect(() => {
    if (DEMO) loadSamples().then(refresh)
    else refresh()
  }, [])

  async function onUpload(files: FileList | null) {
    if (!files?.length) return
    setNotice('')
    for (const f of Array.from(files)) {
      try {
        await uploadDoc(f)
      } catch (e) {
        setNotice(`${f.name}: ${(e as Error).message}`)
      }
    }
    if (fileRef.current) fileRef.current.value = ''
    refresh()
  }

  async function onAsk(e: React.FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || busy) return
    setQuestion('')
    setBusy(true)
    setTurns((t) => [...t, { q }])
    try {
      const a = await ask(q)
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, a } : x)))
    } catch (err) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: (err as Error).message } : x)))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="layout">
      <aside>
        <h1>DocuMind</h1>
        <p className="muted">Upload .txt / .md files, then ask questions. Answers cite their sources.</p>
        {DEMO && (
          <p className="demo">
            Demo: runs entirely in your browser (nothing is uploaded). Try “How are Kafka topics split?” or add your
            own file. The full version has a Flask API, SQLite and optional Claude answers:{' '}
            <a href="https://github.com/viralgautami/documind">source</a>.
          </p>
        )}
        <label className="upload">
          <input ref={fileRef} type="file" accept=".txt,.md" multiple onChange={(e) => onUpload(e.target.files)} />
          Upload documents
        </label>
        {notice && <p className="error" role="alert">{notice}</p>}
        <ul className="docs" aria-label="Documents">
          {docs.map((d) => (
            <li key={d.id}>
              <span>{d.name} <small>({d.chunks} chunks)</small></span>
              <button aria-label={`Delete ${d.name}`} onClick={() => deleteDoc(d.id).then(refresh)}>×</button>
            </li>
          ))}
          {!docs.length && <li className="muted">No documents yet</li>}
        </ul>
      </aside>
      <main>
        <div className="chat">
          {!turns.length && <p className="muted">Ask something about your documents.</p>}
          {turns.map((t, i) => (
            <div key={i} className="turn">
              <div className="q">{t.q}</div>
              {t.error && <div className="error">{t.error}</div>}
              {t.a && (
                <div className="a">
                  <p>{t.a.answer}</p>
                  {t.a.sources.length > 0 && (
                    <details>
                      <summary>{t.a.sources.length} source(s)</summary>
                      {t.a.sources.map((s) => (
                        <blockquote key={s.n}>
                          <b>[{s.n}] {s.doc}</b> <small>match {(s.score * 100).toFixed(0)}%</small>
                          <br />{s.text}
                        </blockquote>
                      ))}
                    </details>
                  )}
                </div>
              )}
              {!t.a && !t.error && <div className="a muted">Thinking…</div>}
            </div>
          ))}
        </div>
        <form onSubmit={onAsk}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question…"
            aria-label="Question"
          />
          <button disabled={busy || !question.trim()}>Ask</button>
        </form>
      </main>
    </div>
  )
}
