export type Doc = { id: number; name: string; chunks: number }
export type Source = { n: number; doc: string; score: number; text: string }
export type Answer = { answer: string; mode: string; sources: Source[] }

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (r.status === 204) return undefined as T
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(body.error ?? `Request failed (${r.status})`)
  return body as T
}

export const listDocs = () => call<Doc[]>('/api/documents')
export const deleteDoc = (id: number) => call<void>(`/api/documents/${id}`, { method: 'DELETE' })
export const uploadDoc = (file: File) => {
  const f = new FormData()
  f.append('file', file)
  return call<Doc>('/api/documents', { method: 'POST', body: f })
}
export const ask = (question: string) =>
  call<Answer>('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
