// Central API client — all backend calls go through here.
// Base URL is /api (proxied by Vite in dev, served directly in prod).

const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Documents ──────────────────────────────────────────────────────────────

export interface Document {
  id: string
  filename: string
  file_type: string
  file_size_bytes: number
  upload_timestamp: string
  chunk_count: number
  status: 'pending' | 'ingesting' | 'ready' | 'error'
  error_message?: string
  has_ocr_pages: boolean
  ocr_pages?: number[]
}

export const documentsApi = {
  list: () => request<Document[]>('/documents/'),
  get: (id: string) => request<Document>(`/documents/${id}`),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<{ id: string; status: string }>('/documents/upload', {
      method: 'POST',
      body: form,
      headers: {}, // let browser set multipart boundary
    })
  },
  delete: (id: string) => request<{ message: string }>(`/documents/${id}`, { method: 'DELETE' }),
  reindex: (id: string) =>
    request<{ message: string }>(`/documents/${id}/reindex`, { method: 'POST' }),
}

// ── Chat ───────────────────────────────────────────────────────────────────

export interface Citation {
  source_type: 'document' | 'sql'
  document_id?: string
  filename?: string
  page_start?: number
  page_end?: number
  section_heading?: string
  is_ocr?: boolean
  rerank_score?: number
  text_preview?: string
  sql_query?: string
  row_count?: number
  table_name?: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  citations?: Citation[]
  sql_query?: string
  router_decision?: string
  top_rerank_score?: number
}

export interface ChatSession {
  id: string
  title: string
  created_at: string
  updated_at: string
  scoped_document_ids?: string[]
  messages?: ChatMessage[]
}

export const chatApi = {
  createSession: (title?: string, scoped_document_ids?: string[]) =>
    request<ChatSession>('/chat/sessions', {
      method: 'POST',
      body: JSON.stringify({ title: title ?? 'New conversation', scoped_document_ids }),
    }),
  listSessions: () => request<ChatSession[]>('/chat/sessions'),
  getSession: (id: string) => request<ChatSession>(`/chat/sessions/${id}`),
  deleteSession: (id: string) =>
    request<{ message: string }>(`/chat/sessions/${id}`, { method: 'DELETE' }),
  updateSession: (id: string, title: string) =>
    request<ChatSession>(`/chat/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    }),

  streamMessage: (
    sessionId: string,
    content: string,
    scopedDocIds?: string[],
  ): EventSource => {
    // We POST first, but SSE needs a GET or EventSource — use fetch streaming instead
    // Return a URL that we'll fetch via fetch() with streaming in the hook
    return null as unknown as EventSource // unused; see useChatStream hook
  },

  sendMessageStream: async function* (
    sessionId: string,
    content: string,
    scopedDocIds?: string[],
    signal?: AbortSignal,
  ): AsyncGenerator<{ type: string; content: unknown }> {
    const res = await fetch(`${BASE}/chat/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, scoped_document_ids: scopedDocIds }),
      signal,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || `HTTP ${res.status}`)
    }
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '[DONE]') return
          try {
            yield JSON.parse(data)
          } catch {
            // skip malformed lines
          }
        }
      }
    }
  },
}

// ── Database ───────────────────────────────────────────────────────────────

export const dbApi = {
  connect: (connection_string: string) =>
    request<{ ok: boolean; dialect: string; url_display: string }>('/database/connect', {
      method: 'POST',
      body: JSON.stringify({ connection_string }),
    }),
  test: (connection_string: string) =>
    request<{ ok: boolean; dialect: string }>('/database/test', {
      method: 'POST',
      body: JSON.stringify({ connection_string }),
    }),
  getSchema: () => request<{ dialect: string; tables: Record<string, unknown> }>('/database/schema'),
  getStatus: () => request<{ connected: boolean; dialect?: string; url_display?: string }>('/database/status'),
  disconnect: () => request<{ message: string }>('/database/disconnect', { method: 'DELETE' }),
}

// ── Settings ───────────────────────────────────────────────────────────────

export interface AppSettings {
  ollama_base_url: string
  ollama_generation_model: string
  ollama_embed_model: string
  reranker_model: string
  reranker_device: string
  reranker_top_k: number
  reranker_initial_top_k: number
  chunk_size_tokens: number
  chunk_overlap_tokens: number
  hybrid_dense_weight: number
  confidence_threshold: number
  llm_temperature: number
  llm_max_tokens: number
  max_concurrent_llm_calls: number
  router_mode: string
}

export interface HealthStatus {
  ollama: { status: string; models_available?: string[]; generation_model_ready?: boolean; embed_model_ready?: boolean; gpu_active?: boolean; warning?: string; error?: string }
  qdrant: { status: string; warning?: string }
  reranker: { loaded: boolean; on_gpu: boolean; device: string; warning?: string }
}

export const settingsApi = {
  get: () => request<AppSettings>('/settings/'),
  update: (data: Partial<AppSettings>) =>
    request<{ message: string }>('/settings/', { method: 'PATCH', body: JSON.stringify(data) }),
  listOllamaModels: () => request<{ models: { name: string; size_gb: number }[] }>('/settings/ollama/models'),
  health: () => request<HealthStatus>('/settings/health'),
}
