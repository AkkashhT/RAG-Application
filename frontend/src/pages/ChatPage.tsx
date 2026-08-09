import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Send, Plus, ChevronDown, ChevronRight, Database, FileText,
  AlertTriangle, CheckCircle, Clock, Code, RefreshCw
} from 'lucide-react'
import { chatApi, type ChatSession, type ChatMessage, type Citation } from '../lib/api'

// ── Citation display ───────────────────────────────────────────────────────

function CitationBadge({ citation, index }: { citation: Citation; index: number }) {
  const [expanded, setExpanded] = useState(false)

  const pageLabel = citation.page_end && citation.page_end !== citation.page_start
    ? `pp. ${citation.page_start}–${citation.page_end}`
    : citation.page_start ? `p. ${citation.page_start}` : null

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden text-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-slate-50 hover:bg-slate-100 text-left"
      >
        {citation.source_type === 'document'
          ? <FileText size={14} className="text-blue-500 shrink-0" />
          : <Database size={14} className="text-green-500 shrink-0" />
        }
        <span className="font-medium text-slate-700">
          {citation.source_type === 'document'
            ? `[${index + 1}] ${citation.filename ?? 'Document'}`
            : `[${index + 1}] SQL: ${citation.table_name ?? 'Query'}`
          }
        </span>
        {pageLabel && (
          <span className="text-slate-500 text-xs">{pageLabel}</span>
        )}
        {citation.is_ocr && (
          <span className="ml-auto text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full shrink-0">
            OCR
          </span>
        )}
        {citation.rerank_score !== undefined && (
          <span className="ml-auto text-xs text-slate-400 shrink-0">
            score: {citation.rerank_score.toFixed(3)}
          </span>
        )}
        {expanded ? <ChevronDown size={14} className="ml-auto shrink-0" /> : <ChevronRight size={14} className="ml-auto shrink-0" />}
      </button>
      {expanded && (
        <div className="px-3 py-2 bg-white text-slate-600 text-xs leading-relaxed">
          {citation.section_heading && (
            <p className="font-semibold text-slate-700 mb-1">§ {citation.section_heading}</p>
          )}
          {citation.is_ocr && (
            <p className="text-amber-600 mb-1 flex items-center gap-1">
              <AlertTriangle size={12} /> OCR-extracted — may contain errors
            </p>
          )}
          {citation.text_preview && (
            <p className="italic text-slate-500">"{citation.text_preview}…"</p>
          )}
          {citation.sql_query && (
            <pre className="mt-1 bg-slate-100 p-2 rounded text-xs overflow-x-auto">{citation.sql_query}</pre>
          )}
          {citation.row_count !== undefined && (
            <p className="mt-1 text-slate-500">{citation.row_count} row(s) returned</p>
          )}
        </div>
      )}
    </div>
  )
}

// ── SQL display (always shown when SQL path taken) ─────────────────────────

function SQLDisplay({ sql }: { sql: string }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="mt-2 border border-green-200 rounded-lg overflow-hidden text-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-green-50 hover:bg-green-100 text-left"
      >
        <Code size={14} className="text-green-600 shrink-0" />
        <span className="font-medium text-green-800">Generated SQL query</span>
        <span className="text-xs text-green-600 ml-1">(verify this is correct)</span>
        {expanded ? <ChevronDown size={14} className="ml-auto shrink-0" /> : <ChevronRight size={14} className="ml-auto shrink-0" />}
      </button>
      {expanded && (
        <pre className="px-3 py-2 bg-white text-xs text-slate-700 overflow-x-auto border-t border-green-100">
          {sql}
        </pre>
      )}
    </div>
  )
}

// ── Message bubble ─────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: ChatMessage & { streaming?: boolean } }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-3xl w-full ${isUser ? 'ml-12' : 'mr-4'}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm ml-auto max-w-xl'
              : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm'
          }`}
        >
          {message.content}
          {message.streaming && (
            <span className="inline-block w-1.5 h-4 bg-slate-400 rounded-full ml-1 animate-pulse" />
          )}
        </div>

        {/* SQL display — always shown when SQL path was used */}
        {!isUser && message.sql_query && (
          <div className="mt-2">
            <SQLDisplay sql={message.sql_query} />
          </div>
        )}

        {/* Citations */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 space-y-1">
            {message.citations.map((c, i) => (
              <CitationBadge key={i} citation={c} index={i} />
            ))}
          </div>
        )}

        {/* Debug score */}
        {!isUser && message.top_rerank_score !== undefined && message.top_rerank_score > 0 && (
          <p className="mt-1 text-xs text-slate-400">
            top rerank score: {message.top_rerank_score.toFixed(3)} · {message.router_decision}
          </p>
        )}
      </div>
    </div>
  )
}

// ── Status indicator ───────────────────────────────────────────────────────

function StatusBar({ status }: { status: string | null }) {
  if (!status) return null
  const isWaiting = status.toLowerCase().includes('waiting')
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500 px-4 py-1">
      {isWaiting
        ? <Clock size={12} className="text-amber-500 animate-pulse" />
        : <RefreshCw size={12} className="animate-spin" />
      }
      {status}
    </div>
  )
}

// ── Main Chat page ─────────────────────────────────────────────────────────

export default function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null)
  const [messages, setMessages] = useState<(ChatMessage & { streaming?: boolean })[]>([])
  const [input, setInput] = useState('')
  const [status, setStatus] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadSessions()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  async function loadSessions() {
    const data = await chatApi.listSessions()
    setSessions(data)
    if (data.length > 0 && !activeSession) {
      await openSession(data[0])
    }
  }

  async function openSession(session: ChatSession) {
    const full = await chatApi.getSession(session.id)
    setActiveSession(full)
    setMessages(full.messages ?? [])
  }

  async function newSession() {
    const session = await chatApi.createSession()
    setSessions(prev => [session, ...prev])
    setActiveSession(session)
    setMessages([])
  }

  async function sendMessage() {
    if (!input.trim() || isStreaming || !activeSession) return

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input,
      created_at: new Date().toISOString(),
    }

    const assistantId = crypto.randomUUID()
    const assistantMsg: ChatMessage & { streaming: boolean } = {
      id: assistantId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
      streaming: true,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
    setIsStreaming(true)
    setStatus('Sending...')

    const ctrl = new AbortController()
    abortRef.current = ctrl

    let accumulatedContent = ''
    let finalCitations: Citation[] = []
    let sqlQuery: string | undefined
    let routerDecision: string | undefined
    let topScore: number | undefined

    try {
      for await (const event of chatApi.sendMessageStream(
        activeSession.id, userMsg.content, undefined, ctrl.signal
      )) {
        if (event.type === 'status') {
          setStatus(event.content as string)
        } else if (event.type === 'token') {
          accumulatedContent += event.content as string
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantId ? { ...m, content: accumulatedContent } : m
            )
          )
        } else if (event.type === 'citations') {
          finalCitations = event.content as Citation[]
        } else if (event.type === 'sql') {
          sqlQuery = event.content as string
        } else if (event.type === 'done') {
          const done = event.content as { router_decision?: string; top_rerank_score?: number }
          routerDecision = done.router_decision
          topScore = done.top_rerank_score
        } else if (event.type === 'error') {
          accumulatedContent += `\n\n⚠️ ${event.content}`
          setMessages(prev =>
            prev.map(m => m.id === assistantId ? { ...m, content: accumulatedContent } : m)
          )
        }
      }
    } catch (err: unknown) {
      if ((err as { name?: string }).name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, content: m.content || 'Request failed. Is the backend running?' }
              : m
          )
        )
      }
    } finally {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? {
                ...m,
                streaming: false,
                citations: finalCitations,
                sql_query: sqlQuery,
                router_decision: routerDecision,
                top_rerank_score: topScore,
              }
            : m
        )
      )
      setIsStreaming(false)
      setStatus(null)

      // Update session title from first user message
      if (messages.length === 0 && activeSession) {
        const newTitle = userMsg.content.slice(0, 40) + (userMsg.content.length > 40 ? '…' : '')
        await chatApi.updateSession(activeSession.id, newTitle)
        setSessions(prev =>
          prev.map(s => s.id === activeSession.id ? { ...s, title: newTitle } : s)
        )
      }
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex h-screen bg-slate-50">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-slate-200 flex flex-col">
        <div className="p-4 border-b border-slate-200">
          <button
            onClick={newSession}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          >
            <Plus size={16} /> New chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map(session => (
            <button
              key={session.id}
              onClick={() => openSession(session)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors ${
                activeSession?.id === session.id
                  ? 'bg-blue-50 text-blue-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              {session.title}
            </button>
          ))}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
          <div>
            <h1 className="font-semibold text-slate-800">
              {activeSession?.title ?? 'LocalRAG'}
            </h1>
            <p className="text-xs text-slate-500">Fully local · no data leaves your machine</p>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-3">
              <CheckCircle size={40} className="text-slate-300" />
              <p className="text-lg font-medium">Ask anything about your documents</p>
              <p className="text-sm">Upload documents or connect a database in the sidebar</p>
            </div>
          )}
          {messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          <StatusBar status={status} />
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-slate-200 px-6 py-4">
          <div className="flex gap-3 items-end max-w-4xl mx-auto">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents or database..."
              rows={1}
              disabled={isStreaming || !activeSession}
              className="flex-1 resize-none rounded-xl border border-slate-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 bg-slate-50"
              style={{ minHeight: 44, maxHeight: 200 }}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isStreaming || !activeSession}
              className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              <Send size={16} />
            </button>
          </div>
          <p className="text-xs text-slate-400 text-center mt-2">
            Press Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  )
}
