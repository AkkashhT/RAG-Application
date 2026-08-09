import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Upload, Trash2, RefreshCw, FileText, FileSpreadsheet, AlertTriangle,
  CheckCircle, Clock, Loader2, X, ScanLine
} from 'lucide-react'
import { documentsApi, type Document } from '../lib/api'

const FILE_ICONS: Record<string, JSX.Element> = {
  pdf: <FileText size={20} className="text-red-500" />,
  docx: <FileText size={20} className="text-blue-500" />,
  txt: <FileText size={20} className="text-slate-400" />,
  md: <FileText size={20} className="text-purple-500" />,
  csv: <FileSpreadsheet size={20} className="text-green-500" />,
}

function StatusBadge({ status }: { status: Document['status'] }) {
  const map = {
    pending: { icon: <Clock size={12} />, label: 'Pending', cls: 'bg-slate-100 text-slate-600' },
    ingesting: { icon: <Loader2 size={12} className="animate-spin" />, label: 'Ingesting…', cls: 'bg-blue-50 text-blue-600' },
    ready: { icon: <CheckCircle size={12} />, label: 'Ready', cls: 'bg-green-50 text-green-700' },
    error: { icon: <AlertTriangle size={12} />, label: 'Error', cls: 'bg-red-50 text-red-600' },
  }
  const { icon, label, cls } = map[status] ?? map.pending
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>
      {icon}{label}
    </span>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    loadDocuments()
    // Poll for status updates while any doc is not ready/error
    pollingRef.current = setInterval(() => {
      setDocuments(prev => {
        const needsPoll = prev.some(d => d.status === 'pending' || d.status === 'ingesting')
        if (needsPoll) loadDocuments()
        return prev
      })
    }, 2500)
    return () => { if (pollingRef.current) clearInterval(pollingRef.current) }
  }, [])

  async function loadDocuments() {
    try {
      const docs = await documentsApi.list()
      setDocuments(docs)
    } catch {
      // silent — don't interrupt existing view
    }
  }

  async function handleFiles(files: FileList | File[]) {
    const allowed = ['pdf', 'docx', 'txt', 'csv', 'md']
    const fileArr = Array.from(files)
    const invalid = fileArr.filter(f => !allowed.includes(f.name.split('.').pop()?.toLowerCase() ?? ''))
    if (invalid.length > 0) {
      setError(`Unsupported file type(s): ${invalid.map(f => f.name).join(', ')}`)
      return
    }
    setError(null)
    setUploading(true)
    for (const file of fileArr) {
      try {
        await documentsApi.upload(file)
      } catch (e: unknown) {
        setError(`Upload failed: ${(e as Error).message}`)
      }
    }
    setUploading(false)
    loadDocuments()
  }

  async function deleteDoc(doc: Document) {
    if (!confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return
    try {
      await documentsApi.delete(doc.id)
      setDocuments(prev => prev.filter(d => d.id !== doc.id))
    } catch (e: unknown) {
      setError(`Delete failed: ${(e as Error).message}`)
    }
  }

  async function reindexDoc(doc: Document) {
    try {
      await documentsApi.reindex(doc.id)
      setDocuments(prev => prev.map(d => d.id === doc.id ? { ...d, status: 'pending' } : d))
    } catch (e: unknown) {
      setError(`Re-index failed: ${(e as Error).message}`)
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Document Library</h1>
        <p className="text-slate-500 text-sm mt-1">
          Upload documents to index them for local RAG search. Supported: PDF, DOCX, TXT, CSV, MD.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
          dragging ? 'border-blue-400 bg-blue-50' : 'border-slate-200 hover:border-slate-300 bg-white'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.csv,.md"
          className="hidden"
          onChange={e => e.target.files && handleFiles(e.target.files)}
        />
        {uploading
          ? <Loader2 size={32} className="mx-auto text-blue-500 animate-spin mb-3" />
          : <Upload size={32} className="mx-auto text-slate-300 mb-3" />
        }
        <p className="text-slate-600 font-medium">
          {uploading ? 'Uploading…' : 'Drop files here or click to browse'}
        </p>
        <p className="text-slate-400 text-sm mt-1">PDF, DOCX, TXT, CSV, MD · Max 100 MB</p>
      </div>

      {error && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          <AlertTriangle size={16} className="shrink-0 mt-0.5" />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-auto shrink-0"><X size={14} /></button>
        </div>
      )}

      {/* Document list */}
      {documents.length === 0 ? (
        <p className="text-center text-slate-400 py-12">No documents yet. Upload some to get started.</p>
      ) : (
        <div className="space-y-2">
          {documents.map(doc => (
            <div
              key={doc.id}
              className="bg-white border border-slate-200 rounded-xl p-4 flex items-start gap-4 hover:shadow-sm transition-shadow"
            >
              <div className="mt-0.5 shrink-0">
                {FILE_ICONS[doc.file_type] ?? <FileText size={20} className="text-slate-400" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="font-medium text-slate-800 truncate">{doc.filename}</p>
                  <StatusBadge status={doc.status} />
                  {doc.has_ocr_pages && (
                    <span className="inline-flex items-center gap-1 text-xs bg-amber-50 text-amber-700 px-2 py-0.5 rounded-full">
                      <ScanLine size={11} /> OCR ({doc.ocr_pages?.length} pages)
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  {formatBytes(doc.file_size_bytes)} · {doc.chunk_count} chunks ·{' '}
                  {new Date(doc.upload_timestamp).toLocaleDateString()}
                </p>
                {doc.status === 'error' && doc.error_message && (
                  <p className="text-xs text-red-600 mt-1">{doc.error_message}</p>
                )}
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <button
                  onClick={() => reindexDoc(doc)}
                  title="Re-index"
                  className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  <RefreshCw size={15} />
                </button>
                <button
                  onClick={() => deleteDoc(doc)}
                  title="Delete"
                  className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
