import { useState, useEffect } from 'react'
import {
  CheckCircle, XCircle, AlertTriangle, RefreshCw, Database,
  Cpu, Server, Sliders, Save, TestTube
} from 'lucide-react'
import { settingsApi, dbApi, type AppSettings, type HealthStatus } from '../lib/api'

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 space-y-4">
      <h2 className="text-base font-semibold text-slate-800 flex items-center gap-2">
        {icon}{title}
      </h2>
      {children}
    </div>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-slate-700">{label}</label>
      {hint && <p className="text-xs text-slate-400">{hint}</p>}
      {children}
    </div>
  )
}

function Input({ value, onChange, type = 'text', step, min, max }: {
  value: string | number; onChange: (v: string) => void
  type?: string; step?: string; min?: number; max?: number
}) {
  return (
    <input
      type={type} value={value} step={step} min={min} max={max}
      onChange={e => onChange(e.target.value)}
      className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  )
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select
      value={value} onChange={e => onChange(e.target.value)}
      className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

function HealthRow({ label, status, warning }: { label: string; status: 'ok' | 'error' | 'unknown'; warning?: string }) {
  return (
    <div className="flex items-start gap-3">
      {status === 'ok'
        ? <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
        : status === 'error'
          ? <XCircle size={16} className="text-red-500 mt-0.5 shrink-0" />
          : <AlertTriangle size={16} className="text-amber-500 mt-0.5 shrink-0" />
      }
      <div>
        <p className="text-sm font-medium text-slate-700">{label}</p>
        {warning && <p className="text-xs text-amber-600">{warning}</p>}
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [ollamaModels, setOllamaModels] = useState<{ name: string; size_gb: number }[]>([])
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')

  // DB connection state
  const [connString, setConnString] = useState('')
  const [dbStatus, setDbStatus] = useState<{ connected: boolean; dialect?: string; url_display?: string } | null>(null)
  const [dbError, setDbError] = useState<string | null>(null)
  const [dbTesting, setDbTesting] = useState(false)

  useEffect(() => {
    loadSettings()
    loadHealth()
    dbApi.getStatus().then(setDbStatus).catch(() => {})
  }, [])

  async function loadSettings() {
    const s = await settingsApi.get()
    setSettings(s)
    try {
      const models = await settingsApi.listOllamaModels()
      setOllamaModels(models.models)
    } catch { /* Ollama may not be ready yet */ }
  }

  async function loadHealth() {
    setHealthLoading(true)
    try {
      setHealth(await settingsApi.health())
    } catch { /* ignore */ } finally {
      setHealthLoading(false)
    }
  }

  async function saveSettings() {
    if (!settings) return
    setSaveStatus('saving')
    try {
      await settingsApi.update(settings)
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch {
      setSaveStatus('error')
    }
  }

  function updateSetting<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setSettings(prev => prev ? { ...prev, [key]: value } : prev)
  }

  async function connectDB() {
    if (!connString.trim()) return
    setDbTesting(true)
    setDbError(null)
    try {
      const result = await dbApi.connect(connString)
      setDbStatus({ connected: true, dialect: result.dialect, url_display: result.url_display })
      setConnString('')
    } catch (e: unknown) {
      setDbError((e as Error).message)
    } finally {
      setDbTesting(false)
    }
  }

  async function disconnectDB() {
    await dbApi.disconnect()
    setDbStatus({ connected: false })
  }

  if (!settings) return <div className="p-6 text-slate-400">Loading settings…</div>

  const modelOptions = ollamaModels.map(m => ({ value: m.name, label: `${m.name} (${m.size_gb} GB)` }))
  if (modelOptions.length === 0) modelOptions.push({ value: settings.ollama_generation_model, label: settings.ollama_generation_model })

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Settings</h1>
          <p className="text-slate-500 text-sm mt-1">All settings take effect on the next request.</p>
        </div>
        <button
          onClick={saveSettings}
          disabled={saveStatus === 'saving'}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium"
        >
          <Save size={15} />
          {saveStatus === 'saving' ? 'Saving…' : saveStatus === 'saved' ? '✓ Saved' : 'Save changes'}
        </button>
      </div>

      {/* System health */}
      <Section title="System Health" icon={<Server size={16} className="text-slate-500" />}>
        <button
          onClick={loadHealth}
          className="flex items-center gap-1 text-xs text-blue-600 hover:underline"
        >
          <RefreshCw size={12} className={healthLoading ? 'animate-spin' : ''} /> Refresh
        </button>
        {health && (
          <div className="space-y-3">
            <HealthRow
              label={`Ollama — ${health.ollama.status === 'ok' ? 'connected' : 'unreachable'}`}
              status={health.ollama.status === 'ok' ? 'ok' : 'error'}
              warning={health.ollama.warning}
            />
            {health.ollama.status === 'ok' && health.ollama.gpu_active === false && (
              <HealthRow label="Ollama is running on CPU — generation will be slower" status="unknown" warning="Set up NVIDIA Container Toolkit for GPU acceleration" />
            )}
            <HealthRow
              label={`Qdrant — ${health.qdrant.status === 'ok' ? 'connected' : 'unreachable'}`}
              status={health.qdrant.status === 'ok' ? 'ok' : 'error'}
              warning={health.qdrant.warning}
            />
            <HealthRow
              label={`Reranker — ${health.reranker.loaded ? (health.reranker.on_gpu ? 'GPU' : 'CPU') : 'not loaded'}`}
              status={health.reranker.loaded ? (health.reranker.on_gpu ? 'ok' : 'unknown') : 'unknown'}
              warning={health.reranker.warning}
            />
          </div>
        )}
      </Section>

      {/* Model selection */}
      <Section title="Models" icon={<Cpu size={16} className="text-slate-500" />}>
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          <strong>VRAM guide:</strong> qwen3:8b (~6GB) · qwen3:14b default (~9GB) · qwen3:27b (~18GB Q4). Embedding (nomic-embed-text) and reranker (bge-reranker-v2-m3) add ~2GB.
        </p>
        <Field label="Generation model" hint="Loaded by Ollama for answer generation">
          <Select value={settings.ollama_generation_model} onChange={v => updateSetting('ollama_generation_model', v)} options={modelOptions} />
        </Field>
        <Field label="Embedding model" hint="Used for document and query embedding (requires re-indexing to change)">
          <Input value={settings.ollama_embed_model} onChange={v => updateSetting('ollama_embed_model', v)} />
        </Field>
        <Field label="Reranker device" hint="Set to 'cuda' for GPU acceleration (recommended)">
          <Select value={settings.reranker_device} onChange={v => updateSetting('reranker_device', v)}
            options={[{ value: 'cuda', label: 'cuda (GPU)' }, { value: 'cpu', label: 'cpu' }]} />
        </Field>
      </Section>

      {/* Retrieval settings */}
      <Section title="Retrieval & Reranking" icon={<Sliders size={16} className="text-slate-500" />}>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Rerank initial top-k" hint="Candidates fetched from Qdrant before reranking">
            <Input type="number" value={settings.reranker_initial_top_k} min={5} max={100} onChange={v => updateSetting('reranker_initial_top_k', parseInt(v))} />
          </Field>
          <Field label="Rerank final top-k" hint="Chunks kept after reranking (into LLM context)">
            <Input type="number" value={settings.reranker_top_k} min={1} max={20} onChange={v => updateSetting('reranker_top_k', parseInt(v))} />
          </Field>
          <Field label="Confidence threshold" hint="Below this reranker score, return 'not found' without calling the LLM. Run `make eval` to calibrate this for your documents.">
            <Input type="number" step="0.01" value={settings.confidence_threshold} min={0} max={1} onChange={v => updateSetting('confidence_threshold', parseFloat(v))} />
          </Field>
          <Field label="Hybrid dense weight" hint="Weight for dense vectors in RRF fusion (sparse weight = 1 − this)">
            <Input type="number" step="0.05" value={settings.hybrid_dense_weight} min={0} max={1} onChange={v => updateSetting('hybrid_dense_weight', parseFloat(v))} />
          </Field>
          <Field label="Chunk size (tokens)">
            <Input type="number" value={settings.chunk_size_tokens} min={100} max={4000} onChange={v => updateSetting('chunk_size_tokens', parseInt(v))} />
          </Field>
          <Field label="Chunk overlap (tokens)">
            <Input type="number" value={settings.chunk_overlap_tokens} min={0} max={500} onChange={v => updateSetting('chunk_overlap_tokens', parseInt(v))} />
          </Field>
        </div>
        <Field label="Router mode">
          <Select value={settings.router_mode} onChange={v => updateSetting('router_mode', v)}
            options={[
              { value: 'auto', label: 'auto (LLM classifies intent)' },
              { value: 'docs', label: 'docs only' },
              { value: 'sql', label: 'sql only' },
              { value: 'both', label: 'both (always search docs + DB)' },
            ]} />
        </Field>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Temperature">
            <Input type="number" step="0.05" value={settings.llm_temperature} min={0} max={2} onChange={v => updateSetting('llm_temperature', parseFloat(v))} />
          </Field>
          <Field label="Max tokens">
            <Input type="number" value={settings.llm_max_tokens} min={256} max={8192} onChange={v => updateSetting('llm_max_tokens', parseInt(v))} />
          </Field>
        </div>
        <Field label="Max concurrent LLM calls" hint="Prevents VRAM OOM from overlapping requests. Keep at 1 for a single GPU.">
          <Input type="number" value={settings.max_concurrent_llm_calls} min={1} max={4} onChange={v => updateSetting('max_concurrent_llm_calls', parseInt(v))} />
        </Field>
      </Section>

      {/* Database connection */}
      <Section title="SQL Database Connection" icon={<Database size={16} className="text-slate-500" />}>
        <div className="text-xs bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-blue-800 space-y-1">
          <p><strong>Security recommendation:</strong> Connect with a <em>read-only</em> database user. This is the primary safety boundary — the application also validates all queries, but a DB-level read-only role is the strongest guarantee.</p>
          <p>Example: <code className="bg-blue-100 px-1 rounded">postgresql://readonly_user:pass@host/db</code></p>
        </div>
        {dbStatus?.connected ? (
          <div className="flex items-center justify-between bg-green-50 border border-green-200 rounded-lg px-4 py-3">
            <div className="flex items-center gap-2 text-green-800 text-sm">
              <CheckCircle size={16} />
              <span>Connected ({dbStatus.dialect})</span>
              {dbStatus.url_display && <span className="text-green-600 font-mono text-xs">{dbStatus.url_display}</span>}
            </div>
            <button onClick={disconnectDB} className="text-xs text-red-600 hover:underline">Disconnect</button>
          </div>
        ) : (
          <div className="space-y-2">
            <Input
              value={connString}
              onChange={setConnString}
              type="password"
            />
            {dbError && (
              <p className="text-xs text-red-600 flex items-center gap-1"><XCircle size={12} />{dbError}</p>
            )}
            <button
              onClick={connectDB}
              disabled={!connString.trim() || dbTesting}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 disabled:opacity-50 text-sm font-medium"
            >
              {dbTesting ? <RefreshCw size={14} className="animate-spin" /> : <TestTube size={14} />}
              {dbTesting ? 'Testing…' : 'Connect & Test'}
            </button>
          </div>
        )}
      </Section>
    </div>
  )
}
