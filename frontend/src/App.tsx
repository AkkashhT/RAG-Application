import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { MessageSquare, Files, Settings, Shield } from 'lucide-react'
import ChatPage from './pages/ChatPage'
import DocumentsPage from './pages/DocumentsPage'
import SettingsPage from './pages/SettingsPage'

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-slate-800 text-white'
            : 'text-slate-400 hover:text-white hover:bg-slate-700'
        }`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-slate-50">
        {/* Left nav */}
        <nav className="w-52 bg-slate-900 flex flex-col p-3 gap-1 shrink-0">
          <div className="px-3 py-4 mb-2">
            <div className="flex items-center gap-2">
              <Shield size={20} className="text-blue-400" />
              <span className="text-white font-bold text-base">LocalRAG</span>
            </div>
            <p className="text-slate-500 text-xs mt-1">Private · Local · Yours</p>
          </div>
          <NavItem to="/chat" icon={<MessageSquare size={16} />} label="Chat" />
          <NavItem to="/documents" icon={<Files size={16} />} label="Documents" />
          <NavItem to="/settings" icon={<Settings size={16} />} label="Settings" />
          <div className="mt-auto px-3 py-2">
            <p className="text-slate-600 text-xs">No data leaves this machine</p>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
