import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const suggestions = [
  'What is the mandate of the National Bank of Cambodia?',
  'How does NBC support financial stability?',
  'What is the role of the Cambodian riel?',
]

function Mark({ children }) { return <span className="mark">{children}</span> }

function App() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [thinkingMode, setThinkingMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [connection, setConnection] = useState('checking')
  const [error, setError] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/health`).then((response) => {
      if (!response.ok) throw new Error()
      setConnection('online')
    }).catch(() => setConnection('offline'))
  }, [])

  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [messages, loading])

  async function ask(value = question) {
    const trimmed = value.trim()
    if (!trimmed || loading) return
    setQuestion('')
    setError('')
    setLoading(true)
    setMessages((current) => [...current, { role: 'user', content: trimmed }])
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, top_k: 4, enable_thinking: thinkingMode }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'The desk could not process that request.')
      setMessages((current) => [...current, { role: 'assistant', ...data }])
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the NBC service.')
      setConnection('offline')
    } finally { setLoading(false) }
  }

  const totalAnswered = messages.filter((message) => message.role === 'assistant').length
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-symbol">N</div><div><div className="eyebrow">LOCAL RAG SYSTEM</div><strong>NBC / DESK</strong></div></div>
        <div className="rail-label">WORKSPACE</div>
        <button className="nav-item active"><Mark>⌁</Mark><span>Research chat</span><span className="nav-key">01</span></button>
        <div className="rail-label second">SYSTEM</div>
        <div className="system-card"><span className={`status-dot ${connection}`}></span><div><b>{connection === 'online' ? 'Service online' : connection === 'offline' ? 'Service offline' : 'Checking service'}</b><small>FastAPI + Ollama</small></div></div>
        <div className="sidebar-bottom"><div className="mini-stat"><span>MODEL</span><b>QWEN3:4B</b></div><div className="mini-stat"><span>INDEX</span><b>NBC QA / 1.2K</b></div><p>Answers are grounded in the local National Bank of Cambodia knowledge base.</p></div>
      </aside>
      <main className="main-panel">
        <header className="topbar"><div><div className="breadcrumb">NBC KNOWLEDGE BASE <span>/</span> RESEARCH CHAT</div><h1>Intelligence desk</h1></div><div className="top-meta"><span className="live-pill"><i></i> LIVE</span><span className="utc">LOCAL INFERENCE · CPU</span></div></header>
        <section className="chat-wrap">
          {messages.length === 0 ? <div className="welcome"><div className="hero-kicker"><Mark>01</Mark> ASK THE KNOWLEDGE BASE</div><h2>Clear answers.<br /><em>Grounded intelligence.</em></h2><p>Ask about the National Bank of Cambodia, its mandate, policies, and financial systems. Every response is retrieved from the local NBC reference index.</p><div className="suggestion-grid">{suggestions.map((item, index) => <button key={item} onClick={() => ask(item)}><span>0{index + 1}</span>{item}<b>↗</b></button>)}</div></div> : <div className="messages">{messages.map((message, index) => message.role === 'user' ? <div className="message user-message" key={index}><div className="message-label">YOU <span>QUESTION {String(index + 1).padStart(2, '0')}</span></div><div className="user-bubble">{message.content}</div></div> : <div className="message assistant-message" key={index}><div className="message-label"><Mark>✦</Mark> NBC DESK <span>{message.model_used} · {message.runtime_ms} ms · {message.used_thinking ? 'THINKING' : 'NON-THINK'}</span></div><div className="answer-card"><p>{message.answer}</p><div className="answer-footer"><span className={message.is_confident ? 'confidence' : 'muted'}>{message.is_confident ? '● GROUNDED RESPONSE' : '○ OUTSIDE INDEX'}</span><span>{message.sources?.length || 0} sources indexed</span></div></div></div>)}</div>}
          {loading && <div className="message assistant-message"><div className="message-label"><Mark>✦</Mark> NBC DESK <span>PROCESSING</span></div><div className="answer-card loading-card"><span className="loader"></span><span>Retrieving reference context and running model…</span></div></div>}
          {error && <div className="error-banner">! {error}</div>}
          <div ref={endRef} />
        </section>
        <div className="composer-area"><div className="mode-row"><span className="mode-label">REASONING MODE</span><div className="mode-switch" role="group" aria-label="Reasoning mode"><button className={!thinkingMode ? 'mode-button selected' : 'mode-button'} onClick={() => setThinkingMode(false)} aria-pressed={!thinkingMode}><span className="mode-dot"></span> NON-THINK</button><button className={thinkingMode ? 'mode-button selected thinking' : 'mode-button'} onClick={() => setThinkingMode(true)} aria-pressed={thinkingMode}><span className="mode-dot"></span> THINKING</button></div></div><div className="composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); ask() } }} placeholder="Type a question about NBC…" rows="1" /><button className="send-button" onClick={() => ask()} disabled={loading || !question.trim()} aria-label="Send question">↗</button></div><div className="composer-note"><span>ENTER TO SEND</span><span>RAG · LOCAL ONLY · {totalAnswered} {totalAnswered === 1 ? 'RESPONSE' : 'RESPONSES'}</span></div></div>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
