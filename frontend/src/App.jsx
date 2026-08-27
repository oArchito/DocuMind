import { useState, useRef, useEffect } from 'react'
import './App.css'

// Read the backend URL from environment variables (set in .env)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Main DocuMind application component.
 * 
 * Flow: Upload PDF → Ask questions → See answers with sources
 */
function App() {
  // ── State ──────────────────────────────────────────────────────────
  const [documentId, setDocumentId] = useState(null)     // ID from backend after upload
  const [filename, setFilename] = useState('')            // Name of uploaded file
  const [numChunks, setNumChunks] = useState(0)           // How many chunks were created
  const [messages, setMessages] = useState([])            // Chat message history
  const [inputText, setInputText] = useState('')          // Current text in the input box
  const [isUploading, setIsUploading] = useState(false)   // Upload in progress?
  const [isAsking, setIsAsking] = useState(false)         // Waiting for an answer?
  const [error, setError] = useState(null)                // Error message to display

  // Refs for auto-scrolling and file input
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  // Auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isAsking])


  // ── Upload Handler ─────────────────────────────────────────────────

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return

    // Client-side validation
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10 MB.')
      return
    }

    setIsUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Upload failed (status ${response.status})`)
      }

      const data = await response.json()
      setDocumentId(data.document_id)
      setFilename(data.filename)
      setNumChunks(data.num_chunks)
      setMessages([])  // Clear chat for new document
    } catch (err) {
      setError(err.message || 'Failed to upload the file. Is the backend running?')
    } finally {
      setIsUploading(false)
      // Reset file input so the same file can be re-uploaded
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }


  // ── Ask Handler ────────────────────────────────────────────────────

  async function handleAsk(e) {
    e?.preventDefault()
    const question = inputText.trim()
    if (!question || !documentId || isAsking) return

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', text: question }])
    setInputText('')
    setIsAsking(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: documentId,
          question: question,
          top_k: 4,
        }),
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Request failed (status ${response.status})`)
      }

      const data = await response.json()
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: data.answer,
          sources: data.sources,
        },
      ])
    } catch (err) {
      setError(err.message || 'Failed to get an answer. Please try again.')
    } finally {
      setIsAsking(false)
    }
  }


  // ── Remove Document ────────────────────────────────────────────────

  function handleRemoveDoc() {
    setDocumentId(null)
    setFilename('')
    setNumChunks(0)
    setMessages([])
    setError(null)
  }


  // ── Keyboard shortcut: Enter to send ───────────────────────────────

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }


  function handlePresetQuestion(qText) {
    if (!documentId || isAsking) return
    handleAskWithQuestion(qText)
  }

  async function handleAskWithQuestion(qText) {
    if (!qText || !documentId || isAsking) return

    setMessages(prev => [...prev, { role: 'user', text: qText }])
    setInputText('')
    setIsAsking(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: documentId,
          question: qText,
          top_k: 4,
        }),
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `Request failed (status ${response.status})`)
      }

      const data = await response.json()
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: data.answer,
          sources: data.sources,
        },
      ])
    } catch (err) {
      setError(err.message || 'Failed to get an answer. Please try again.')
    } finally {
      setIsAsking(false)
    }
  }

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div className="studio-app">
      {/* Top Navbar */}
      <nav className="studio-navbar">
        <div className="studio-navbar__brand">
          <div className="brand-icon-box">
            <span className="brand-sparkle">✦</span>
          </div>
          <div className="brand-text">
            <h1 className="brand-title">DocuMind Studio</h1>
            <span className="brand-subtitle">Vector Document Intelligence</span>
          </div>
        </div>

        <div className="studio-navbar__actions">
          <div className="model-badge">
            <span className="model-dot"></span>
            <span>Gemini 3.5 Flash RAG</span>
          </div>
        </div>
      </nav>

      {/* Error Toast */}
      {error && (
        <div className="error-toast" id="error-toast">
          <span className="error-toast__icon">⚠️</span>
          <span>{error}</span>
          <button className="error-toast__close" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* 2-Column Studio Workspace Grid */}
      <div className="studio-grid">
        {/* LEFT PANEL: Document Control & Insights */}
        <aside className="left-panel">
          <div className="panel-card">
            <div className="panel-card__header">
              <span className="panel-card__title">Document Source</span>
              {documentId && (
                <button className="change-doc-btn" onClick={handleRemoveDoc} title="Change PDF">
                  <span>Change</span> ✕
                </button>
              )}
            </div>

            {!documentId ? (
              <div
                className={`studio-upload-zone ${isUploading ? 'upload--disabled' : ''}`}
                onClick={() => !isUploading && fileInputRef.current?.click()}
              >
                <div className="upload-icon-circle">
                  {isUploading ? (
                    <div className="spinner"></div>
                  ) : (
                    <svg className="upload-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  )}
                </div>
                <div className="upload-primary-text">
                  {isUploading ? 'Building Vector Index...' : 'Upload PDF Document'}
                </div>
                <div className="upload-secondary-text">
                  Drag & drop your PDF file or click to browse
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleUpload}
                  disabled={isUploading}
                  style={{ display: 'none' }}
                />
              </div>
            ) : (
              <div className="active-doc-card">
                <div className="doc-file-badge">
                  <span className="doc-pdf-icon">📄</span>
                  <div className="doc-file-info">
                    <div className="doc-filename">{filename}</div>
                    <div className="doc-status-line">
                      <span className="status-tag">Vector Index Active</span>
                      <span className="chunk-count">{numChunks} Chunks</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Quick Insights Cards */}
          <div className="panel-card">
            <div className="panel-card__header">
              <span className="panel-card__title">Quick Insights</span>
            </div>
            <div className="quick-action-list">
              <button
                className="insight-btn"
                disabled={!documentId || isAsking}
                onClick={() => handlePresetQuestion('Provide a concise executive summary of this document.')}
              >
                <span className="insight-icon">✦</span>
                <div className="insight-text">
                  <strong>Executive Summary</strong>
                  <span>High-level overview & takeaways</span>
                </div>
              </button>

              <button
                className="insight-btn"
                disabled={!documentId || isAsking}
                onClick={() => handlePresetQuestion('What technical skills, frameworks, tools, or tech stack are mentioned?')}
              >
                <span className="insight-icon">🛠️</span>
                <div className="insight-text">
                  <strong>Tech Stack & Skills</strong>
                  <span>Tools, technologies & frameworks</span>
                </div>
              </button>

              <button
                className="insight-btn"
                disabled={!documentId || isAsking}
                onClick={() => handlePresetQuestion('What are the key dates, metrics, numbers, or performance figures?')}
              >
                <span className="insight-icon">📊</span>
                <div className="insight-text">
                  <strong>Key Metrics & Figures</strong>
                  <span>Dates, statistics & measurements</span>
                </div>
              </button>
            </div>
          </div>
        </aside>

        {/* RIGHT PANEL: Interactive Chat & Knowledge Stream */}
        <section className="right-panel">
          <div className="messages-workspace" id="messages-container">
            {messages.length === 0 && (
              <div className="studio-empty-state">
                <div className="empty-icon-box">✦</div>
                <h2 className="empty-title">
                  {documentId ? 'Document Ready for Query' : 'Welcome to DocuMind Studio'}
                </h2>
                <p className="empty-description">
                  {documentId
                    ? 'Ask questions or click an insight card on the left panel to generate structured AI answers.'
                    : 'Upload a PDF on the left panel to initialize your vector knowledge base.'}
                </p>
              </div>
            )}

            {messages.map((msg, i) => (
              <Message key={i} message={msg} />
            ))}

            {isAsking && (
              <div className="message message--assistant">
                <div className="message__avatar">✦</div>
                <div className="message__content">
                  <div className="typing-indicator">
                    <span>Searching vector chunks with FAISS...</span>
                    <div className="loading-dots">
                      <div className="loading-dots__dot"></div>
                      <div className="loading-dots__dot"></div>
                      <div className="loading-dots__dot"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Bottom Chat Bar */}
          <form className="studio-chat-bar" onSubmit={handleAsk}>
            <input
              className="studio-chat-input"
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={documentId ? 'Ask anything about your document...' : 'Upload a PDF on the left first...'}
              disabled={!documentId || isAsking}
            />
            <button
              className="studio-send-btn"
              type="submit"
              disabled={!documentId || isAsking || !inputText.trim()}
              title="Send query"
            >
              <span>Ask</span>
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
              </svg>
            </button>
          </form>
        </section>
      </div>
    </div>
  )
}


/**
 * A robust, crash-proof Markdown renderer component.
 * Formats headers, bullet lists, bold text, code tags, and paragraphs.
 */
function FormattedMarkdown({ content }) {
  if (!content) return null

  const lines = content.split('\n')
  const blocks = []
  let currentList = []

  const flushList = () => {
    if (currentList.length > 0) {
      blocks.push({ type: 'list', items: [...currentList] })
      currentList = []
    }
  }

  lines.forEach((line) => {
    const trimmed = line.trim()
    if (!trimmed) {
      flushList()
      return
    }

    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      currentList.push(trimmed.slice(2))
    } else {
      flushList()
      if (trimmed.startsWith('### ')) {
        blocks.push({ type: 'h3', text: trimmed.slice(4) })
      } else if (trimmed.startsWith('## ')) {
        blocks.push({ type: 'h2', text: trimmed.slice(3) })
      } else if (trimmed.startsWith('# ')) {
        blocks.push({ type: 'h1', text: trimmed.slice(2) })
      } else {
        blocks.push({ type: 'p', text: trimmed })
      }
    }
  })
  flushList()

  const formatInline = (text) => {
    // Parse bold text **bold**
    const parts = text.split(/(\*\*.*?\*\*)/g)
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return <strong key={i}>{part.slice(2, -2)}</strong>
      }
      return part
    })
  }

  return (
    <div className="markdown-content">
      {blocks.map((block, i) => {
        if (block.type === 'h1') return <h1 key={i}>{formatInline(block.text)}</h1>
        if (block.type === 'h2') return <h2 key={i}>{formatInline(block.text)}</h2>
        if (block.type === 'h3') return <h3 key={i}>{formatInline(block.text)}</h3>
        if (block.type === 'list') {
          return (
            <ul key={i}>
              {block.items.map((item, j) => (
                <li key={j}>{formatInline(item)}</li>
              ))}
            </ul>
          )
        }
        return <p key={i}>{formatInline(block.text)}</p>
      })}
    </div>
  )
}


/**
 * A single chat message (user or assistant).
 * Assistant messages can have expandable sources.
 */
function Message({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div className={`message message--${message.role}`}>
      <div className="message__avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message__content">
        <div className="message__bubble">
          {isUser ? (
            message.text
          ) : (
            <FormattedMarkdown content={message.text} />
          )}
        </div>

        {/* Sources (only for assistant messages that have them) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="sources">
            <button
              className="sources__toggle"
              onClick={() => setShowSources(!showSources)}
            >
              <span
                className={`sources__toggle-arrow ${showSources ? 'sources__toggle-arrow--open' : ''}`}
              >
                ▶
              </span>
              {showSources ? 'Hide' : 'Show'} {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
            </button>

            {showSources && (
              <div className="sources__list">
                {message.sources.map((source, i) => (
                  <div className="source-chip" key={i}>
                    <div className="source-chip__header">
                      <span>Chunk #{source.chunk_id}</span>
                      <span className="source-chip__score">
                        Score: {source.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="source-chip__text">{source.chunk_text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}


export default App
