import { useEffect, useRef, useState } from 'react'
import { fetchHistory, streamChat } from './api.ts'
import { Composer } from './components/Composer.tsx'
import { EmptyState } from './components/EmptyState.tsx'
import { Markdown } from './components/Markdown.tsx'
import { RouteTrace } from './components/RouteTrace.tsx'
import { applyEvent, emptyTrace, type Trace } from './trace.ts'

interface UserMessage {
  id: string
  role: 'user'
  content: string
}

interface AssistantMessage {
  id: string
  role: 'assistant'
  content: string
  trace: Trace
  status: 'working' | 'done' | 'error'
  error?: string
}

type ChatMessage = UserMessage | AssistantMessage

const THREAD_KEY = 'wayfinder.threadId'

function newThreadId(): string {
  const id = crypto.randomUUID()
  try {
    localStorage.setItem(THREAD_KEY, id)
  } catch {
    // Storage can be unavailable (private mode); the chat still works for this page view.
  }
  return id
}

function savedThreadId(): string {
  try {
    return localStorage.getItem(THREAD_KEY) ?? newThreadId()
  } catch {
    return newThreadId()
  }
}

export default function App() {
  const [threadId, setThreadId] = useState(savedThreadId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef<HTMLElement>(null)
  const jumpToBottom = useRef(true) // set when the user sends or a chat is restored

  // The backend keeps the conversation (LangGraph checkpointer): restore it after a reload.
  useEffect(() => {
    let cancelled = false
    fetchHistory(threadId)
      .then((history) => {
        if (cancelled || history.length === 0) return
        const restored: ChatMessage[] = history.map((message) =>
          message.role === 'user'
            ? { id: crypto.randomUUID(), role: 'user', content: message.content }
            : { id: crypto.randomUUID(), role: 'assistant', content: message.content, trace: emptyTrace, status: 'done' },
        )
        jumpToBottom.current = true
        setMessages((current) => (current.length ? current : restored))
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [threadId])

  // Follow the conversation as it grows, unless the reader has scrolled up to read.
  useEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
    if (jumpToBottom.current || distanceFromBottom < 240) element.scrollTop = element.scrollHeight
    jumpToBottom.current = false
  }, [messages])

  async function send(text: string) {
    const question = text.trim()
    if (!question || busy) return
    const assistantId = crypto.randomUUID()
    jumpToBottom.current = true
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: 'user', content: question },
      { id: assistantId, role: 'assistant', content: '', trace: emptyTrace, status: 'working' },
    ])
    setBusy(true)

    const update = (change: (message: AssistantMessage) => AssistantMessage) =>
      setMessages((current) =>
        current.map((message) => (message.id === assistantId && message.role === 'assistant' ? change(message) : message)),
      )

    try {
      await streamChat(threadId, question, (event) => {
        if (event.type === 'final') update((m) => ({ ...m, content: event.content, status: 'done' }))
        else if (event.type === 'error') update((m) => ({ ...m, status: 'error', error: event.message }))
        else update((m) => ({ ...m, trace: applyEvent(m.trace, event) }))
      })
      update((m) =>
        m.status === 'working' ? { ...m, status: 'error', error: 'The response ended before an answer arrived. Try again.' } : m,
      )
    } catch (error) {
      // fetch() rejects with a TypeError when the server can't be reached at all.
      const message =
        error instanceof TypeError
          ? "Can't reach the backend. Start it on port 8000 (see the README), then try again."
          : error instanceof Error
            ? error.message
            : 'Something went wrong. Try again.'
      update((m) => ({ ...m, status: 'error', error: message }))
    } finally {
      setBusy(false)
    }
  }

  function startNewConversation() {
    setMessages([])
    setThreadId(newThreadId())
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M8 24 V14 a6 6 0 0 1 6 -6 H24" />
            <circle cx="8" cy="24" r="4.5" />
            <circle cx="24" cy="8" r="4.5" />
          </svg>
          <span className="brand-name">Wayfinder</span>
        </div>
        <button type="button" className="ghost" onClick={startNewConversation} disabled={busy || messages.length === 0}>
          New conversation
        </button>
      </header>

      <main className="conversation" ref={scrollRef}>
        <div className="column">
          {messages.length === 0 ? (
            <EmptyState onAsk={send} />
          ) : (
            messages.map((message) =>
              message.role === 'user' ? (
                <p key={message.id} className="user-message">
                  {message.content}
                </p>
              ) : (
                <article key={message.id} className="assistant-message">
                  <RouteTrace trace={message.trace} working={message.status === 'working'} />
                  {message.content && <Markdown>{message.content}</Markdown>}
                  {message.status === 'error' && (
                    <p className="error" role="alert">
                      {message.error}
                    </p>
                  )}
                </article>
              ),
            )
          )}
        </div>
      </main>

      <footer className="composer-bar">
        <div className="column">
          <Composer busy={busy} onSend={send} />
        </div>
      </footer>
    </div>
  )
}
