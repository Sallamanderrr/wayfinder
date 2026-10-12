import { useState, type FormEvent, type KeyboardEvent } from 'react'

export function Composer({ busy, onSend }: { busy: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState('')

  function submit(event?: FormEvent) {
    event?.preventDefault()
    if (busy || !text.trim()) return
    onSend(text)
    setText('')
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter adds a new line.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) submit(event)
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="question" className="visually-hidden">
        Your question
      </label>
      <textarea
        id="question"
        rows={2}
        maxLength={4000}
        value={text}
        placeholder="Ask about a trip…"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
      />
      <button type="submit" disabled={busy || !text.trim()}>
        {busy ? 'Working…' : 'Send'}
      </button>
    </form>
  )
}
