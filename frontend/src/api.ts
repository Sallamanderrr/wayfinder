import type { ServerEvent } from './trace.ts'

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

/** POST a message and call `onEvent` for every Server-Sent Event until the stream ends. */
export async function streamChat(
  threadId: string,
  message: string,
  onEvent: (event: ServerEvent) => void,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, message }),
  })
  if (!response.ok || !response.body) {
    throw new Error(`The backend couldn't handle that request (HTTP ${response.status}). Check the backend logs.`)
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += value
    // Events are separated by a blank line; each carries one "data: <json>" line.
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as ServerEvent)
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

/** The conversation as the backend remembers it (used to restore a chat after a page reload). */
export async function fetchHistory(threadId: string): Promise<HistoryMessage[]> {
  const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`)
  if (!response.ok) return []
  const data = (await response.json()) as { messages: HistoryMessage[] }
  return data.messages
}
