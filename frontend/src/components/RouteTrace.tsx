import { useState } from 'react'
import { AGENT_LABELS, countToolCalls, type Leg, type Trace } from '../trace.ts'
import { Markdown } from './Markdown.tsx'

/**
 * The agents' route for one answer, drawn like a transit line: each agent run is a
 * station in that agent's colour, and each tool call is a stop along the line.
 * Open while the agents work; collapsed to a one-line summary once the answer is in.
 */
export function RouteTrace({ trace, working }: { trace: Trace; working: boolean }) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const open = userOpen ?? working
  const tools = countToolCalls(trace)

  if (!working && trace.legs.length === 0) return null // answered directly by the supervisor

  return (
    <section className="trace" aria-label="How the agents worked on this answer">
      <button type="button" className="trace-toggle" aria-expanded={open} onClick={() => setUserOpen(!open)}>
        <MiniRoute legs={trace.legs} />
        <span className="trace-toggle-text">{summarise(trace, working, open, tools)}</span>
        <span className="trace-toggle-hint">{open ? 'Hide steps' : 'Show steps'}</span>
      </button>

      {open && (
        <ol className="route">
          {trace.legs.map((leg, index) => (
            <Station key={index} leg={leg} />
          ))}
          {trace.finish && (
            <li className="station station--supervisor station--terminus">
              <p className="station-head">
                <span className="station-name">Supervisor</span> {trace.finish.reason}
              </p>
            </li>
          )}
          {working && !trace.finish && (
            <li className="station station--pending" aria-live="polite">
              <p className="station-head">{pendingLabel(trace)}</p>
            </li>
          )}
        </ol>
      )}
    </section>
  )
}

function Station({ leg }: { leg: Leg }) {
  return (
    <li className={`station station--${leg.agent}`}>
      <p className="station-head">
        <span className="station-name">{AGENT_LABELS[leg.agent]}</span> {leg.reason}
      </p>
      {leg.task && (
        <details className="detail">
          <summary>Task from the supervisor</summary>
          <p className="detail-body">{leg.task}</p>
        </details>
      )}
      {leg.tools.length > 0 && (
        <ul className="stops">
          {leg.tools.map((stop, index) => (
            <li key={index} className={stop.result === undefined ? 'stop stop--running' : 'stop'}>
              <code className="call">
                {stop.tool}({formatArgs(stop.args)})
              </code>
              {stop.result === undefined ? (
                <span className="stop-status">running</span>
              ) : (
                <details className="detail">
                  <summary>Result</summary>
                  <pre className="detail-body result">{stop.result}</pre>
                </details>
              )}
            </li>
          ))}
        </ul>
      )}
      {leg.output && (
        <details className="detail">
          <summary>{leg.agent === 'planner' ? 'Draft' : 'Findings'}</summary>
          <div className="detail-body">
            <Markdown>{leg.output}</Markdown>
          </div>
        </details>
      )}
      {leg.review && (
        <div className={leg.review.approved ? 'verdict verdict--approved' : 'verdict verdict--changes'}>
          <p>
            <strong>{leg.review.approved ? 'Approved.' : 'Changes requested.'}</strong> {leg.review.summary}
          </p>
          {leg.review.issues.length > 0 && (
            <ul>
              {leg.review.issues.map((issue, index) => (
                <li key={index}>{issue}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  )
}

function MiniRoute({ legs }: { legs: Leg[] }) {
  return (
    <span className="mini-route" aria-hidden="true">
      <span className="mini-dot mini-dot--supervisor" />
      {legs.map((leg, index) => (
        <span key={index} className={`mini-dot mini-dot--${leg.agent}`} />
      ))}
    </span>
  )
}

function summarise(trace: Trace, working: boolean, open: boolean, tools: number): string {
  // While collapsed and working, the summary is the live status; otherwise it's the route so far.
  if (working && !open) return pendingLabel(trace)
  if (trace.legs.length === 0) return 'Starting…'
  const route = trace.legs.map((leg) => AGENT_LABELS[leg.agent]).join(' → ')
  return `${route}, ${tools} tool ${tools === 1 ? 'call' : 'calls'}`
}

function pendingLabel(trace: Trace): string {
  const leg = trace.legs.at(-1)
  if (!leg) return 'Supervisor is choosing who should handle this…'
  if (leg.review) return 'Supervisor is deciding what to do with the review…'
  const running = leg.tools.find((stop) => stop.result === undefined)
  if (running) return `${AGENT_LABELS[leg.agent]} is calling ${running.tool}…`
  if (leg.output) return 'Supervisor is deciding the next step…'
  return `${AGENT_LABELS[leg.agent]} is working…`
}

function formatArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => `${key}: ${typeof value === 'object' || typeof value === 'string' ? JSON.stringify(value) : String(value)}`)
    .join(', ')
}
