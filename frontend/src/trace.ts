// Events streamed by the backend (see app/graph.py) and the trace the UI builds from them.

export type AgentName = 'researcher' | 'planner' | 'verifier'

export type ServerEvent =
  | { type: 'route'; next: AgentName | 'finish'; task: string; reason: string }
  | { type: 'tool_call'; agent: AgentName; tool: string; args: Record<string, unknown> }
  | { type: 'tool_result'; agent: AgentName; tool: string; content: string }
  | { type: 'agent_output'; agent: AgentName; content: string }
  | { type: 'review'; approved: boolean; issues: string[]; summary: string }
  | { type: 'final'; content: string }
  | { type: 'error'; message: string }

export interface ToolStop {
  tool: string
  args: Record<string, unknown>
  result?: string
}

/** One agent run, started by a supervisor handoff. */
export interface Leg {
  agent: AgentName
  reason: string
  task: string
  tools: ToolStop[]
  output?: string
  review?: { approved: boolean; issues: string[]; summary: string }
}

export interface Trace {
  legs: Leg[]
  finish?: { reason: string }
}

export const emptyTrace: Trace = { legs: [] }

export const AGENT_LABELS: Record<AgentName, string> = {
  researcher: 'Researcher',
  planner: 'Planner',
  verifier: 'Verifier',
}

export function applyEvent(trace: Trace, event: ServerEvent): Trace {
  switch (event.type) {
    case 'route':
      if (event.next === 'finish') return { ...trace, finish: { reason: event.reason } }
      return {
        ...trace,
        legs: [...trace.legs, { agent: event.next, reason: event.reason, task: event.task, tools: [] }],
      }
    case 'tool_call':
      return updateLastLeg(trace, (leg) => ({ ...leg, tools: [...leg.tools, { tool: event.tool, args: event.args }] }))
    case 'tool_result':
      return updateLastLeg(trace, (leg) => {
        const index = leg.tools.findIndex((stop) => stop.tool === event.tool && stop.result === undefined)
        if (index === -1) return leg
        return { ...leg, tools: leg.tools.map((stop, i) => (i === index ? { ...stop, result: event.content } : stop)) }
      })
    case 'agent_output':
      return updateLastLeg(trace, (leg) => ({ ...leg, output: event.content }))
    case 'review':
      return updateLastLeg(trace, (leg) => ({
        ...leg,
        review: { approved: event.approved, issues: event.issues, summary: event.summary },
      }))
    default:
      return trace
  }
}

function updateLastLeg(trace: Trace, update: (leg: Leg) => Leg): Trace {
  if (trace.legs.length === 0) return trace
  return { ...trace, legs: [...trace.legs.slice(0, -1), update(trace.legs[trace.legs.length - 1])] }
}

export function countToolCalls(trace: Trace): number {
  return trace.legs.reduce((total, leg) => total + leg.tools.length, 0)
}
