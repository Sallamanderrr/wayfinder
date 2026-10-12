import type { AgentName } from '../trace.ts'

interface ExampleGroup {
  title: string
  description: string
  agents: AgentName[] // the agents a question in this group typically involves
  questions: string[]
}

// Grouped by what each question demonstrates: RAG, tool use, multi-agent collaboration.
const EXAMPLES: ExampleGroup[] = [
  {
    title: 'Answer from the guides',
    description: 'Searches the city guides (RAG) and cites the passage it used.',
    agents: ['researcher'],
    questions: [
      "How do I get from Rome's airport to the centre, and what does it cost?",
      'Which neighbourhood in Stockholm suits a first visit?',
    ],
  },
  {
    title: 'Use live tools',
    description: 'Weather forecast, exchange rates and the attractions database.',
    agents: ['researcher', 'planner'],
    questions: [
      "What's the weather in Barcelona over the next three days?",
      'Which museums in Paris are open on Mondays, and what do they cost in SEK?',
    ],
  },
  {
    title: 'Plan a trip',
    description: 'The researcher gathers facts, the planner builds the plan, the verifier checks it.',
    agents: ['researcher', 'planner', 'verifier'],
    questions: [
      'Plan two days in Lisbon this weekend for two people. Our budget for sights is 1,500 SEK, and we prefer indoor sights if it rains.',
    ],
  },
]

export function EmptyState({ onAsk }: { onAsk: (question: string) => void }) {
  return (
    <div className="empty">
      <h1 className="empty-title">Where to?</h1>
      <p className="empty-lede">
        Ask about a city break in Lisbon, Barcelona, Rome, Paris or Stockholm. A researcher, a planner and a
        verifier work on each answer, and you can follow the route they take.
      </p>
      <div className="examples">
        {EXAMPLES.map((group) => (
          <section key={group.title} className="example-group">
            <h2>
              <span className="mini-route" aria-hidden="true">
                {group.agents.map((agent) => (
                  <span key={agent} className={`mini-dot mini-dot--${agent}`} />
                ))}
              </span>
              {group.title}
            </h2>
            <p className="example-description">{group.description}</p>
            {group.questions.map((question) => (
              <button key={question} type="button" className="example" onClick={() => onAsk(question)}>
                {question}
              </button>
            ))}
          </section>
        ))}
      </div>
    </div>
  )
}
