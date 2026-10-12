import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/** Renders agent output. Raw HTML is not rendered (react-markdown's safe default). */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: label }) => (
            <a href={href} target="_blank" rel="noreferrer">
              {label}
            </a>
          ),
          table: ({ children: rows }) => (
            <div className="table-scroll">
              <table>{rows}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
