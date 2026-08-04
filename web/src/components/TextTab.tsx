import { useState } from 'react'
import { detectText, type TextPIISpan } from '../lib/api'
import { l1Rgb } from '../lib/colors'
import Legend from './Legend'

const EXAMPLE_TEXT = `DISCHARGE SUMMARY
Patient: Mr. John Doe (Male, 45 yrs), DOB 12-03-1979, Blood group O+.
Address: 14 MG Road, Indiranagar, Bangalore, Karnataka 560038.
Aadhaar: 1234 5678 9012  |  PAN: ABCDE1234F  |  ABHA: 14-1234-5678-9012.
MRN/UHID: UH00219834. Insurance policy: TPA-IND-99213.
Contact: +91 98765 43210, email john.doe@example.com.
Treating physician: Dr. Asha Menon (NMC Reg. 2011/04/1123).
Payment received to UPI johndoe@okhdfcbank, A/C 50100123456789.`

function renderHighlighted(text: string, spans: TextPIISpan[]) {
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  const parts: React.ReactNode[] = []
  let prev = 0
  for (const [i, sp] of sorted.entries()) {
    if (sp.start < prev) continue
    parts.push(text.slice(prev, sp.start))
    parts.push(
      <mark key={i} style={{ background: l1Rgb(sp.l1) }} title={`${sp.category} · ${sp.score != null ? sp.score.toFixed(3) : '?'}`}>
        {text.slice(sp.start, sp.end)}
      </mark>,
    )
    prev = sp.end
  }
  parts.push(text.slice(prev))
  return parts
}

export default function TextTab({ entities, ready }: { entities: string[] | null; ready: boolean }) {
  const [text, setText] = useState(EXAMPLE_TEXT)
  const [exclude, setExclude] = useState<Set<string>>(new Set())
  const [spans, setSpans] = useState<TextPIISpan[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ranOnce, setRanOnce] = useState(false)

  function toggle(value: string) {
    const next = new Set(exclude)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    setExclude(next)
  }

  async function run() {
    setLoading(true)
    setError(null)
    try {
      setSpans(await detectText(text, [...exclude]))
      setRanOnce(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const groups = [...new Set(spans.map((s) => s.l1))].sort()

  return (
    <div className="tab-panel">
      <p className="tab-caption">Detects PII in raw text and highlights each span (color-coded by group).</p>

      {entities && (
        <details className="exclude-panel">
          <summary>Exclude categories (never detect)</summary>
          <div className="exclude-columns">
            <div>
              {entities.map((c) => (
                <label key={c} className="checkbox-row">
                  <input type="checkbox" checked={exclude.has(c)} onChange={() => toggle(c)} />
                  {c}
                </label>
              ))}
            </div>
          </div>
        </details>
      )}

      <textarea
        className="text-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={10}
      />

      <button type="button" className="primary-btn" onClick={run} disabled={!text.trim() || !ready || loading}>
        {loading ? 'Detecting…' : 'Detect PII'}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {ranOnce && (
        <>
          {spans.length > 0 ? (
            <>
              <Legend groups={groups} />
              <div className="highlighted-text">{renderHighlighted(text, spans)}</div>
              <table className="entity-table">
                <thead>
                  <tr>
                    <th>category</th>
                    <th>l1</th>
                    <th>text</th>
                    <th>start</th>
                    <th>end</th>
                    <th>score</th>
                  </tr>
                </thead>
                <tbody>
                  {spans.map((s, i) => (
                    <tr key={i}>
                      <td>{s.category}</td>
                      <td>{s.l1}</td>
                      <td>{s.text}</td>
                      <td>{s.start}</td>
                      <td>{s.end}</td>
                      <td>{s.score != null ? s.score.toFixed(3) : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p className="tab-caption">No PII detected.</p>
          )}
        </>
      )}
    </div>
  )
}
