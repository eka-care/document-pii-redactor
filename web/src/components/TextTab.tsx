import { useState } from 'react'
import {
  anonymizeText,
  deidentifyText,
  detectText,
  redactText,
  type PseudonymMapping,
  type TextPIISpan,
} from '../lib/api'
import { l1Rgb, l1Tint } from '../lib/colors'
import Legend from './Legend'
import MappingTable from './MappingTable'

type Action = 'detect' | 'redact' | 'deidentify' | 'anonymize'

const ACTION_BUTTON_LABELS: Record<Action, string> = {
  detect: 'Detect PII',
  redact: 'Redact',
  deidentify: 'De-identify',
  anonymize: 'Anonymize',
}

const ACTION_HINTS: Record<Action, string> = {
  detect: 'Finds PII spans and highlights them in place — nothing is changed.',
  redact: 'Replaces each span with a [category] token. One-way, nothing kept.',
  deidentify:
    'Replaces each entity with a consistent pseudonym (Person_1) and returns the mapping for authorized re-linking.',
  anonymize:
    'One-way: generalizes ages/dates/geography, collapses names and IDs to unnumbered tokens. No mapping exists.',
}

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
      <mark
        key={i}
        className="pii-mark"
        style={{ background: l1Tint(sp.l1), borderColor: l1Rgb(sp.l1) }}
        title={`${sp.category} · ${sp.score != null ? sp.score.toFixed(3) : '?'}`}
      >
        {text.slice(sp.start, sp.end)}
        <sup className="pii-tag" style={{ color: l1Rgb(sp.l1) }}>
          {sp.category}
        </sup>
      </mark>,
    )
    prev = sp.end
  }
  parts.push(text.slice(prev))
  return parts
}

export default function TextTab({ entities, ready }: { entities: string[] | null; ready: boolean }) {
  const [text, setText] = useState(EXAMPLE_TEXT)
  const [action, setAction] = useState<Action>('detect')
  const [exclude, setExclude] = useState<Set<string>>(new Set())
  const [spans, setSpans] = useState<TextPIISpan[]>([])
  const [outputText, setOutputText] = useState<string | null>(null)
  const [outputTitle, setOutputTitle] = useState('')
  const [mapping, setMapping] = useState<PseudonymMapping | null>(null)
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
    setMapping(null)
    setOutputText(null)
    try {
      const excludeList = [...exclude]
      if (action === 'detect') {
        setSpans(await detectText(text, excludeList))
      } else {
        setSpans([])
        if (action === 'redact') {
          setOutputText(await redactText(text, excludeList))
          setOutputTitle('Redacted')
        } else if (action === 'deidentify') {
          const result = await deidentifyText(text, excludeList)
          setOutputText(result.text)
          setMapping(result.mapping)
          setOutputTitle('De-identified')
        } else {
          setOutputText(await anonymizeText(text, excludeList))
          setOutputTitle('Anonymized')
        }
      }
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
      <p className="tab-caption">Detects PII in raw text and highlights each span, color-coded by group.</p>

      <section className="card">
        <h2 className="card-title">Input</h2>

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

        <div className="controls-row">
          <label className="field">
            <span className="field-label">Action</span>
            <select value={action} onChange={(e) => setAction(e.target.value as Action)}>
              <option value="detect">Detect (highlight)</option>
              <option value="redact">Redact</option>
              <option value="anonymize">Anonymize</option>
              <option value="deidentify">De-identify</option>
            </select>
          </label>
          <button type="button" className="primary-btn" onClick={run} disabled={!text.trim() || !ready || loading}>
            {loading ? 'Working…' : ACTION_BUTTON_LABELS[action]}
          </button>
        </div>
        <p className="tab-caption action-hint">{ACTION_HINTS[action]}</p>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {outputText != null && (
        <section className="card">
          <h2 className="card-title">{outputTitle}</h2>
          <div className="output-text">{outputText}</div>
        </section>
      )}

      {mapping && <MappingTable mapping={mapping} />}

      {ranOnce && action === 'detect' && (
        <>
          {spans.length > 0 ? (
            <>
              <section className="card">
                <h2 className="card-title">Highlighted</h2>
                <Legend groups={groups} />
                <div className="highlighted-text">{renderHighlighted(text, spans)}</div>
              </section>

              <section className="card">
                <h2 className="card-title">Detected spans</h2>
                <div className="table-scroll">
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
                          <td>
                            <span className="l1-dot" style={{ background: l1Rgb(s.l1) }} />
                            {s.l1}
                          </td>
                          <td>{s.text}</td>
                          <td>{s.start}</td>
                          <td>{s.end}</td>
                          <td>{s.score != null ? s.score.toFixed(3) : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : (
            <p className="tab-caption">No PII detected.</p>
          )}
        </>
      )}
    </div>
  )
}
