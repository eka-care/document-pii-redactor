import { useEffect, useRef, useState } from 'react'
import {
  anonymizeImage,
  deidentifyImage,
  detect,
  redact,
  type EntityTaxonomy,
  type PIIEntity,
  type PseudonymMapping,
  type RedactMode,
} from '../lib/api'
import { l1Rgb } from '../lib/colors'
import Legend from './Legend'
import MappingTable from './MappingTable'

type Action = 'redact' | 'deidentify' | 'anonymize'

const ACTION_BUTTON_LABELS: Record<Action, string> = {
  redact: 'Detect & Redact',
  deidentify: 'Detect & De-identify',
  anonymize: 'Detect & Anonymize',
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp']

// Bundled sample documents (no real PII) under web/public/samples/ — add files
// with these exact names to enable the "try a sample" shortcuts.
const SAMPLE_IMAGES = [
  { name: 'Sample 1', src: '/samples/sample-1.jpg' },
  { name: 'Sample 2', src: '/samples/sample-2.jpg' },
  { name: 'Sample 3', src: '/samples/sample-3.jpg' },
]

async function urlToFile(url: string, name: string): Promise<File> {
  const res = await fetch(url)
  const blob = await res.blob()
  return new File([blob], name, { type: blob.type })
}

export default function ImageTab({
  entities: taxonomy,
  ready,
}: {
  entities: EntityTaxonomy | null
  ready: boolean
}) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [action, setAction] = useState<Action>('redact')
  const [mode, setMode] = useState<RedactMode>('blur')
  const [color, setColor] = useState('#000000')
  const [excludeText, setExcludeText] = useState<Set<string>>(new Set())
  const [excludeVisual, setExcludeVisual] = useState<Set<string>>(new Set())

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [detectedEntities, setDetectedEntities] = useState<PIIEntity[]>([])
  const [redactedUrl, setRedactedUrl] = useState<string | null>(null)
  const [resultTitle, setResultTitle] = useState('Redacted')
  const [mapping, setMapping] = useState<PseudonymMapping | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!previewUrl || !canvasRef.current) return
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current!
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      for (const e of detectedEntities) {
        const [x0, y0, x1, y1] = e.bbox
        ctx.strokeStyle = l1Rgb(e.l1)
        ctx.lineWidth = Math.max(2, img.naturalWidth / 400)
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0)
        ctx.fillStyle = l1Rgb(e.l1)
        const fontSize = Math.max(12, img.naturalWidth / 90)
        ctx.font = `${fontSize}px sans-serif`
        const label = e.category
        const tw = ctx.measureText(label).width
        ctx.fillRect(x0, Math.max(0, y0 - fontSize - 4), tw + 6, fontSize + 4)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x0 + 3, Math.max(fontSize, y0 - 4))
      }
    }
    img.src = previewUrl
  }, [previewUrl, detectedEntities])

  function pickFile(f: File) {
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setDetectedEntities([])
    setRedactedUrl(null)
    setMapping(null)
    setError(null)
  }

  async function pickSample(src: string, name: string) {
    try {
      pickFile(await urlToFile(src, name))
    } catch {
      setError(`Couldn't load ${name} — is it present under web/public/samples/?`)
    }
  }

  function onFileInput(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError(`Unsupported file type: ${f.type || 'unknown'}`)
      return
    }
    pickFile(f)
  }

  function toggle(set: Set<string>, setter: (s: Set<string>) => void, value: string) {
    const next = new Set(set)
    if (next.has(value)) next.delete(value)
    else next.add(value)
    setter(next)
  }

  const exclude = [...excludeText, ...excludeVisual]

  async function run() {
    if (!file) return
    setLoading(true)
    setError(null)
    setMapping(null)
    try {
      const detecting = detect(file, exclude)
      if (action === 'redact') {
        const rgb = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16)) as [
          number,
          number,
          number,
        ]
        const [ents, blob] = await Promise.all([detecting, redact(file, mode, rgb, exclude)])
        setDetectedEntities(ents)
        setRedactedUrl(URL.createObjectURL(blob))
        setResultTitle(`Redacted · ${mode}`)
      } else if (action === 'deidentify') {
        const [ents, result] = await Promise.all([detecting, deidentifyImage(file, exclude)])
        setDetectedEntities(ents)
        setRedactedUrl(result.imageUrl)
        setMapping(result.mapping)
        setResultTitle('De-identified')
      } else {
        const [ents, blob] = await Promise.all([detecting, anonymizeImage(file, exclude)])
        setDetectedEntities(ents)
        setRedactedUrl(URL.createObjectURL(blob))
        setResultTitle('Anonymized')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const groups = [...new Set(detectedEntities.map((e) => e.l1))].sort()

  return (
    <div className="tab-panel">
      <p className="tab-caption">Upload a document image — it will be detected and redacted.</p>

      <div className="panel-grid">
        <section className="card">
          <h2 className="card-title">Source</h2>
          <div className="samples-row">
            {SAMPLE_IMAGES.map((s) => (
              <button
                key={s.src}
                type="button"
                className="sample-thumb"
                onClick={() => pickSample(s.src, s.name)}
                disabled={!ready}
              >
                <img
                  src={s.src}
                  alt={s.name}
                  onError={(e) => (e.currentTarget.parentElement!.style.display = 'none')}
                />
                <span>{s.name}</span>
              </button>
            ))}
            <label className="upload-btn">
              Upload image
              <input type="file" accept={ACCEPTED_TYPES.join(',')} onChange={onFileInput} hidden />
            </label>
          </div>
        </section>

        <section className="card">
          <h2 className="card-title">Options</h2>
          <div className="controls-row">
            <label className="field">
              <span className="field-label">Action</span>
              <select value={action} onChange={(e) => setAction(e.target.value as Action)}>
                <option value="redact">Redact</option>
                <option value="anonymize">Anonymize</option>
                <option value="deidentify">De-identify</option>
              </select>
            </label>
            {action === 'redact' && (
              <>
                <label className="field">
                  <span className="field-label">Redaction mode</span>
                  <select value={mode} onChange={(e) => setMode(e.target.value as RedactMode)}>
                    <option value="solid">solid</option>
                    <option value="blur">blur</option>
                    <option value="pixelate">pixelate</option>
                  </select>
                </label>
                <label className="field">
                  <span className="field-label">Fill color</span>
                  <input
                    type="color"
                    value={color}
                    disabled={mode !== 'solid'}
                    onChange={(e) => setColor(e.target.value)}
                  />
                  {mode !== 'solid' && <span className="field-hint">Solid mode only</span>}
                </label>
              </>
            )}
          </div>
          <p className="tab-caption action-hint">
            {action === 'redact' &&
              'Destroys the PII regions — nothing is kept, nothing is recoverable.'}
            {action === 'deidentify' &&
              'Replaces text PII with consistent pseudonyms (Person_1) rendered in place, faces/signatures with placeholders, and returns the mapping for authorized re-linking.'}
            {action === 'anonymize' &&
              'One-way: generalizes ages/dates/geography, collapses names and IDs to tokens, blacks out faces/signatures. No mapping exists.'}
          </p>

          {taxonomy && (
            <details className="exclude-panel">
              <summary>Exclude categories (never detect/redact)</summary>
              <div className="exclude-columns">
                <div>
                  <h3 className="exclude-heading">Text</h3>
                  {taxonomy.text.map((c) => (
                    <label key={c} className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={excludeText.has(c)}
                        onChange={() => toggle(excludeText, setExcludeText, c)}
                      />
                      {c}
                    </label>
                  ))}
                </div>
                <div>
                  <h3 className="exclude-heading">Visual</h3>
                  {taxonomy.visual.map((c) => (
                    <label key={c} className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={excludeVisual.has(c)}
                        onChange={() => toggle(excludeVisual, setExcludeVisual, c)}
                      />
                      {c}
                    </label>
                  ))}
                </div>
              </div>
            </details>
          )}
        </section>
      </div>

      <button type="button" className="primary-btn" onClick={run} disabled={!file || !ready || loading}>
        {loading ? 'Working…' : ACTION_BUTTON_LABELS[action]}
      </button>

      {error && <div className="error-banner">{error}</div>}

      {detectedEntities.length > 0 && <Legend groups={groups} />}

      {previewUrl && (
        <div className="results-row">
          <section className="card">
            <h2 className="card-title">Detections</h2>
            <canvas ref={canvasRef} className="result-image" />
          </section>
          {redactedUrl && (
            <section className="card">
              <h2 className="card-title">{resultTitle}</h2>
              <img src={redactedUrl} alt={`${resultTitle} result`} className="result-image" />
              <a className="btn-secondary download-btn" href={redactedUrl} download="result.png">
                Download PNG
              </a>
            </section>
          )}
        </div>
      )}

      {mapping && <MappingTable mapping={mapping} />}

      {detectedEntities.length > 0 && (
        <section className="card">
          <h2 className="card-title">Detected entities</h2>
          <div className="table-scroll">
            <table className="entity-table">
              <thead>
                <tr>
                  <th>kind</th>
                  <th>category</th>
                  <th>l1</th>
                  <th>text</th>
                  <th>score</th>
                  <th>bbox</th>
                </tr>
              </thead>
              <tbody>
                {detectedEntities.map((e, i) => (
                  <tr key={i}>
                    <td>{e.kind}</td>
                    <td>{e.category}</td>
                    <td>
                      <span className="l1-dot" style={{ background: l1Rgb(e.l1) }} />
                      {e.l1}
                    </td>
                    <td>{e.text ?? ''}</td>
                    <td>{e.score != null ? e.score.toFixed(3) : ''}</td>
                    <td>{e.bbox.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
