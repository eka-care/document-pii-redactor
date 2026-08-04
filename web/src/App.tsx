import { useEffect, useRef, useState } from 'react'
import { getEntities, getHealth, getTextEntities, type EntityTaxonomy } from './lib/api'
import ImageTab from './components/ImageTab'
import TextTab from './components/TextTab'
import './App.css'

type Tab = 'image' | 'text'

export default function App() {
  const [tab, setTab] = useState<Tab>('image')
  const [ready, setReady] = useState(false)
  const [device, setDevice] = useState<string | null>(null)
  const [taxonomy, setTaxonomy] = useState<EntityTaxonomy | null>(null)
  const [textEntities, setTextEntities] = useState<string[] | null>(null)
  const backoffRef = useRef(1000)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    async function poll() {
      try {
        const h = await getHealth()
        if (cancelled) return
        setReady(true)
        setDevice(h.device)
        return // healthy — stop polling
      } catch {
        if (cancelled) return
        backoffRef.current = Math.min(backoffRef.current * 1.5, 10000)
      }
      timer = setTimeout(poll, backoffRef.current)
    }
    poll()

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    if (!ready) return
    getEntities().then(setTaxonomy).catch(() => {})
    getTextEntities().then((r) => setTextEntities(r.text)).catch(() => {})
  }, [ready])

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛡️ Eka PII Redactor</h1>
        <p className="subtitle">
          {ready ? `Models ready · device=${device}` : 'Waking up the model… this can take a minute on the first request.'}
        </p>
      </header>

      <nav className="tabs">
        <button className={tab === 'image' ? 'tab active' : 'tab'} onClick={() => setTab('image')}>
          🖼️ Image
        </button>
        <button className={tab === 'text' ? 'tab active' : 'tab'} onClick={() => setTab('text')}>
          📝 Text
        </button>
      </nav>

      {tab === 'image' ? (
        <ImageTab entities={taxonomy} ready={ready} />
      ) : (
        <TextTab entities={textEntities} ready={ready} />
      )}
    </div>
  )
}
