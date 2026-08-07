export interface PIIEntity {
  category: string
  kind: 'text' | 'visual'
  bbox: [number, number, number, number]
  l1: string
  text: string | null
  score: number | null
}

export interface TextPIISpan {
  category: string
  start: number
  end: number
  l1: string
  text: string
  score: number | null
}

export interface EntityTaxonomy {
  text: string[]
  visual: string[]
}

export type RedactMode = 'solid' | 'blur' | 'pixelate'

async function assertOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ''}`)
  }
  return res
}

export async function getHealth(): Promise<{ status: string; device: string }> {
  const res = await assertOk(await fetch('/health'))
  return res.json()
}

export async function getEntities(): Promise<EntityTaxonomy> {
  const res = await assertOk(await fetch('/entities'))
  return res.json()
}

export async function getTextEntities(): Promise<{ text: string[] }> {
  const res = await assertOk(await fetch('/entities-text'))
  return res.json()
}

export async function detect(file: File, categories?: string[]): Promise<PIIEntity[]> {
  const form = new FormData()
  form.append('file', file)
  const qs = categories?.length ? `?categories=${encodeURIComponent(categories.join(','))}` : ''
  const res = await assertOk(await fetch(`/detect${qs}`, { method: 'POST', body: form }))
  const data = await res.json()
  return data.entities as PIIEntity[]
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  return [r, g, b].map((c) => c.toString(16).padStart(2, '0')).join('')
}

export async function redact(
  file: File,
  mode: RedactMode,
  color: [number, number, number],
  categories?: string[],
): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)
  form.append('color', rgbToHex(color))
  const qs = categories?.length ? `?categories=${encodeURIComponent(categories.join(','))}` : ''
  const res = await assertOk(await fetch(`/redact${qs}`, { method: 'POST', body: form }))
  return res.blob()
}

export async function detectText(text: string, categories?: string[]): Promise<TextPIISpan[]> {
  const res = await assertOk(
    await fetch('/detect-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, categories }),
    }),
  )
  const data = await res.json()
  return data.spans as TextPIISpan[]
}

// label -> {pseudonym: original} plus per-label counters; opaque to the UI
// beyond rendering the entries table.
export interface PseudonymMapping {
  entries: Record<string, Record<string, string>>
  counters: Record<string, number>
}

async function postJson(path: string, body: unknown): Promise<any> {
  const res = await assertOk(
    await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
  return res.json()
}

export async function redactText(text: string, categories?: string[]): Promise<string> {
  const data = await postJson('/redact-text', { text, categories, mask: '[{category}]' })
  return data.text
}

export type DeidStrategy = 'counter' | 'hash'

export async function deidentifyText(
  text: string,
  categories?: string[],
  strategy: DeidStrategy = 'counter',
): Promise<{ text: string; mapping: PseudonymMapping }> {
  return postJson('/deidentify-text', { text, categories, strategy })
}

export async function anonymizeText(text: string, categories?: string[]): Promise<string> {
  const data = await postJson('/anonymize-text', { text, categories })
  return data.text
}

export async function deidentifyImage(
  file: File,
  categories?: string[],
  strategy: DeidStrategy = 'counter',
): Promise<{ imageUrl: string; mapping: PseudonymMapping }> {
  const form = new FormData()
  form.append('file', file)
  const params = new URLSearchParams({ strategy })
  if (categories?.length) params.set('categories', categories.join(','))
  const res = await assertOk(
    await fetch(`/deidentify?${params}`, { method: 'POST', body: form }),
  )
  const data = await res.json()
  return { imageUrl: `data:image/png;base64,${data.image}`, mapping: data.mapping }
}

export async function anonymizeImage(file: File, categories?: string[]): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)
  const qs = categories?.length ? `?categories=${encodeURIComponent(categories.join(','))}` : ''
  const res = await assertOk(await fetch(`/anonymize${qs}`, { method: 'POST', body: form }))
  return res.blob()
}
