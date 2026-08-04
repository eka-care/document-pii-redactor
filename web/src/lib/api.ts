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

export async function detect(file: File, exclude?: string[]): Promise<PIIEntity[]> {
  const form = new FormData()
  form.append('file', file)
  const qs = exclude?.length ? `?exclude=${encodeURIComponent(exclude.join(','))}` : ''
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
  exclude?: string[],
): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)
  form.append('color', rgbToHex(color))
  const qs = exclude?.length ? `?exclude=${encodeURIComponent(exclude.join(','))}` : ''
  const res = await assertOk(await fetch(`/redact${qs}`, { method: 'POST', body: form }))
  return res.blob()
}

export async function detectText(text: string, exclude?: string[]): Promise<TextPIISpan[]> {
  const res = await assertOk(
    await fetch('/detect-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, exclude }),
    }),
  )
  const data = await res.json()
  return data.spans as TextPIISpan[]
}
