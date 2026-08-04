// A muted "stamp ink" palette — one distinct ink per PII category group.
// Deliberately desaturated relative to default chart colors so the set reads
// as a designed classification system (like a records office's stamps)
// rather than a rainbow. Same category keys as the model's taxonomy.
export const L1_COLORS: Record<string, [number, number, number]> = {
  person: [161, 61, 61], // oxblood
  location: [43, 91, 140], // denim
  date_time: [150, 99, 28], // ochre
  contact: [30, 107, 82], // pine
  uid: [106, 63, 140], // plum
  device_net: [32, 108, 116], // teal
  credential: [140, 47, 73], // crimson
  entity: [85, 86, 92], // graphite
  biometric_visual: [161, 61, 116], // rose
  unknown: [107, 106, 99], // stone
}

export function l1Rgb(group: string): string {
  const [r, g, b] = L1_COLORS[group] ?? L1_COLORS.unknown
  return `rgb(${r}, ${g}, ${b})`
}

// Light tint of the same color, for subtle highlight backgrounds (dark text
// stays readable on top, unlike the solid chip fill used elsewhere).
export function l1Tint(group: string, alpha = 0.14): string {
  const [r, g, b] = L1_COLORS[group] ?? L1_COLORS.unknown
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
