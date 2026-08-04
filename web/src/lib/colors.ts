// Ported from streamlit_app.py's L1_COLORS so the React demo and the
// Streamlit tester stay visually consistent.
export const L1_COLORS: Record<string, [number, number, number]> = {
  person: [220, 38, 38],
  location: [37, 99, 235],
  date_time: [217, 119, 6],
  contact: [5, 150, 105],
  uid: [124, 58, 237],
  device_net: [8, 145, 178],
  credential: [190, 18, 60],
  entity: [100, 116, 139],
  biometric_visual: [219, 39, 119],
  unknown: [75, 85, 99],
}

export function l1Rgb(group: string): string {
  const [r, g, b] = L1_COLORS[group] ?? L1_COLORS.unknown
  return `rgb(${r}, ${g}, ${b})`
}
