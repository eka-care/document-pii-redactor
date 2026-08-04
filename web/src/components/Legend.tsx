import { l1Rgb } from '../lib/colors'

export default function Legend({ groups }: { groups: string[] }) {
  if (groups.length === 0) return null
  return (
    <div className="legend">
      {groups.map((g) => (
        <span key={g} className="legend-chip" style={{ background: l1Rgb(g) }}>
          {g}
        </span>
      ))}
    </div>
  )
}
