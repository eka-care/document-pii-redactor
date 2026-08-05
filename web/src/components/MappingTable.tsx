import type { PseudonymMapping } from '../lib/api'

// The entity -> pseudonym table returned by de-identification. Shown so demo
// users understand the mapping is real and theirs to store; the server never
// keeps it.
export default function MappingTable({ mapping }: { mapping: PseudonymMapping }) {
  const rows = Object.entries(mapping.entries).flatMap(([label, byPseudonym]) =>
    Object.entries(byPseudonym).map(([pseudonym, original]) => ({ label, pseudonym, original })),
  )
  if (rows.length === 0) return null
  return (
    <section className="card">
      <h2 className="card-title">Pseudonym mapping</h2>
      <p className="tab-caption">
        Returned to the caller for secure storage — this is what makes de-identification
        reversible by an authorized party. The server does not keep it.
      </p>
      <div className="table-scroll">
        <table className="entity-table">
          <thead>
            <tr>
              <th>group</th>
              <th>pseudonym</th>
              <th>original</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.label}-${r.pseudonym}`}>
                <td>{r.label}</td>
                <td>{r.pseudonym}</td>
                <td>{r.original}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
