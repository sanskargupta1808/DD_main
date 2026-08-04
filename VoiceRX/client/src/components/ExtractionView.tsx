import type { MedicalExtraction } from "../types";

function Chips({ items, kind }: { items: string[]; kind: string }) {
  if (items.length === 0) return <p className="empty">None detected.</p>;
  return (
    <div className="chips">
      {items.map((item, i) => (
        <span key={`${kind}-${i}`} className={`chip chip-${kind}`}>
          {item}
        </span>
      ))}
    </div>
  );
}

export function ExtractionView({ data }: { data: MedicalExtraction }) {
  const { patient, prescriptions, followUp } = data;
  const hasPatient = patient.name || patient.age || patient.gender;

  return (
    <div className="extraction">
      <section className="card">
        <h3>🧑‍⚕️ Patient</h3>
        {hasPatient ? (
          <ul className="kv">
            {patient.name && <li><span>Name</span><b>{patient.name}</b></li>}
            {patient.age && <li><span>Age</span><b>{patient.age}</b></li>}
            {patient.gender && <li><span>Gender</span><b>{patient.gender}</b></li>}
          </ul>
        ) : (
          <p className="empty">No patient details detected.</p>
        )}
      </section>

      <section className="card">
        <h3>🤒 Symptoms</h3>
        <Chips items={data.symptoms} kind="symptom" />
      </section>

      <section className="card">
        <h3>🩺 Diagnoses</h3>
        <Chips items={data.diagnoses} kind="diagnosis" />
      </section>

      <section className="card card-wide">
        <h3>💊 Prescriptions</h3>
        {prescriptions.length === 0 ? (
          <p className="empty">None detected.</p>
        ) : (
          <table className="rx-table">
            <thead>
              <tr>
                <th>Medication</th>
                <th>Dosage</th>
                <th>Frequency / Timing</th>
                <th>Duration</th>
                <th>Instructions</th>
              </tr>
            </thead>
            <tbody>
              {prescriptions.map((p, i) => (
                <tr key={`rx-${i}`}>
                  <td><b>{p.medication}</b></td>
                  <td>{p.dosage ?? "—"}</td>
                  <td>{p.frequency ?? "—"}</td>
                  <td>{p.duration ?? "—"}</td>
                  <td>{p.instructions ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h3>📅 Follow-up</h3>
        {followUp.nextVisit || followUp.instructions ? (
          <ul className="kv">
            {followUp.nextVisit && <li><span>Next visit</span><b>{followUp.nextVisit}</b></li>}
            {followUp.instructions && <li><span>Note</span><b>{followUp.instructions}</b></li>}
          </ul>
        ) : (
          <p className="empty">No follow-up detected.</p>
        )}
      </section>

      <section className="card">
        <h3>⚠️ Allergies</h3>
        <Chips items={data.allergies} kind="allergy" />
      </section>

      <section className="card">
        <h3>📈 Vitals</h3>
        <Chips items={data.vitals} kind="vital" />
      </section>

      {data.notes.length > 0 && (
        <section className="card card-wide">
          <h3>📝 Notes</h3>
          <ul className="notes">
            {data.notes.map((n, i) => (
              <li key={`note-${i}`}>{n}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
