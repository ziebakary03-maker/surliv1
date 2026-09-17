import type { JobResult } from "../api";

interface Props {
  result: JobResult;
  videoUrl: string;
  onReset: () => void;
}

export default function Result({ result, videoUrl, onReset }: Props) {
  const metrics = result.metrics ?? {};

  return (
    <div>
      <div className="viewfinder">
        <span className="vf-tl" />
        <span className="vf-br" />
        <video className="result-video" src={videoUrl} controls autoPlay loop />
      </div>

      <div className="stat-grid">
        <div className="stat">
          <div className="stat-label">Identity switches</div>
          <div className="stat-value">{String(metrics.identity_switches ?? "—")}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Confidence finale</div>
          <div className="stat-value">
            {metrics.final_confidence != null ? `${Number(metrics.final_confidence).toFixed(1)}%` : "—"}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">État final</div>
          <div className="stat-value">
            {metrics.final_state ? (
              <span className={`state-badge state-${metrics.final_state}`}>{String(metrics.final_state)}</span>
            ) : (
              "—"
            )}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Frames traitées</div>
          <div className="stat-value">{String(metrics.total_frames ?? "—")}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Conteneur final</div>
          <div className="stat-value">
            {metrics.final_container_id != null ? `CUP #${metrics.final_container_id}` : "—"}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Frames ambiguës</div>
          <div className="stat-value">{String(metrics.ambiguous_frames ?? "—")}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Ré-identifications</div>
          <div className="stat-value">{String(metrics.reidentification_events ?? "—")}</div>
        </div>
      </div>

      {metrics.semantic_mode === false && (
        <p className="mono" style={{ fontSize: "0.8rem", color: "var(--text-dim)", marginTop: "8px" }}>
          Mode sémantique ball/cup indisponible pour cette analyse (détecteur sans compréhension
          sémantique) : les objets ont été différenciés uniquement par mouvement/apparence/position.
        </p>
      )}

      <div className="actions-row">
        <a className="btn primary" href={videoUrl} download="tracked_result.mp4">
          Télécharger la vidéo
        </a>
        <button className="btn ghost" onClick={onReset}>
          Analyser une autre vidéo
        </button>
      </div>
    </div>
  );
}
