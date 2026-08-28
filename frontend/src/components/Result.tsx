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
      </div>

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
