import type { JobProgress } from "../api";

interface Props {
  progress: JobProgress;
}

export default function Processing({ progress }: Props) {
  const pct = Math.min(100, Math.max(0, progress.progress_percent));

  return (
    <div className="viewfinder">
      <span className="vf-tl" />
      <span className="vf-br" />
      <div style={{ padding: "20px 6px" }}>
        <div className="mono" style={{ fontSize: "0.85rem", color: "var(--text-dim)" }}>
          ANALYSE DE LA VIDÉO…
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="label-row" style={{ marginTop: 0 }}>
          <span>{pct.toFixed(1)}%</span>
          <span>FRAME {progress.current_frame} / {progress.total_frames}</span>
        </div>

        <div className="stat-grid">
          <div className="stat">
            <div className="stat-label">Target</div>
            <div className="stat-value">#{progress.target_id ?? "—"}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Confidence</div>
            <div className="stat-value">
              {progress.confidence_percent != null ? `${progress.confidence_percent.toFixed(1)}%` : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Status</div>
            <div className="stat-value">
              {progress.target_state ? (
                <span className={`state-badge state-${progress.target_state}`}>{progress.target_state}</span>
              ) : (
                "—"
              )}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Identity switches</div>
            <div className="stat-value">{progress.identity_switches}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
