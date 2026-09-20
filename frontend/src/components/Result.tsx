import type { JobResult } from "../api";

interface Props {
  result: JobResult;
  videoUrl: string;
  onReset: () => void;
}

const POSITION_LABELS: Record<string, string> = { LEFT: "Gauche", MIDDLE: "Milieu", RIGHT: "Droite" };
const POSITIONS: Array<"LEFT" | "MIDDLE" | "RIGHT"> = ["LEFT", "MIDDLE", "RIGHT"];

export default function Result({ result, videoUrl, onReset }: Props) {
  const metrics = result.metrics ?? {};
  const finalPosition = metrics.final_position as string | undefined;
  const displayState = (metrics.display_state ?? metrics.final_state) as string | undefined;
  const revealed = displayState === "FOUND";

  return (
    <div>
      <div className="viewfinder">
        <span className="vf-tl" />
        <span className="vf-br" />
        <video className="result-video" src={videoUrl} controls autoPlay loop />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 16,
          margin: "20px 0",
          flexWrap: "wrap",
        }}
      >
        {POSITIONS.map((pos) => {
          const isWinner = revealed && finalPosition === pos;
          return (
            <div
              key={pos}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "14px 22px",
                minWidth: 84,
                borderRadius: 12,
                border: isWinner ? "2px solid #4ade80" : "1px solid rgba(255,255,255,0.15)",
                background: isWinner ? "rgba(74, 222, 128, 0.15)" : "transparent",
                boxShadow: isWinner ? "0 0 18px rgba(74, 222, 128, 0.55)" : "none",
                transition: "all 0.3s ease",
              }}
            >
              <div style={{ fontSize: 34, lineHeight: 1 }}>{isWinner ? "🎯" : "🥤"}</div>
              <div
                style={{
                  marginTop: 8,
                  fontWeight: isWinner ? 700 : 400,
                  color: isWinner ? "#4ade80" : "inherit",
                  fontSize: "0.9rem",
                }}
              >
                {POSITION_LABELS[pos]}
              </div>
            </div>
          );
        })}
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
            {displayState ? (
              <span className={`state-badge state-${displayState}`}>{String(displayState)}</span>
            ) : (
              "—"
            )}
          </div>
        </div>
        <div className="stat">
          <div className="stat-label">Position</div>
          <div className="stat-value">{finalPosition ? POSITION_LABELS[finalPosition] ?? finalPosition : "—"}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Frames traitées</div>
          <div className="stat-value">{String(metrics.total_frames ?? "—")}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Conteneur final</div>
          <div className="stat-value">
            {metrics.final_container_label
              ? String(metrics.final_container_label)
              : metrics.final_container_id != null
              ? `CUP #${metrics.final_container_id}`
              : "—"}
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
