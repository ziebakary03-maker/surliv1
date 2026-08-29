import { useRef, useState } from "react";
import type { FramePreview } from "../api";

interface Props {
  frame: FramePreview;
  onScrub: (frameIndex: number) => void;
  onConfirm: (frameIndex: number, x: number, y: number) => void;
  loadingFrame: boolean;
  confirming: boolean;
  error: string | null;
}

export default function TargetSelector({ frame, onScrub, onConfirm, loadingFrame, confirming, error }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [selected, setSelected] = useState<{ x: number; y: number } | null>(null);

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    const img = imgRef.current;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setSelected({ x, y });
  }

  const displayScale = imgRef.current
    ? imgRef.current.getBoundingClientRect().width / (imgRef.current.naturalWidth || 1)
    : 1;

  return (
    <div>
      <div className="viewfinder">
        <span className="vf-tl" />
        <span className="vf-br" />
        <div className="frame-stage" onClick={handleClick}>
          <img ref={imgRef} src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Frame ${frame.frame_index}`} />
          {frame.detections.map((d) => (
            <div
              key={d.detection_id}
              className="det-box"
              style={{
                left: d.bbox.x * displayScale,
                top: d.bbox.y * displayScale,
                width: d.bbox.width * displayScale,
                height: d.bbox.height * displayScale,
              }}
            />
          ))}
          {selected && (
            <div
              className="target-pin"
              style={{ left: selected.x * displayScale, top: selected.y * displayScale - 8 }}
            >
              📌
              <br />
              TARGET #1
            </div>
          )}
        </div>
      </div>

      <input
        type="range"
        className="frame-scrubber"
        min={0}
        max={Math.max(frame.total_frames - 1, 0)}
        value={frame.frame_index}
        onChange={(e) => onScrub(Number(e.target.value))}
        disabled={loadingFrame}
      />
      <div className="label-row">
        <span>FRAME {frame.frame_index} / {frame.total_frames}</span>
        <span>{frame.fps.toFixed(0)} FPS</span>
      </div>

      <p className="hint" style={{ marginTop: 16 }}>
        Cliquez sur l'un des objets détectés (encadrés) pour le choisir comme cible à suivre.
      </p>

      {error && <div className="error-box">{error}</div>}

      <div className="actions-row">
        <button
          className="btn primary"
          disabled={!selected || confirming}
          onClick={() => selected && onConfirm(frame.frame_index, selected.x, selected.y)}
        >
          {confirming ? "Confirmation…" : "Commencer l'analyse"}
        </button>
      </div>
    
