import { useRef, useState } from "react";
import type { FramePreview } from "../api";

interface Props {
  frame: FramePreview;
  onScrub: (frameIndex: number) => void;
  onConfirm: (frameIndex: number, x: number, y: number, manualBbox?: { x: number; y: number; width: number; height: number }) => void;
  loadingFrame: boolean;
  confirming: boolean;
  error: string | null;
}

interface Point {
  x: number;
  y: number;
}

export default function TargetSelector({ frame, onScrub, onConfirm, loadingFrame, confirming, error }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [selected, setSelected] = useState<Point | null>(null);
  const [manualBbox, setManualBbox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [dragStart, setDragStart] = useState<Point | null>(null);
  const [dragCurrent, setDragCurrent] = useState<Point | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);

  function toImageCoords(clientX: number, clientY: number): Point | null {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }

  // Trouve le gobelet détecté dont le centre est le plus proche du curseur,
  // pour donner un retour visuel AVANT le clic (amélioration du ciblage).
  function nearestDetectionId(point: Point | null): string | number | null {
    if (!point || frame.detections.length === 0) return null;
    let bestId: string | number | null = null;
    let bestDist = Infinity;
    for (const d of frame.detections) {
      const cx = d.bbox.x + d.bbox.width / 2;
      const cy = d.bbox.y + d.bbox.height / 2;
      const dist = Math.hypot(cx - point.x, cy - point.y);
      if (dist < bestDist) {
        bestDist = dist;
        bestId = d.detection_id;
      }
    }
    return bestId;
  }

  const hoveredDetectionId = !isDragging ? nearestDetectionId(hoverPoint) : null;

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    const point = toImageCoords(e.clientX, e.clientY);
    if (!point) return;
    setDragStart(point);
    setDragCurrent(point);
    setIsDragging(true);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const point = toImageCoords(e.clientX, e.clientY);
    if (!point) return;
    if (isDragging) {
      setDragCurrent(point);
    } else {
      setHoverPoint(point);
    }
  }

  function handlePointerLeave() {
    setHoverPoint(null);
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (!isDragging || !dragStart) return;
    const end = toImageCoords(e.clientX, e.clientY) || dragStart;
    setIsDragging(false);

    const dx = Math.abs(end.x - dragStart.x);
    const dy = Math.abs(end.y - dragStart.y);
    const MIN_DRAG_PX = 8; // seuil en pixels image réels, sous ce seuil = simple clic

    if (dx < MIN_DRAG_PX && dy < MIN_DRAG_PX) {
      // Clic simple: mode détection automatique (comportement existant)
      setSelected(dragStart);
      setManualBbox(null);
    } else {
      // Glissement: mode sélection manuelle, on construit le rectangle
      const x = Math.min(dragStart.x, end.x);
      const y = Math.min(dragStart.y, end.y);
      const width = dx;
      const height = dy;
      setManualBbox({ x, y, width, height });
      setSelected({ x: x + width / 2, y: y + height / 2 });
    }

    setDragStart(null);
    setDragCurrent(null);
  }

  const displayScale = imgRef.current
    ? imgRef.current.getBoundingClientRect().width / (imgRef.current.naturalWidth || 1)
    : 1;

  const liveDragRect =
    isDragging && dragStart && dragCurrent
      ? {
          left: Math.min(dragStart.x, dragCurrent.x) * displayScale,
          top: Math.min(dragStart.y, dragCurrent.y) * displayScale,
          width: Math.abs(dragCurrent.x - dragStart.x) * displayScale,
          height: Math.abs(dragCurrent.y - dragStart.y) * displayScale,
        }
      : null;

  function handleConfirm() {
    if (!selected) return;
    onConfirm(frame.frame_index, selected.x, selected.y, manualBbox ?? undefined);
  }

  return (
    <div>
      <div className="viewfinder">
        <span className="vf-tl" />
        <span className="vf-br" />
        <div
          className="frame-stage"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerLeave}
          style={{ touchAction: "none" }}
        >
          <img ref={imgRef} src={`data:image/jpeg;base64,${frame.image_base64}`} alt={`Frame ${frame.frame_index}`} />

          {frame.detections.map((d) => {
            const isHovered = d.detection_id === hoveredDetectionId;
            return (
              <div
                key={d.detection_id}
                className="det-box"
                style={{
                  left: d.bbox.x * displayScale,
                  top: d.bbox.y * displayScale,
                  width: d.bbox.width * displayScale,
                  height: d.bbox.height * displayScale,
                  ...(isHovered
                    ? {
                        border: "2px solid #facc15",
                        background: "rgba(250, 204, 21, 0.18)",
                        boxShadow: "0 0 10px rgba(250, 204, 21, 0.65)",
                        transition: "all 0.12s ease",
                      }
                    : { transition: "all 0.12s ease" }),
                }}
              />
            );
          })}

          {liveDragRect && (
            <div
              className="manual-drag-box"
              style={{
                position: "absolute",
                left: liveDragRect.left,
                top: liveDragRect.top,
                width: liveDragRect.width,
                height: liveDragRect.height,
                border: "2px dashed #4ade80",
                background: "rgba(74, 222, 128, 0.15)",
                pointerEvents: "none",
              }}
            />
          )}

          {manualBbox && !isDragging && (
            <div
              className="manual-drag-box"
              style={{
                position: "absolute",
                left: manualBbox.x * displayScale,
                top: manualBbox.y * displayScale,
                width: manualBbox.width * displayScale,
                height: manualBbox.height * displayScale,
                border: "2px solid #4ade80",
                background: "rgba(74, 222, 128, 0.1)",
                pointerEvents: "none",
              }}
            />
          )}

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
        Survolez un gobelet détecté pour voir lequel sera choisi (surligné en jaune), cliquez pour le
        sélectionner, ou dessinez (glissez) un cadre pour une sélection manuelle précise.
      </p>

      {error && <div className="error-box">{error}</div>}

      <div className="actions-row">
        <button
          className="btn primary"
          disabled={!selected || confirming}
          onClick={handleConfirm}
        >
          {confirming ? "Confirmation…" : "Commencer l'analyse"}
        </button>
      </div>
    </div>
  );
}
