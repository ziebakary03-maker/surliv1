import { useEffect, useRef, useState } from "react";
import { api, type FramePreview, type JobProgress, type JobResult } from "./api";
import Upload from "./components/Upload";
import TargetSelector from "./components/TargetSelector";
import Processing from "./components/Processing";
import Result from "./components/Result";

type Stage = "upload" | "select_target" | "processing" | "result" | "error";

export default function App() {
  const [stage, setStage] = useState<Stage>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [frame, setFrame] = useState<FramePreview | null>(null);
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<number | null>(null);

  function reset() {
    if (pollRef.current) window.clearInterval(pollRef.current);
    setStage("upload");
    setJobId(null);
    setFrame(null);
    setProgress(null);
    setResult(null);
    setError(null);
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const res = await api.upload(file);
      setJobId(res.job_id);
      const firstFrame = await api.preview(res.job_id, 0);
      setFrame(firstFrame);
      setStage("select_target");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Échec de l'upload.");
    } finally {
      setUploading(false);
    }
  }

  async function handleScrub(frameIndex: number) {
    if (!jobId) return;
    setLoadingFrame(true);
    try {
      const f = await api.preview(jobId, frameIndex);
      setFrame(f);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Impossible de charger cette frame.");
    } finally {
      setLoadingFrame(false);
    }
  }

  async function handleConfirmTarget(frameIndex: number, x: number, y: number) {
    if (!jobId) return;
    setConfirming(true);
    setError(null);
    try {
      await api.selectTarget(jobId, frameIndex, x, y);
      setStage("processing");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Aucun objet détecté à cet endroit, réessayez.");
    } finally {
      setConfirming(false);
    }
  }

  useEffect(() => {
    if (stage !== "processing" || !jobId) return;

    pollRef.current = window.setInterval(async () => {
      try {
        const p = await api.progress(jobId);
        setProgress(p);
        if (p.status === "completed") {
          window.clearInterval(pollRef.current!);
          const r = await api.result(jobId);
          setResult(r);
          setStage("result");
        } else if (p.status === "failed") {
          window.clearInterval(pollRef.current!);
          setError(p.error || "Le traitement a échoué.");
          setStage("error");
        }
      } catch {
        // erreur réseau transitoire : on continue de sonder
      }
    }, 1500);

    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [stage, jobId]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <span className="reticle">◎</span> OBJECT TRACKER
        </div>
        <div className="app-tagline">Suivez un objet précis dans une vidéo</div>
      </header>

      <div className="panel">
        {stage === "upload" && <Upload onUpload={handleUpload} busy={uploading} error={error} />}

        {stage === "select_target" && frame && (
          <TargetSelector
            frame={frame}
            onScrub={handleScrub}
            onConfirm={handleConfirmTarget}
            loadingFrame={loadingFrame}
            confirming={confirming}
            error={error}
          />
        )}

        {stage === "processing" && progress && <Processing progress={progress} />}
        {stage === "processing" && !progress && (
          <div className="mono hint">Démarrage de l'analyse…</div>
        )}

        {stage === "result" && result && jobId && (
          <Result result={result} videoUrl={api.videoUrl(jobId)} onReset={reset} />
        )}

        {stage === "error" && (
          <div>
            <div className="error-box">{error}</div>
            <div className="actions-row">
              <button className="btn ghost" onClick={reset}>
                Recommencer
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
