export interface UploadResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface DetectedObject {
  detection_id: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
}

export interface FramePreview {
  job_id: string;
  frame_index: number;
  fps: number;
  total_frames: number;
  detections: DetectedObject[];
  image_base64: string;
}

export interface TargetSelectionResponse {
  job_id: string;
  target_id: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  status: string;
}

export interface JobProgress {
  job_id: string;
  status: string;
  current_frame: number;
  total_frames: number;
  progress_percent: number;

  target_id: number | null;
  target_state: string | null;

  confidence_percent: number | null;
  confidence_level: "HIGH" | "MEDIUM" | "LOW" | null;

  identity_switches: number;

  error: string | null;

  /*
   * Nouveau système Cup Lock
   * -------------------------
   * Le suivi porte maintenant sur le gobelet
   * contenant la boule plutôt que sur la boule
   * lorsqu'elle est cachée.
   */
  container_id: number | null;
  container_confidence: number | null;
  container_label: string | null;

  /*
   * Indique si la boule est actuellement
   * visible dans l'image.
   */
  ball_visible: boolean | null;
}

export interface JobResult {
  job_id: string;
  status: string;
  result_video_url: string | null;
  metrics: Record<string, unknown> | null;
}

const BASE = "";
// Même origine en production.
// Le proxy Vite est utilisé en développement.

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }

  return (await res.json()) as T;
}

export const api = {
  /*
   * Upload d'une vidéo
   */
  upload(file: File): Promise<UploadResponse> {
    const form = new FormData();

    form.append("file", file);

    return fetch(`${BASE}/api/upload`, {
      method: "POST",
      body: form,
    }).then((res) => json<UploadResponse>(res));
  },

  /*
   * Récupération d'une frame de prévisualisation.
   */
  preview(
    jobId: string,
    frame: number
  ): Promise<FramePreview> {
    return fetch(
      `${BASE}/api/jobs/${jobId}/preview/${frame}`
    ).then((res) => json<FramePreview>(res));
  },

  /*
   * Sélection de la boule par l'utilisateur.
   *
   * Le backend détermine ensuite automatiquement
   * le gobelet qui contient la boule et le verrouille
   * comme cible permanente (ex: Cup_A).
   */
  selectTarget(
    jobId: string,
    frame: number,
    x: number,
    y: number
  ): Promise<TargetSelectionResponse> {
    return fetch(`${BASE}/api/jobs/${jobId}/target`, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        frame,
        x,
        y,
      }),
    }).then((res) =>
      json<TargetSelectionResponse>(res)
    );
  },

  /*
   * État et progression du traitement.
   *
   * container_label permet notamment au frontend
   * d'afficher Cup_A, Cup_B ou Cup_C.
   */
  progress(jobId: string): Promise<JobProgress> {
    return fetch(
      `${BASE}/api/jobs/${jobId}`
    ).then((res) => json<JobProgress>(res));
  },

  /*
   * Résultat final du traitement.
   */
  result(jobId: string): Promise<JobResult> {
    return fetch(
      `${BASE}/api/jobs/${jobId}/result`
    ).then((res) => json<JobResult>(res));
  },

  /*
   * Suppression d'un job.
   */
  deleteJob(jobId: string): Promise<void> {
    return fetch(
      `${BASE}/api/jobs/${jobId}`,
      {
        method: "DELETE",
      }
    ).then(() => undefined);
  },

  /*
   * URL de la vidéo finale annotée.
   */
  videoUrl(jobId: string): string {
    return `${BASE}/api/jobs/${jobId}/video`;
  },
};
