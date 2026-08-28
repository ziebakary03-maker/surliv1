export interface UploadResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface DetectedObject {
  detection_id: number;
  bbox: { x: number; y: number; width: number; height: number };
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
  bbox: { x: number; y: number; width: number; height: number };
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
}

export interface JobResult {
  job_id: string;
  status: string;
  result_video_url: string | null;
  metrics: Record<string, unknown> | null;
}

const BASE = ""; // même origine en prod (servi par le backend) ; proxy Vite en dev

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  upload(file: File): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/api/upload`, { method: "POST", body: form }).then(json);
  },

  preview(jobId: string, frame: number): Promise<FramePreview> {
    return fetch(`${BASE}/api/jobs/${jobId}/preview/${frame}`).then(json);
  },

  selectTarget(jobId: string, frame: number, x: number, y: number): Promise<TargetSelectionResponse> {
    return fetch(`${BASE}/api/jobs/${jobId}/target`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame, x, y }),
    }).then(json);
  },

  progress(jobId: string): Promise<JobProgress> {
    return fetch(`${BASE}/api/jobs/${jobId}`).then(json);
  },

  result(jobId: string): Promise<JobResult> {
    return fetch(`${BASE}/api/jobs/${jobId}/result`).then(json);
  },

  deleteJob(jobId: string): Promise<void> {
    return fetch(`${BASE}/api/jobs/${jobId}`, { method: "DELETE" }).then(() => undefined);
  },

  videoUrl(jobId: string): string {
    return `${BASE}/api/jobs/${jobId}/video`;
  },
};
