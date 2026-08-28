import { useRef, useState } from "react";

interface Props {
  onUpload: (file: File) => void;
  busy: boolean;
  error: string | null;
}

const ACCEPTED = [".mp4", ".mov", ".webm"];

export default function Upload({ onUpload, busy, error }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    onUpload(files[0]);
  }

  return (
    <div
      className={`upload-zone${dragging ? " dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div style={{ fontSize: "1.05rem", marginBottom: 10 }}>
        {busy ? "Envoi en cours…" : "Glissez une vidéo ici, ou cliquez pour parcourir"}
      </div>
      <div className="mono hint">MP4 · MOV · WEBM</div>
      {error && <div className="error-box">{error}</div>}
    </div>
  );
}
