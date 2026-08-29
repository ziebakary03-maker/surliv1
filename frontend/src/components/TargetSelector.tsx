@router.post("/api/jobs/{job_id}/target", response_model=TargetSelectionResponse)
def select_target(job_id: str, req: TargetSelectionRequest):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")

    local_path = storage.path_for_read(job["input_key"])
    detector = build_detector()

    # Chauffe le détecteur en rejouant les frames précédentes
    cap = cv2.VideoCapture(local_path)
    detections = []
    for i in range(req.frame + 1):
        ok, frame_img = cap.read()
        if not ok:
            break
        detections = detector.detect(frame_img)
    cap.release()

    if not detections:
        raise HTTPException(400, "Aucun objet détecté à cet endroit de la frame.")

    best = None
    best_dist = float("inf")
    for det in detections:
        cx = det.bbox.x + det.bbox.width / 2
        cy = det.bbox.y + det.bbox.height / 2
        dist = ((cx - req.x) ** 2 + (cy - req.y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = det

    if best is None:
        raise HTTPException(400, "Aucun objet détecté à cet endroit de la frame.")

    job_manager.set_target(job_id, req.frame, req.x, req.y)

    return TargetSelectionResponse(
        job_id=job_id, target_id=1, bbox=best.bbox, status=JobStatus.QUEUED
    )
