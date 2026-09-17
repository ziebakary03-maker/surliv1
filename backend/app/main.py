"""
Point d'entrée FastAPI. Sert uniquement l'API + orchestration des jobs.
Le traitement lourd (pipeline vidéo) tourne dans le process `worker`
(voir worker/worker.py), jamais dans une requête HTTP (section 18).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Suivi d'identité d'un objet précis parmi plusieurs objets identiques dans une vidéo.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok", "app": settings.APP_NAME, "version": "1.0.0"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production au domaine du frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

