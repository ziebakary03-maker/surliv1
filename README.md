# Object Tracker

Application web qui permet de sélectionner **un seul objet** parmi plusieurs
objets identiques (ex : trois gobelets) au début d'une vidéo, puis de suivre
son **identité** — et non simplement sa position initiale — pendant toute la
vidéo, même si les objets échangent leurs positions, accélèrent, se croisent
ou se masquent temporairement.

---

## 1. Présentation

Le problème central n'est pas "détecter trois gobelets" mais : *"une fois
que l'utilisateur a choisi le gobelet A, savoir en permanence lequel des
trois est A"*. Le pipeline combine donc :

- une **détection** par frame (objets candidats),
- un **tracking multi-objets** qui associe les détections d'une frame à
  l'autre (par mouvement + apparence, jamais par simple position),
- un **gestionnaire d'identité** qui garde le lien entre le clic initial de
  l'utilisateur et l'ID de piste correspondant,
- un **moteur de confiance** qui reflète honnêtement le niveau de certitude,
  y compris en affichant `AMBIGUOUS` quand l'information manque réellement
  (ex : objets identiques totalement masqués trop longtemps).

## 2. Architecture

```
object-tracker/
├── backend/
│   └── app/
│       ├── api/          # routes FastAPI
│       ├── models/       # schémas Pydantic
│       ├── services/     # jobs (SQLite), stockage (local/S3)
│       ├── tracking/      # détecteur, tracker, Kalman, ré-id, confiance
│       └── video/         # extraction de frames, rendu de la vidéo résultat
├── worker/                 # process séparé qui exécute le pipeline (async)
├── scripts/                # test_video.py : mode développement en CLI
├── tests/                  # tests pytest + générateur de vidéos synthétiques
├── frontend/                # React + Vite + TypeScript
├── Dockerfile / docker-compose.yml
└── fly.toml
```

### Pourquoi cette architecture de tracking

- **Détecteur** : `DETECTOR_BACKEND=opencv` par défaut — soustraction de fond
  (MOG2) + contours, sans aucun poids de modèle à télécharger, donc
  utilisable immédiatement sans accès Internet. `DETECTOR_BACKEND=yolo`
  bascule vers Ultralytics YOLO (dépendances lourdes, commentées dans
  `requirements.txt`) pour une détection sémantique bien plus robuste en
  production — recommandé pour un usage réel.
- **Tracking** : plutôt que d'importer ByteTrack/DeepSORT tels quels (qui
  supposent presque tous un détecteur YOLO + torch pour l'apparence), le
  tracker réimplémente leurs idées centrales : prédiction de Kalman +
  assignation optimale (algorithme hongrois, `scipy.optimize`) sur un coût
  combinant **distance de mouvement prédite** et **similarité d'apparence**
  (histogramme couleur HSV). C'est un choix documenté : stable, sans
  dépendance lourde, et remplaçable par un vrai ByteTrack/BoT-SORT en gardant
  la même interface (`tracker.step(frame, detections)`).
- **Identité** : `IdentityManager` ne relie jamais le target à "l'objet le
  plus proche de sa dernière position" seul — il s'appuie sur les
  `track_id` maintenus par le tracker (mouvement + apparence), et affiche un
  état honnête (`TRACKING`, `OCCLUDED`, `REIDENTIFYING`, `AMBIGUOUS`, `LOST`…)
  plutôt que de deviner silencieusement.

## 3. Installation locale (sans Docker)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd ../frontend
npm install
```

Copiez `.env.example` vers `.env` à la racine et ajustez si besoin.

### Lancer le backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Lancer le worker (dans un autre terminal)

```bash
python worker/worker.py
```

### Lancer le frontend

```bash
cd frontend
npm run dev
```

L'interface est disponible sur `http://localhost:5173` (proxy configuré vers
le backend sur `:8000`).

## 4. Lancement avec Docker (recommandé)

```bash
docker compose up --build
```

Cela démarre le serveur web (`:8000`) et le worker, avec un volume partagé
pour le stockage local de développement. Le frontend est buildé dans l'image
et servi en statique (voir Dockerfile) — vous pouvez aussi le lancer en mode
dev séparément avec `npm run dev` pendant que `docker compose` fait tourner
uniquement l'API.

## 5. Mode développement en ligne de commande

Pour tester le pipeline sans interface, sur une vidéo locale :

```bash
python scripts/test_video.py --input ma_video.mp4 --frame 30 --x 320 --y 240
```

Si vous omettez `--x`/`--y`, le script détecte automatiquement le premier
objet visible sur la frame choisie et l'utilise comme cible. Le résultat est
écrit dans `output/tracked.mp4`.

Pour générer des vidéos de test synthétiques (croisement, occlusion,
mouvement rapide) :

```bash
python tests/synthetic_video.py
```

## 6. Variables d'environnement

Voir `.env.example` pour la liste complète. Les plus importantes :

| Variable | Rôle |
|---|---|
| `MAX_VIDEO_SIZE_MB` / `MAX_VIDEO_DURATION_SECONDS` | Limites d'upload |
| `STORAGE_BACKEND` | `local` (dev) ou `s3` (production) |
| `S3_ENDPOINT` / `S3_BUCKET` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` | Stockage objet |
| `DEVICE` | `cpu` ou `cuda` |
| `DETECTOR_BACKEND` | `opencv` (par défaut) ou `yolo` |
| `MAX_OCCLUSION_FRAMES` | Durée max avant qu'une identité soit déclarée `LOST` |

## 7. API

| Endpoint | Rôle |
|---|---|
| `POST /api/upload` | Upload d'une vidéo, retourne `job_id` |
| `GET /api/jobs/{job_id}/preview/{frame}` | Image + détections d'une frame (pour la sélection UI) |
| `POST /api/jobs/{job_id}/target` | Sélectionne le target à partir de coordonnées cliquées |
| `GET /api/jobs/{job_id}` | Progression du job (frame courante, état, confiance) |
| `GET /api/jobs/{job_id}/result` | Métadonnées du résultat une fois terminé |
| `GET /api/jobs/{job_id}/video` | Téléchargement/lecture de la vidéo annotée |
| `DELETE /api/jobs/{job_id}` | Supprime le job et ses fichiers |
| `GET /health` | Health check |

Documentation interactive (Swagger) sur `http://localhost:8000/docs`.

## 8. Tests

```bash
cd backend
pytest ../tests -v
```

Les tests couvrent : filtre de Kalman (vitesse/direction), moteur de
confiance (y compris la chute de confiance après occlusion longue),
non-changement d'identité sans occlusion/croisement, préservation
d'identité à travers une occlusion, et les endpoints API (upload,
validation, cycle de vie d'un job).

**Validation réelle effectuée pendant le développement** (pas seulement du
code non testé) : le pipeline a été exécuté de bout en bout sur trois
scénarios synthétiques :

- **Occlusion** : l'identité est parfaitement préservée à travers
  l'obstacle (0 identity switch, confiance stable ~85%, état `TRACKING`).
- **Croisement** (3 objets de couleurs proches se croisant au centre) : le
  système ne devine jamais — il affiche `AMBIGUOUS` puis `LOST` quand
  l'ambiguïté devient trop forte, conformément à l'exigence d'honnêteté
  scientifique (section 11 du cahier des charges original).
- **Mouvement très rapide** : a révélé une limite réelle du détecteur
  `opencv` (voir section "Limites actuelles" ci-dessous) — gardée
  volontairement visible plutôt que masquée.

## 9. Fonctionnement du tracking (résumé)

1. Chaque frame passe par le **détecteur** → liste de bounding boxes.
2. Le **tracker** prédit la position de chaque piste existante (Kalman),
   puis résout l'association détections↔pistes par algorithme hongrois sur
   un coût mouvement + apparence.
3. Les pistes non retrouvées passent en **occlusion** (identité conservée)
   jusqu'à `MAX_OCCLUSION_FRAMES`, au-delà de quoi elles sont perdues.
4. L'**IdentityManager** ne fait le lien qu'une fois, au clic initial de
   l'utilisateur, avec le `track_id` du tracker — jamais en se basant sur
   la position à un instant donné.
5. Le **moteur de confiance** combine confiance de détection, régularité de
   trajectoire, écart prédiction/mesure, similarité d'apparence et durée
   d'occlusion pour produire un pourcentage et un niveau
   (`HIGH`/`MEDIUM`/`LOW`).

## 10. Déploiement Fly.io

```bash
fly launch --no-deploy      # utilise le fly.toml fourni
fly storage create          # stockage objet Tigris (compatible S3), recommandé
fly secrets set S3_ACCESS_KEY=... S3_SECRET_KEY=... S3_ENDPOINT=... S3_BUCKET=...
fly deploy
fly logs
fly status
```

Le `fly.toml` définit deux groupes de processus : `web` (API, léger) et
`worker` (traitement vidéo, VM plus puissante). **Ne pas** utiliser le
filesystem local d'une machine Fly.io comme stockage permanent : il n'est ni
garanti persistant, ni partagé entre `web` et `worker`. Utilisez
`STORAGE_BACKEND=s3` avec Tigris (ou tout backend compatible S3) en
production — c'est déjà la configuration par défaut de `fly.toml`.

## 11. Configuration GPU

`DEVICE=cpu` par défaut, fonctionne partout. `DEVICE=cuda` active
l'utilisation d'un GPU si `torch`/`ultralytics` sont installés (voir
`requirements.txt`) et qu'un GPU CUDA est disponible sur la machine Fly.io
choisie. Le détecteur `opencv` par défaut n'utilise pas le GPU (il n'en a
pas besoin).

## 12. Dépannage

- **`Aucun objet détecté à cet endroit`** au moment de choisir le target :
  le détecteur `opencv` a besoin de quelques frames pour "apprendre" le
  fond avant de détecter des objets de façon fiable. Essayez une frame un
  peu plus loin dans la vidéo (ex : frame 15-20) plutôt que la toute
  première image.
- **Le worker ne traite aucun job** : vérifiez que `worker/worker.py`
  tourne bien dans un process séparé (`docker compose ps` doit montrer
  `web` et `worker`), et que les deux processus partagent le même
  `LOCAL_STORAGE_PATH` ou la même config S3.
- **Vidéo résultat illisible dans le navigateur** : certains navigateurs
  sont capricieux avec le codec `mp4v` d'OpenCV. Si besoin, ré-encodez avec
  ffmpeg (déjà présent dans l'image Docker) en H.264 :
  `ffmpeg -i tracked.mp4 -c:v libx264 -pix_fmt yuv420p tracked_h264.mp4`.

---

## LIMITES ACTUELLES

- **Détecteur par défaut sans sémantique** : `OpenCVDetector` détecte des
  "blobs en mouvement différents du fond", pas des "gobelets" au sens
  sémantique. Il fonctionne bien sur fond stable avec objets de taille
  comparable, mais :
  - il peut fusionner deux objets qui se superposent complètement en un
    seul blob (cas des croisements très serrés) ;
  - sur des objets **très rapides**, l'adaptation lente du modèle de fond
    (MOG2) laisse une traînée résiduelle ("fantôme") qui peut créer une
    fausse piste concurrente pendant quelques frames — observé et non
    corrigé dans cette V1, documenté plutôt que masqué. `DETECTOR_BACKEND=yolo`
    résout ce problème en production (détection sémantique par frame, sans
    dépendance à la stabilité du fond).
  - les objets parfaitement statiques dès le début (jamais en mouvement)
    ne sont pas détectés par soustraction de fond — nécessite YOLO ou un
    détecteur de forme/couleur dédié si ce cas est fréquent chez vous.
- **Tracker "maison"** plutôt qu'une bibliothèque ByteTrack/BoT-SORT
  éprouvée : les idées centrales sont reprises (Kalman, coût combiné,
  assignation optimale), mais sans le tuning fin qu'apportent des années
  de benchmarks MOT Challenge. Voir section "Améliorations futures".
- **Trois objets réellement identiques, totalement masqués longtemps** :
  comme précisé dans le cahier des charges d'origine, aucune IA ne peut
  reconstituer une information totalement absente de la vidéo avec une
  seule caméra. Le système reflète cela honnêtement via l'état `AMBIGUOUS`
  et une confiance basse, plutôt que d'inventer une identité.
- **Stockage SQLite pour les jobs** : convient à une instance unique. Pour
  plusieurs workers/instances à forte charge, prévoir Redis + RQ/Celery
  (l'interface publique de `JobManager` resterait la même).
- **Pas de signature physique par objet ni multi-caméra** en V1 (prévus
  comme extensions, voir ci-dessous).
- **Ce projet n'a pas été testé avec un vrai poids YOLO** dans
  l'environnement de développement (pas d'accès Internet pour le
  télécharger) : le code de `YoloDetector` est écrit et prêt, mais à valider
  chez vous avec `pip install torch ultralytics` puis `DETECTOR_BACKEND=yolo`.

## AMÉLIORATIONS FUTURES

- **Signature individuelle par objet** (section 12 du cahier des charges
  d'origine) : `ReIdentifier`/`AppearanceSignature` sont déjà isolés dans
  leur propre module — il suffirait d'y ajouter une extraction de marqueur
  discret (motif imprimé, tag QR miniature, marquage IR invisible à l'œil
  nu) en complément de l'histogramme couleur, pour lever l'ambiguïté même
  sur des objets visuellement identiques.
- **Multi-caméra** : le module `MultiCameraFusion` prévu dans le cahier des
  charges n'est pas implémenté en V1, mais l'architecture s'y prête :
  chaque caméra pourrait faire tourner son propre `MultiObjectTracker`, et
  un module de fusion combinerait les `AppearanceSignature` + positions 3D
  reconstruites pour maintenir l'identité même si le target est masqué
  dans une caméra mais visible dans une autre. Nécessiterait calibration
  et synchronisation temporelle des flux.
- **Remplacer le tracker "maison" par BoT-SORT/ByteTrack officiels** une
  fois `torch`/`ultralytics` disponibles, pour bénéficier de leur
  ré-identification par réseau de neurones (bien plus robuste que
  l'histogramme couleur sur des objets très similaires).
- **Ré-encodage systématique H.264** en sortie (ffmpeg) plutôt que le codec
  `mp4v` d'OpenCV, pour une compatibilité navigateur garantie sans étape
  manuelle.
- **File de jobs Redis/RQ** pour supporter plusieurs workers en parallèle
  à grande échelle, au lieu de SQLite.
