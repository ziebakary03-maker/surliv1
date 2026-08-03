# Sûrliv V5 — Cloud Ready

Cette version remplace SQLite par PostgreSQL pour permettre une base de données persistante et partagée entre plusieurs téléphones.

## Déploiement recommandé

Le projet est préparé pour un hébergeur Node.js compatible PostgreSQL. Le fichier `render.yaml` permet de décrire le service web et la base PostgreSQL.

Variables d'environnement :
- `DATABASE_URL` : chaîne de connexion PostgreSQL
- `PORT` : fourni automatiquement par l'hébergeur
- `NODE_ENV=production`

## Comptes de test

- Admin : `0700000000` / `admin123`
- Marchand : `0700000001` / `marchand123`
- Livreur : `0700000002` / `livreur123`
- Livreur 2 : `0700000003` / `livreur123`

⚠️ Ces comptes sont uniquement pour les tests. Change les mots de passe avant toute utilisation réelle.

## API

- `GET /api/health`
- `GET /api/bootstrap`
- `POST /api/auth/login`
- `POST /api/orders`
- `PATCH /api/orders/:id`
- `PATCH /api/livreurs/:id`
- `POST /api/sync`
- `GET /api/events` (SSE)
- `GET /api/export`

## Important avant production

Il faut encore ajouter une vraie gestion de session/token côté API, des rôles/permissions côté serveur, du stockage objet pour les photos/signatures, des notifications push, HTTPS, sauvegardes et une politique de confidentialité.
