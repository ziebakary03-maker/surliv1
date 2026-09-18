#!/bin/bash
# Configuration de l'environnement de dev local (Codespace / machine locale).
# Usage : source setup_dev_env.sh   (avec "source", pas "./", sinon les
# variables exportées ne s'appliquent pas au shell courant)
#
# Ne contient AUCUN secret de production : uniquement des identifiants
# de développement local, jetables, valables seulement sur cette machine.

set -e

# --- Base de données locale (Postgres) ---
export DATABASE_URL="${DATABASE_URL:-postgresql://surliv:surliv@localhost:5432/surliv_test}"

# --- Stockage local des vidéos/résultats ---
export LOCAL_STORAGE_PATH="${LOCAL_STORAGE_PATH:-/tmp/surliv-storage}"
mkdir -p "$LOCAL_STORAGE_PATH"

# --- Démarre Postgres s'il ne tourne pas déjà ---
if ! pg_isready -q 2>/dev/null; then
  echo "Démarrage de PostgreSQL..."
  sudo pg_ctlcluster 16 main start 2>/dev/null || sudo service postgresql start 2>/dev/null || true
  sleep 1
fi

# --- Crée le rôle et la base de dev s'ils n'existent pas encore ---
if ! sudo su - postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='surliv'\"" 2>/dev/null | grep -q 1; then
  echo "Création du rôle 'surliv'..."
  sudo su - postgres -c "psql -c \"CREATE USER surliv WITH PASSWORD 'surliv' SUPERUSER;\""
fi
if ! sudo su - postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='surliv_test'\"" 2>/dev/null | grep -q 1; then
  echo "Création de la base 'surliv_test'..."
  sudo su - postgres -c "psql -c \"CREATE DATABASE surliv_test OWNER surliv;\""
fi

echo "Environnement de dev prêt."
echo "  DATABASE_URL=$DATABASE_URL"
echo "  LOCAL_STORAGE_PATH=$LOCAL_STORAGE_PATH"
