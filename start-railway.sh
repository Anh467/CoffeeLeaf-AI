#!/usr/bin/env bash
set -euo pipefail

if [ -z "${DAGSHUB_USER_TOKEN:-}" ]; then
  echo "ERROR: DAGSHUB_USER_TOKEN is not set"
  exit 1
fi

echo "Configuring DVC remote credentials..."
dvc remote modify dagshub-s3 --local access_key_id "$DAGSHUB_USER_TOKEN"
dvc remote modify dagshub-s3 --local secret_access_key "$DAGSHUB_USER_TOKEN"

echo "Pulling deployment bundle from DagsHub..."
dvc pull deployment -r dagshub-s3

ls -lh deployment

echo "Starting FastAPI on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
