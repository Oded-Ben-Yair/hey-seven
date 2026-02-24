#!/usr/bin/env bash
# Secret rotation script for Hey Seven GCP Secret Manager secrets.
#
# Usage: ./scripts/rotate-secret.sh <secret-name> <new-value>
# Example: ./scripts/rotate-secret.sh hey-seven-api-key "new-key-value"
#
# Steps:
#   1. Create new secret version
#   2. Update Cloud Run to use new version
#   3. Verify health endpoint (3 retries)
#   4. Disable old versions (keep for rollback, never delete)
#
# Emergency rollback:
#   gcloud secrets versions enable <secret-name> --version=<old-version>

set -euo pipefail

SECRET_NAME="${1:?Usage: $0 <secret-name> <new-value>}"
NEW_VALUE="${2:?Usage: $0 <secret-name> <new-value>}"
SERVICE="hey-seven"
REGION="us-central1"

echo "=== Secret Rotation: $SECRET_NAME ==="

# Step 1: Create new secret version
echo "Creating new secret version..."
echo -n "$NEW_VALUE" | gcloud secrets versions add "$SECRET_NAME" --data-file=-
NEW_VERSION=$(gcloud secrets versions list "$SECRET_NAME" --limit=1 --format='value(name)')
echo "New version: $NEW_VERSION"

# Step 2: Update Cloud Run to use new version
echo "Updating Cloud Run service..."
gcloud run services update "$SERVICE" --region="$REGION" \
  --update-secrets="${SECRET_NAME}=${SECRET_NAME}:${NEW_VERSION}"

# Step 3: Verify health (3 retries with 10s intervals)
echo "Waiting 30s for container restart..."
sleep 30
SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')
HEALTH_STATUS=""
for i in 1 2 3; do
  HEALTH_STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$SERVICE_URL/health" --max-time 15)
  echo "Health check attempt $i: HTTP $HEALTH_STATUS"
  if [ "$HEALTH_STATUS" = "200" ]; then
    break
  fi
  sleep 10
done
if [ "$HEALTH_STATUS" != "200" ]; then
  echo "HEALTH CHECK FAILED after 3 attempts (HTTP $HEALTH_STATUS)."
  echo "Emergency rollback: gcloud secrets versions enable $SECRET_NAME --version=<old-version>"
  exit 1
fi
echo "Health check passed (HTTP 200)"

# Step 4: Disable old versions (keep for rollback, never delete)
OLD_VERSIONS=$(gcloud secrets versions list "$SECRET_NAME" --filter="state=enabled AND name!=$NEW_VERSION" --format='value(name)')
for ver in $OLD_VERSIONS; do
  echo "Disabling old version: $ver"
  gcloud secrets versions disable "$ver" --secret="$SECRET_NAME" 2>/dev/null || true
done

echo "=== Rotation complete: $SECRET_NAME -> version $NEW_VERSION ==="
