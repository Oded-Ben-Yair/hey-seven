#!/usr/bin/env bash
# Deployment script for Casino Host Agent to GCP Cloud Run.
#
# Usage:
#   ./gcp/deploy.sh                     # Deploy with defaults
#   ./gcp/deploy.sh --project my-proj   # Override project ID
#   ./gcp/deploy.sh --dry-run           # Print commands without executing
#
# Prerequisites:
#   - gcloud CLI authenticated (`gcloud auth login`)
#   - Docker installed and running
#   - Artifact Registry repository created
#   - Secret Manager secrets configured:
#       google-api-key  (Gemini API key)
#       api-key         (Application API key for request auth)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-hey-seven-prod}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-casino-host-agent}"
REPOSITORY="${REPOSITORY:-hey-seven}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT_ID="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --service) SERVICE_NAME="$2"; shift 2 ;;
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "====================================="
echo "Casino Host Agent -- Deployment"
echo "====================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_URI}"
echo "Dry Run:  ${DRY_RUN}"
echo "====================================="

run_cmd() {
    echo "[CMD] $*"
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Set GCP project
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Configure GCP project ---"
run_cmd gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Step 2: Ensure Artifact Registry repository exists
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Ensure Artifact Registry repository ---"
run_cmd gcloud artifacts repositories describe "${REPOSITORY}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || \
run_cmd gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="Hey Seven Docker images"

# Configure Docker for Artifact Registry
run_cmd gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------------------------------------------------------------------------
# Step 3: Build Docker image
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Build Docker image ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

run_cmd docker build \
    -t "${IMAGE_URI}" \
    -f "${PROJECT_ROOT}/gcp/Dockerfile" \
    "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Step 4: Push to Artifact Registry
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Push to Artifact Registry ---"
run_cmd docker push "${IMAGE_URI}"

# ---------------------------------------------------------------------------
# Step 5: Deploy to Cloud Run (IAM auth -- no --allow-unauthenticated)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Deploy to Cloud Run ---"
run_cmd gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --platform=managed \
    --no-allow-unauthenticated \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300s \
    --concurrency=80 \
    --set-env-vars="APP_VERSION=${IMAGE_TAG},USE_FIRESTORE=true,GCP_PROJECT_ID=${PROJECT_ID},ENVIRONMENT=production" \
    --set-secrets="GOOGLE_API_KEY=google-api-key:latest,API_KEY=api-key:latest"

# ---------------------------------------------------------------------------
# Step 6: Verify deployment
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: Verify deployment ---"
if [ "$DRY_RUN" = false ]; then
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region="${REGION}" \
        --format="value(status.url)")

    echo "Service URL: ${SERVICE_URL}"
    echo ""
    echo "NOTE: Service requires IAM authentication."
    echo "  To invoke, use an identity token:"
    echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
    echo "       -H \"X-API-Key: <YOUR_API_KEY>\" \\"
    echo "       ${SERVICE_URL}/api/v1/health"
    echo ""
    echo "Waiting 15 seconds for cold start..."
    sleep 15

    echo "Health check (via identity token):"
    TOKEN=$(gcloud auth print-identity-token 2>/dev/null || echo "")
    if [ -n "${TOKEN}" ]; then
        curl -s -H "Authorization: Bearer ${TOKEN}" \
            "${SERVICE_URL}/health" | python3 -m json.tool || echo "(health check failed)"
    else
        echo "(Could not obtain identity token -- verify manually)"
    fi

    echo ""
    echo "Deployment complete."
else
    echo "[DRY RUN] Would deploy and verify at Cloud Run URL."
fi
