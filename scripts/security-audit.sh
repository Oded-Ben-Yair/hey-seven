#!/usr/bin/env bash
# Security audit for Python dependencies.
# Run in CI/CD pipeline or locally before release.
#
# Prerequisites:
#   pip install pip-audit
#
# Usage:
#   ./scripts/security-audit.sh [requirements-file]
#
# Exit codes:
#   0 = no vulnerabilities found
#   1 = vulnerabilities found or tool error

set -euo pipefail

REQ_FILE="${1:-requirements.txt}"

echo "=== Hey Seven Security Audit ==="
echo "Checking: ${REQ_FILE}"
echo ""

# 1. pip-audit: Check for known vulnerabilities (CVEs)
echo "--- Step 1: pip-audit (CVE check) ---"
if command -v pip-audit &>/dev/null; then
    pip-audit -r "${REQ_FILE}" --strict --desc || {
        echo "FAIL: Vulnerabilities found. Fix before deploying."
        exit 1
    }
    echo "PASS: No known vulnerabilities."
else
    echo "SKIP: pip-audit not installed (pip install pip-audit)"
fi

echo ""

# 2. Check for unpinned dependencies
echo "--- Step 2: Pin verification ---"
UNPINNED=$(grep -E '^\s*[a-zA-Z]' "${REQ_FILE}" | grep -v '==' | grep -v '^#' | grep -v '^-' || true)
if [ -n "${UNPINNED}" ]; then
    echo "WARN: Unpinned dependencies found:"
    echo "${UNPINNED}"
else
    echo "PASS: All dependencies pinned with =="
fi

echo ""

# 3. Check for --require-hashes in prod requirements
echo "--- Step 3: Hash verification (prod only) ---"
if [ -f "requirements-prod.txt" ]; then
    if grep -q -- '--hash=' requirements-prod.txt; then
        echo "PASS: Production requirements use --hash pins"
    else
        echo "WARN: requirements-prod.txt missing --hash pins"
    fi
fi

echo ""
echo "=== Audit complete ==="
