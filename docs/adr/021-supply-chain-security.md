# ADR-021: Supply Chain Security Strategy

## Status
Accepted

## Context
Production deployments need protection against dependency confusion, typosquatting, and known CVE vulnerabilities in transitive dependencies.

## Decision
Three-layer supply chain protection:

### Layer 1: Build-time (Dockerfile)
- `--require-hashes` for pip install (requirements-prod.txt)
- Digest-pinned base image (`@sha256:...`)
- Multi-stage build (builder deps don't enter production image)

### Layer 2: CI/CD (pre-deploy)
- `pip-audit` for CVE scanning (`scripts/security-audit.sh`)
- `syft` for SBOM generation (SPDX/CycloneDX)
- `grype` for SBOM-based vulnerability scanning
- All dependency versions pinned with `==`

### Layer 3: Runtime (Cloud Run)
- Non-root user (`appuser`)
- No shell utilities (no curl, wget)
- Read-only root filesystem (configurable)
- No new privileges (configurable)

## Consequences
- Positive: Known CVEs caught before deployment
- Positive: SBOM provides audit trail for compliance
- Positive: Hash pinning prevents MITM attacks on pip install
- Negative: Hash regeneration required on every dependency update
- Negative: pip-audit adds ~30s to CI pipeline
