# ADR-013: SSE Timeout (60s Default)

## Status
Accepted

## Context
SSE streams can hang if the LLM stops generating tokens mid-response. Without a timeout, the connection stays open indefinitely, consuming Cloud Run concurrency slots.

## Decision
60-second SSE timeout configured via `SSE_TIMEOUT_SECONDS`. Cloud Run request timeout (180s) provides an outer bound; the SSE timeout is the application-level inner bound.

## Consequences
- Abandoned streams are cleaned up within 60s
- Cloud Run concurrency slot freed for new requests
- Client receives `done` event or connection close
