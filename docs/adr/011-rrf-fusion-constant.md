# ADR-011: RRF Fusion Constant (k=60)

## Status
Accepted

## Context
Reciprocal Rank Fusion (RRF) combines multiple retrieval strategies. The `k` parameter controls how much higher-ranked results are favored over lower-ranked ones.

## Decision
Use k=60 per the original RRF paper (Cormack et al., 2009). This is the standard value used in production systems and provides good balance between emphasizing top results and incorporating diverse rankings.

## Consequences
- Configurable via `RRF_K` setting (default 60)
- Higher k = more equal weighting across ranks (flatter distribution)
- Lower k = more emphasis on top-ranked results
