# ADR-015: Circuit Breaker Parameters

## Status
Accepted

## Context
The circuit breaker protects against cascading LLM failures. Parameters must balance between: (a) tripping too easily (false positives block legitimate traffic) and (b) not tripping fast enough (sustained failures propagate).

## Decision
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| failure_threshold | 5 | 5 failures in window = consistent problem, not transient |
| cooldown_seconds | 60 | 1-minute cooldown before half-open probe |
| rolling_window_seconds | 300 | 5-minute window smooths burst errors |

## Consequences
- Transient errors (1-4 in 5 min) do not trip the breaker
- Sustained outage (5+ in 5 min) trips within seconds
- Recovery probe after 60s; if probe succeeds, halves failure count (not full reset)
- Cross-instance sync via Redis reduces window to 2s propagation
