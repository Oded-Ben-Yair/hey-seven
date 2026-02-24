# ADR 010: Middleware Execution Order

## Status
Accepted (R48)

## Context
Starlette executes ASGI middleware in REVERSE add order. The ordering determines which security checks run first and which requests consume resources.

## Decision

Execution order (outermost → innermost):

```
Request → BodyLimit → ErrorHandling → Logging → Security → RateLimit → ApiKey → App
```

### Rationale per layer:
1. **BodyLimit** (outermost): Reject oversized payloads before consuming memory
2. **ErrorHandling**: Catch unhandled exceptions from all inner layers
3. **Logging**: Record all requests including those rejected by downstream middleware
4. **SecurityHeaders**: Add X-Content-Type-Options, HSTS, etc. to ALL responses
5. **RateLimit**: Throttle before auth to prevent API key brute-force
6. **ApiKey** (innermost): Authenticate only after rate-limited

### Why RateLimit before ApiKey (R48 fix)
Previously ApiKey executed before RateLimit. An attacker could attempt unlimited wrong keys without being rate-limited. The R48 fix swaps the order so rate limiting applies to ALL requests, including unauthenticated ones.

## Consequences
- Positive: API key brute-force is rate-limited
- Positive: Oversized payloads never reach the LLM
- Negative: Rate limit state is consumed by unauthenticated requests (minor resource cost)
