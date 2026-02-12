# Frontend & Design — Hostile Review Round 2

**Date:** 2026-02-12
**Reviewer:** frontend-critic (code-judge, hostile mode, Claude Opus 4.6)
**Score: 79/100** (Round 1: 58/100, Delta: +21)

## Round 1 Critical Issues — Resolution Status

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| C1 | `dark` class on HTML | FIXED | Clean removal |
| C2 | No serif font | FIXED | Playfair Display loaded and wired |
| C3 | Entire page `"use client"` | FIXED | Server/Client split with composition pattern |
| C4 | `bg-white` surfaces | FIXED | All brand tokens applied |
| C5 | No textarea auto-grow | FIXED | Standard auto-grow with cap |
| C6 | Zero accessibility | PARTIALLY_FIXED | 0→33 ARIA attributes, proper roles/landmarks, some gaps |
| C7 | chatStream unused | FIXED | Primary path with fallback to regular chat |
| C8 | Header `"use client"` unnecessary | FIXED | Directive removed |
| C9 | Neon tier colors | FIXED | Warm brand palette (amber, gold, gray) |
| C10 | No error boundary | MOSTLY_FIXED | error.tsx + loading.tsx, missing global-error.tsx |
| BONUS | `adaptChatResponse` hardcodes type | FIXED | `detectMessageType` with 3-layer detection |

**Summary: 9 FULLY/MOSTLY FIXED, 2 PARTIALLY FIXED**

## New Issues Found

### Important
- **N2:** `chatStream` in api.ts:157-163 hardcodes `type: "text"` — streaming path (now PRIMARY) loses specialized card rendering
- **N4:** `Suspense` wrapping `PlayerPanel` (page.tsx:97) is cosmetic — data is pre-fetched, never suspends. CTO would see through this.
- **N6:** Reservation/escalation cards still use `emerald-600`/`amber-600` (ChatInterface.tsx:317,325,339) — off-brand

### Minor
- N1: Header inside client boundary — `"use client"` removal is cosmetic
- N3: Streaming fallback `usedStream` flag is fragile under refactoring
- N5: Header status uses `emerald-500` — arguably off-brand
- N7: `next.config.ts` still empty (no CSP, no poweredByHeader)
- N9: No `global-error.tsx` or `not-found.tsx`
- N10: `DEMO_PLAYER` data hardcoded in page.tsx (37 lines)
- N11: `formatDate` timezone bug (UTC midnight → wrong date in negative UTC offsets)

### Remaining from Round 1
- No mobile player panel (hidden on <1024px)
- No React 19 features (useOptimistic, useFormStatus)
- No `next/image` optimization
- `:root` and `@theme inline` duplicate values
