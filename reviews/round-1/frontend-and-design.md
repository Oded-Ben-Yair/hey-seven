# Frontend & Design -- Hostile Review Round 1

Date: 2026-02-12
Reviewer: frontend-critic

## Overall Score: 58/100

This is a competent boilerplate that would pass a "does it compile" test but would NOT survive a CTO design review for a company that sells premium casino experiences. The brand alignment has the right hex values but misapplies them. The React architecture is stuck in 2024 patterns. The UX is a chat template with casino labels stuck on top. Several decisions actively harm the brand they claim to represent.

---

## Section Scores

| Component | Score | Critical Issues | Important Issues |
|-----------|-------|----------------|-----------------|
| Brand Alignment | 52 | 3 | 4 |
| Page Layout (page.tsx) | 55 | 2 | 3 |
| Header | 62 | 1 | 3 |
| ChatInterface | 58 | 2 | 5 |
| PlayerPanel | 65 | 1 | 3 |
| API Client | 70 | 1 | 2 |
| Types | 72 | 0 | 3 |
| CSS / Tailwind | 60 | 2 | 3 |
| UX Quality | 45 | 3 | 4 |
| Accessibility | 30 | 5 | 3 |
| React/Next.js Patterns | 50 | 3 | 4 |

---

## Critical Issues (MUST fix)

### C1. `layout.tsx` applies `dark` class to `<html>` but the brand is LIGHT theme

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/layout.tsx:26`

```tsx
<html lang="en" className={`${inter.variable} dark`}>
```

The brand-design.md explicitly states: "Warm luxury feel (NOT dark/neon casino aesthetic)", "Light theme with cream backgrounds". Yet the root element has `className="dark"`, which is a Tailwind v3 dark mode toggle. In Tailwind v4, the `dark` variant uses `prefers-color-scheme: dark` by default, not a class. Adding `dark` as a class does nothing in Tailwind v4 without explicit selector configuration -- so this is both wrong AND dead code. The CSS also sets `color-scheme: light` on `html`, directly contradicting the class. This signals a developer who copy-pasted from a dark-mode template without understanding the brand.

**Severity**: Critical (brand contradiction + dead code + signals unfamiliarity with Tailwind v4).

---

### C2. No serif font loaded -- headings use system fallback, not brand typography

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/layout.tsx:5-9`

```tsx
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});
```

The brand specifies serif headings for elegance (Georgia / serif stack). The code loads Google's `Inter` (a sans-serif), assigns it to `--font-inter`, but NEVER USES the variable anywhere. The CSS `body` uses `ui-sans-serif, system-ui` (not `var(--font-inter)`), and headings use `ui-serif, Georgia`. So Inter is downloaded from Google Fonts on every page load for ZERO reason -- it is pure dead weight.

More critically: there is no proper serif font loaded. The brand would benefit from a real serif like Playfair Display, Cormorant Garamond, or similar. Using `ui-serif` as the primary font-family means the user gets whatever serif their OS provides. On Linux this could be DejaVu Serif. On Windows, Cambria. There is zero brand consistency across platforms.

**Severity**: Critical (dead font download + no brand-appropriate serif + cross-platform inconsistency).

---

### C3. `page.tsx` is a massive `"use client"` blob -- zero Server Components

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/page.tsx:1`

```tsx
"use client";
```

The ENTIRE page is a client component. This means:
- `DEMO_PLAYER` data (lines 10-47) is bundled into the client JS instead of being fetched server-side
- Zero SEO benefit from Next.js SSR
- The player profile could be a Server Component that fetches data on the server, with only the chat interface being client-side
- No streaming/Suspense boundaries
- No loading.tsx or error.tsx anywhere in the app

In Feb 2026 with Next.js 15 and React 19, this is a 2023 pattern. The App Router exists specifically to let you keep components on the server. A CTO interviewer will immediately ask "why isn't the player panel a Server Component?"

**Severity**: Critical (defeats the purpose of Next.js 15 App Router).

---

### C4. Sidebar background is `bg-white`, not `bg-hs-cream` -- brand violation

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/page.tsx:59`

```tsx
<aside className="hidden w-80 flex-shrink-0 overflow-y-auto border-r border-hs-border bg-white lg:block">
```

The brand background is cream `#f5f3ef`. The sidebar uses `bg-white` (`#ffffff`). This creates a visible seam between the sidebar and the main content area. The brand doc says cream is THE background -- white should be used for elevated surfaces (cards), not structural panels. The chat input area (ChatInterface.tsx:126) also uses `bg-white` instead of `bg-hs-cream` or `bg-hs-elevated`.

**Severity**: Critical (visible brand inconsistency in the two largest surface areas).

---

### C5. Textarea does NOT auto-grow -- fixed `rows={1}` with `resize-none`

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/ChatInterface.tsx:128-141`

```tsx
<textarea
  rows={1}
  className={clsx(
    "flex-1 resize-none rounded-lg...",
  )}
/>
```

The textarea is locked to one row with resize disabled. If a user types a multi-line message (common in casino host conversations where they describe situations), the text scrolls horizontally or wraps in a tiny single-line box. Every modern chat app (ChatGPT, Slack, iMessage) auto-grows the input as you type. This is a 2020 UX pattern that makes the app feel cheap.

**Severity**: Critical (fundamental UX failure for a chat-primary interface).

---

### C6. Zero accessibility -- no ARIA labels, no roles, no focus management

**Files**: All component files.

There are ZERO `aria-label`, `aria-live`, `role`, or `aria-describedby` attributes in the entire codebase. Specific failures:

1. **Chat messages area** has no `role="log"` or `aria-live="polite"` -- screen readers have no idea new messages appeared
2. **Send button** has no `aria-label="Send message"` -- screen reader says "button" with no context
3. **Status indicator** in header has no `aria-label` -- the pulsing dot is purely visual
4. **Player panel** sections have no landmark roles
5. **Textarea** has no `aria-label` (placeholder is not a substitute for an accessible label)
6. **Loading state** ("Hey Seven is thinking...") has no `aria-live` region -- invisible to screen readers
7. **Navigation** between sidebar and chat has no skip links
8. **Focus is not moved to the new message** after it arrives -- keyboard users lose their place
9. **Specialized cards** (comp offer, reservation, escalation) have no semantic structure -- just divs

A casino company selling to US enterprises will be asked about WCAG 2.1 AA compliance. This would fail every automated accessibility audit.

**Severity**: Critical (legal liability for enterprise casino customers, fails WCAG on multiple criteria).

---

### C7. `chatStream` function exists in API client but is never used

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/lib/api.ts:89-127`

The `chatStream` function implements SSE streaming. The ChatInterface uses the non-streaming `chat` function. This means:
- Users stare at "Hey Seven is thinking..." for the ENTIRE response generation time
- No progressive text display
- The streaming code is dead -- it will silently rot

For a demo to a CTO, the difference between "wait 5 seconds then see a wall of text" and "see text stream in real-time" is the difference between "this feels like ChatGPT" and "this feels like a 2018 chatbot".

**Severity**: Critical (streaming exists but is unused -- worst of both worlds: dead code + bad UX).

---

### C8. The `"use client"` directive on Header.tsx is unnecessary

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/Header.tsx:1`

```tsx
"use client";
```

The Header component has ZERO client interactivity. It receives `status` as a prop and renders JSX. It uses no hooks, no event handlers, no browser APIs. It could be a Server Component that receives status via props. The `"use client"` boundary should be as deep as possible -- the Header needs it only if its PARENT passes it interactive props, but since status is just a string, the parent could pass it as a prop to a Server Component.

This compounds with C3 -- every component is client-side for no reason.

**Severity**: Critical (every unnecessary `"use client"` increases bundle size and defeats SSR).

---

### C9. TIER_STYLES in PlayerPanel uses neon/saturated colors that violate the brand

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/PlayerPanel.tsx:19-40`

```tsx
Gold: { text: "text-yellow-400", ... },
Platinum: { text: "text-slate-300", ... },
Diamond: { text: "text-cyan-300", ... },
```

The brand has ONE accent color: gold `#c5a467`. The tier badges use `yellow-400` (a bright lemon yellow, NOT brand gold), `slate-300` (a cold blue-gray), and `cyan-300` (a NEON cyan). The brand doc explicitly says: "Gold as the singular accent color (no secondary colors like red/blue)." Cyan is as far from "warm luxury" as you can get -- it belongs in a dark-mode developer dashboard, not a premium casino app.

Only `Seven Star` uses the actual brand gold. The other three tiers introduce off-brand colors that a designer would reject immediately.

**Severity**: Critical (three of four tier badges use off-brand colors in a visible, permanent UI element).

---

### C10. No error boundary anywhere in the application

**Files**: No `error.tsx`, no `global-error.tsx`, no `not-found.tsx`, no React error boundaries.

If any component throws (e.g., the API returns unexpected data), the entire app crashes to a white screen. Next.js 15 App Router provides `error.tsx` at any route segment level. There are none. This means:
- A malformed API response crashes the whole page
- A runtime error in PlayerPanel takes down the chat
- There is no recovery path -- the user must hard refresh

**Severity**: Critical (production app with zero error handling at the UI level).

---

## Important Issues (SHOULD fix)

### I1. `DEMO_PLAYER` hardcoded data in page.tsx -- Rule 1 violation risk

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/page.tsx:10-47`

The page embeds 37 lines of fake player data. While the comment says "Replace with real API call", this data will ship to production if anyone forgets. The `getPlayer` API function exists but is never called. Better pattern: fetch on the server side, show a loading state, and display "NOT CONNECTED" if the API is down.

---

### I2. The Inter font import is dead code

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/layout.tsx:5-9`

`Inter` is imported and assigned to `--font-inter` but the variable is never referenced in any CSS or className. The body uses `font-sans` (which maps to the Tailwind default, NOT `--font-inter`). This is downloading a 30KB+ font file that is never rendered. Delete it or wire it up.

---

### I3. Reservation and escalation cards use emerald and amber colors -- off-brand

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/ChatInterface.tsx:242, 264`

Reservation cards use `text-emerald-400` and escalation cards use `text-amber-400`. The brand has ONE accent color (gold). Semantic colors for success/warning should use variations of the brand palette (e.g., a warm green derived from the gold, a muted amber). Raw Tailwind palette colors (`emerald-400`, `amber-400`) feel generic.

---

### I4. `timestamp` is typed as `Date` but created from `new Date()` -- serialization trap

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/lib/types.ts:15`

```tsx
timestamp: Date;
```

The `ChatMessage.timestamp` is a `Date` object. If messages are ever serialized to JSON (localStorage, SSR hydration, API response), `Date` objects become strings and `.toLocaleTimeString()` on a string will crash. Should use ISO string or unix timestamp with a formatter.

---

### I5. `metadata` field uses a union type but has no discriminated union pattern

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/lib/types.ts:18`

```tsx
metadata?: CompOfferMetadata | ReservationMetadata | EscalationMetadata;
```

This is a "god union" -- there is no way to narrow the type at runtime without unsafe casting. The `SpecializedContent` component uses `message.metadata as CompOfferMetadata | undefined` (line 208). This `as` cast is a code smell. The correct pattern is a discriminated union on `type`:

```tsx
type ChatMessage =
  | { type: "text"; ... }
  | { type: "comp_offer"; metadata: CompOfferMetadata; ... }
  | { type: "reservation_confirmation"; metadata: ReservationMetadata; ... }
  | { type: "escalation_notice"; metadata: EscalationMetadata; ... }
```

---

### I6. No mobile experience for the player panel

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/page.tsx:59`

```tsx
<aside className="hidden w-80 ... lg:block">
```

On screens below `lg` (1024px), the player panel vanishes entirely. There is no mobile drawer, no slide-out panel, no hamburger toggle. For a demo at an interview, if the interviewer opens this on a laptop with a smaller screen, they see ONLY the chat -- no player context at all. The casino host use case (seeing player data while chatting) is the core value prop. Hiding it on smaller screens defeats the demo.

---

### I7. `msgId()` uses `Date.now()` -- non-unique under rapid sends

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/ChatInterface.tsx:29-31`

```tsx
function msgId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
```

While `Math.random()` adds entropy, the `Date.now()` prefix is millisecond precision. On fast networks where send + response complete within 1ms, IDs could collide. The API client uses `crypto.randomUUID()` in one place (line 120) and this pattern elsewhere. Pick one. `crypto.randomUUID()` is available in all modern browsers and is the correct choice.

---

### I8. `adaptChatResponse` hardcodes `type: "text"` -- specialized messages impossible

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/lib/api.ts:15-25`

```tsx
function adaptChatResponse(backend: BackendChatResponse): ChatResponse {
  return {
    message: {
      type: "text",  // Always "text", never comp_offer, reservation, etc.
    },
  };
}
```

The frontend has beautiful specialized cards for comp offers, reservations, and escalations. But the API adapter ALWAYS maps to `type: "text"`. Those cards will NEVER render. The comp offer, reservation, and escalation card components are entirely dead code. This is the worst kind of demo: the code exists to show ambition, but the wiring is missing so it never actually works.

**Severity**: Important-borderline-critical. A CTO would ask "show me the comp offer card" and you could not.

---

### I9. `BackendChatResponse.escalation` and `compliance_flags` are received but discarded

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/lib/api.ts:11-12, 15-25`

```tsx
interface BackendChatResponse {
  escalation: boolean;
  compliance_flags: string[];
}
```

These fields are defined but `adaptChatResponse` throws them away. The backend sends escalation status and compliance flags -- both are critical for a casino AI host. The frontend should display compliance warnings and escalation status. Ignoring them suggests the developer didn't think through the data flow.

---

### I10. `PlayerPanel` "use client" is unnecessary

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/PlayerPanel.tsx:1`

Like the Header, PlayerPanel has zero interactivity. No hooks, no event handlers, no browser APIs. It is a pure render component that could be a Server Component.

---

### I11. H1 heading in Header uses sans-serif class, not serif

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/Header.tsx:32`

```tsx
<h1 className="text-lg font-semibold tracking-tight text-white">
```

The CSS applies `font-family: ui-serif, Georgia...` to `h1, h2, h3, h4`. But the Header h1 has no explicit font class. It should get the serif via the CSS rule -- UNLESS Tailwind's reset or the `font-sans` on body overrides it. Need to verify whether the CSS specificity actually applies. If Tailwind resets headings to inherit body font, the h1 is sans-serif, violating the brand.

---

### I12. Section headings in PlayerPanel use h3 with sans-serif styling override

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/PlayerPanel.tsx:199`

```tsx
<h3 className="text-xs font-semibold uppercase tracking-wider text-hs-text-muted">
```

The `text-xs font-semibold uppercase tracking-wider` styling makes these look like utility labels, not headings. The CSS sets serif for h3, but at `text-xs` (12px), a serif font looks terrible. Either use a different element (`<span>`) or accept that headings at 12px violate the typography system.

---

### I13. `formatDate` in PlayerPanel creates new Date objects -- potential timezone bug

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/PlayerPanel.tsx:51-56`

```tsx
function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", { ... });
}
```

`new Date("2025-01-28")` is parsed as UTC midnight. `toLocaleDateString` converts to local timezone. A user in UTC-8 would see "Jan 27" instead of "Jan 28". Use date-fns `parseISO` or explicitly handle timezone.

---

### I14. Chat input area uses `bg-white` instead of brand surface color

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/ChatInterface.tsx:126`

```tsx
<div className="border-t border-hs-border bg-white px-6 py-4">
```

Same issue as C4. `bg-white` should be `bg-hs-surface` or `bg-hs-elevated` to stay on-brand.

---

## Outdated Patterns (for Feb 2026)

### O1. No React 19 features used

React 19 (shipped with Next.js 15) introduced:
- **`use` hook** for reading promises/context -- could replace manual async state management in ChatInterface
- **Server Actions** -- the chat API call could be a server action
- **`useOptimistic`** -- the user message could appear instantly with optimistic UI
- **`useFormStatus`** -- the send button could disable automatically
- **`<form>` actions** -- the chat input could use the new form action pattern

None of these are used. The chat still uses `useState` + `useCallback` + manual `try/catch` -- a React 18 pattern.

### O2. No Suspense boundaries

Next.js 15 App Router uses Suspense for loading states. There is no `loading.tsx`, no `<Suspense>` wrapper, no streaming SSR. The player panel should load inside a Suspense boundary with a skeleton.

### O3. No `next/image` for player photos

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/components/PlayerPanel.tsx:69`

```tsx
<img src={player.photoUrl} alt={player.name} ... />
```

Raw `<img>` tag instead of `next/image`. Loses automatic optimization, WebP conversion, lazy loading, and will trigger ESLint warnings with `@next/next/no-img-element`.

### O4. Tailwind v4 `@theme inline` duplicates `:root` CSS variables

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/src/app/globals.css:8-24, 26-39`

The file defines `:root` CSS variables AND `@theme inline` Tailwind tokens with identical values. In Tailwind v4, `@theme inline` IS the source of truth for design tokens. The `:root` block is redundant. Either use `@theme inline` exclusively (and reference with `--color-hs-*` in CSS) or use `:root` exclusively (and abandon Tailwind token integration). Having both means changes require updating two places.

### O5. No `tailwind.config.*` file -- but also no `@source` directive

Tailwind v4 uses `@source` in CSS to specify content paths, or auto-detects from the project. The CSS file has `@import "tailwindcss"` but no `@source`. This likely works because Tailwind v4 auto-detects, but it should be explicit for production: `@source "../src/**/*.{tsx,ts}"`.

### O6. `next.config.ts` is empty

**File**: `/home/odedbe/projects/hey-seven/boilerplate/frontend/next.config.ts`

No `images.remotePatterns` configured (needed for `next/image` with external photos). No `reactStrictMode` explicitly set. No `poweredByHeader: false` (security). No `headers()` for CSP.

---

## Missing Features (what a CTO would expect)

### M1. No authentication / session management

A casino host app without auth is a toy. Even a demo should show an auth pattern (NextAuth, Clerk, or even a simple cookie).

### M2. No conversation history / persistence

Messages are in React state. Refresh the page and everything is gone. No localStorage backup, no API-side conversation history.

### M3. No typing indicator from agent

The "Hey Seven is thinking..." text is static. A real chat shows a pulsing dot animation or progressive text streaming. See C7 -- streaming exists but is unused.

### M4. No ability to clear chat / start new conversation

No "New Conversation" button. The user is stuck with one conversation thread.

### M5. No dark/light theme toggle

While the brand is light theme, enterprise demos often happen on projectors in dark rooms. A theme toggle shows design system maturity.

### M6. No keyboard shortcuts

No `Cmd+K` for search, no `Escape` to clear input, no keyboard navigation between messages. Chat power users expect these.

### M7. No message retry / regenerate

If the AI gives a bad response, there is no "Regenerate" button. This is standard in 2026 AI chat interfaces.

### M8. No breadcrumb or navigation structure

Single page app with no routes. A real host dashboard would have: `/chat`, `/players`, `/comps`, `/analytics`. This feels like a CodePen, not a product.

### M9. No favicon or OG image

`metadata.icons` references `/favicon.ico` but there is no evidence the file exists. No Open Graph images for link previews.

### M10. No testing infrastructure

No `vitest`, no `@testing-library/react`, no Playwright e2e. Zero tests of any kind. A CTO would ask about test strategy.

---

## Design Recommendations

### D1. Replace system serif with a proper brand serif font

Load Playfair Display or Cormorant Garamond from Google Fonts for headings. This immediately elevates the "warm luxury" feel beyond generic Georgia.

### D2. Create a proper design token system

The current approach has CSS variables, Tailwind tokens, AND hardcoded Tailwind classes (`bg-emerald-500/10`, `text-cyan-300`) all fighting each other. Centralize everything in `@theme inline` and reference tokens exclusively. No raw Tailwind palette colors.

### D3. Add subtle animations

The brand is "warm luxury". The only animation is a pulsing dot and a spinning icon. Add:
- Fade-in on new messages
- Slide-in on sidebar sections
- Subtle hover transitions on cards
- Gold shimmer on comp offer cards

### D4. Use the CTA gradient button consistently

The brand CTA is `linear-gradient(to right, #d4b872, #c5a467)`. The send button uses this correctly. But other interactive elements (tier badge, metric cards) don't have any hover/interactive state.

### D5. Add the Hey Seven logo

The header uses a `Dice5` Lucide icon as the logo. The brand has a real logo (gold icon on dark background). Using a generic library icon signals "this is a prototype, not a product."

---

## Minor Issues

### m1. Double-dash `--` used as separator in metadata cards

**Files**: ChatInterface.tsx lines 231, 251, 273

Using `--` as a visual separator looks unprofessional. Use a proper `<span className="mx-1 text-hs-text-muted">|</span>` or a `<span className="mx-1">&#8226;</span>` dot separator.

### m2. `clsx` used but Tailwind v4 has built-in conditional class support

In Tailwind v4, you can use `data-*` attributes and `group-*` variants for many conditional styling cases. `clsx` is still valid, but some usages could be simplified with data attributes.

### m3. `STATUS_CONFIG` maps icon type as `typeof Wifi` which is imprecise

**File**: Header.tsx:13

`typeof Wifi` works but `LucideIcon` would be the proper type import from lucide-react.

### m4. Missing `key` prop risk in preference list

**File**: PlayerPanel.tsx:126

```tsx
{player.preferences.map((pref) => (
  <li key={pref} ...>
```

Using the preference string as key works until two preferences have the same text. Use index or generate IDs.

### m5. `CompOfferCard` shows `metadata.currency` before `metadata.value` with no separator

**File**: ChatInterface.tsx:231

```tsx
{metadata.compType} -- {metadata.currency}
{metadata.value.toLocaleString()}
```

The line break between `metadata.currency` and `metadata.value.toLocaleString()` is a JSX whitespace artifact. It renders as "USD1,250" with no space. Should be `{metadata.currency} {metadata.value.toLocaleString()}` or use `Intl.NumberFormat` with the currency option.

### m6. Time display format may confuse international users

**File**: ChatInterface.tsx:194-197

`toLocaleTimeString` with `hour: "2-digit", minute: "2-digit"` is locale-dependent. Casino operations span time zones. Should show timezone or use 24-hour format for professional context.

### m7. No ESLint configuration visible

No `.eslintrc.*` or `eslint.config.*` file. The `package.json` has a `lint` script (`next lint`) but no eslint config to customize rules. Next.js defaults are minimal.

### m8. `postcss.config.mjs` is correct but should be `.ts` for consistency

The project uses TypeScript everywhere except PostCSS config. Minor consistency issue.

### m9. Console errors would be invisible to the user

The `catch` in `sendMessage` shows the error inline in the chat, which is good. But there is no console.error or logging. In production, API errors should be tracked (Sentry, etc.).

### m10. `request<T>` function does not handle network failures gracefully

**File**: api.ts:27-45

If `fetch` throws (network down, DNS failure), the error message would be the raw JS error ("Failed to fetch"), not a user-friendly message. The ChatInterface catch handler wraps this but the raw error leaks into the UI.

---

## Summary Table

| Category | Score | Verdict |
|----------|-------|---------|
| Would a CTO be impressed? | No | This is a functional chat template with casino vocabulary, not a product demo |
| Brand fidelity | Partial | Right hex values in CSS, wrong application in components |
| Modern React | No | 2024 patterns, zero React 19 / Next.js 15 features used |
| Production readiness | No | No auth, no persistence, no error boundaries, no tests |
| Accessibility | Fail | Would fail every automated audit tool |
| Demo quality | Marginal | The chat works, the streaming does not, the special cards never render |

---

## Priority Fix Order

1. **C3 + C8 + I10**: Restructure Server/Client component boundary (biggest architectural win)
2. **C6**: Add comprehensive ARIA labels and roles
3. **C5**: Auto-growing textarea
4. **C7 + I8**: Wire up streaming AND specialized message type mapping
5. **C1**: Remove `dark` class, verify light theme
6. **C2 + I2**: Delete Inter, load a proper serif font
7. **C4 + I14**: Replace `bg-white` with brand tokens
8. **C9**: Replace neon tier colors with brand-derived palette
9. **C10**: Add error.tsx and global-error.tsx
10. **O1-O3**: Adopt React 19 patterns (useOptimistic, Suspense, next/image)
