# Frontend Tools & Patterns (Feb 2026)

## Sources
- Perplexity deep research (60 citations)
- React official blog, Next.js blog, Tailwind blog
- npm download stats, GitHub release notes

---

## Recommended Stack for Casino Host Chat UI

```
Framework:     Next.js 15.2+ (App Router, patched for CVEs)
React:         React 19.2 (Server Components, Compiler)
Styling:       Tailwind CSS 4.1 (CSS-first config)
Components:    shadcn/ui + Prompt Kit + AI Elements
AI Data:       Vercel AI SDK 6 (useChat, streamText)
Streaming:     SSE via AI SDK (standard for LLM)
Animation:     Motion (or Tailwind CSS Motion lightweight)
State:         Zustand (client), Server Components (server)
Validation:    Zod (forms + AI structured output)
```

---

## 1. Next.js 15 (+ 16)

### Release Timeline
- **15.0** (Oct 2024): React 19 RC, Turbopack stable for dev
- **15.1** (Dec 2024): React 19 stable, `after()` API
- **15.2** (Feb 2025): Streaming metadata, redesigned error overlay
- **16** (2025): Turbopack default, `refresh()` API, incremental prefetching

### Key Features for Chat UIs
- **Partial Prerendering (PPR)**: Static shell + streaming dynamic via Suspense
- **`use cache` directive**: Explicit caching (opt-in, not automatic like v14)
- **Streaming metadata**: Async metadata doesn't block initial HTML
- **Server Actions**: Direct server mutation without API routes
- **`after()` API**: Non-blocking post-response operations (logging, analytics)

### Gotchas
- Caching is **opt-in** (NOT automatic like v14)
- Suspense boundary must wrap one level ABOVE data-fetching component
- POST requests NOT auto-memoized
- **CVE-2025-66478** (CVSS 10.0): RSC protocol RCE — MUST be on patched version

---

## 2. React 19.2

### Production Features
| Feature | Use Case |
|---------|----------|
| Server Components | Data fetching, reducing JS bundle |
| Server Actions | Form handling, mutations |
| `use()` hook | Read promises/context during render |
| `useActionState` | Manage Server Action state |
| `useFormStatus` | Read parent form pending state |
| `useOptimistic` | Instant UI before server confirmation |
| React Compiler 1.0 | Auto-memoization (eliminates useMemo/useCallback) |

### React 19.2 New (Feb 2026)
- **`<Activity>`**: `mode="visible"|"hidden"` — preserves state without unmounting. Ideal for chat tab switching
- **Batched Suspense**: Multiple boundaries resolving simultaneously batched into single update
- **View Transitions API**: Integration with Suspense
- **React Compiler stable at 1.0**: 20-40% less manual memoization code

### Chat-Specific Patterns
```typescript
// Optimistic message sending
const [optimisticMessages, addOptimistic] = useOptimistic(
  messages,
  (state, newMessage) => [...state, { ...newMessage, sending: true }]
);

// Parallel data fetching with use()
async function ChatPage({ params }) {
  const messagesPromise = fetchMessages(params.chatId);
  const usersPromise = fetchUsers(params.chatId);
  return (
    <Suspense fallback={<ChatSkeleton />}>
      <ChatView messages={messagesPromise} users={usersPromise} />
    </Suspense>
  );
}
```

---

## 3. Tailwind CSS 4

### Performance
| Metric | v3.4 | v4.0 | Improvement |
|--------|------|------|-------------|
| Full builds | 378ms | 100ms | 3.78x |
| Incremental (new CSS) | 44ms | 5ms | 8.8x |
| Incremental (no new) | 35ms | 192us | 182x |

### Breaking Changes (Migration Gotchas)
| Change | v3 | v4 | Fix |
|--------|----|----|-----|
| `ring` default | 3px | 1px | Use `ring-3` explicitly |
| `border` color | gray-200 | currentColor | Add `border-gray-200` |
| `hover` variant | Always | `@media (hover: hover)` | Override: `@custom-variant hover (&:hover)` |
| Config | tailwind.config.js | CSS `@theme` | Run `npx @tailwindcss/upgrade` |
| Button cursor | pointer | default | Add `cursor-pointer` |

### New Features
- Container queries built-in (`@container`, `@sm`, `@lg`)
- 3D transforms (`rotate-x-*`, `perspective-*`)
- `@starting-style` variant (CSS entry animations)
- Text shadows (v4.1): `text-shadow-sm/md/lg`
- Masking utilities (v4.1)
- `pointer-fine`/`pointer-coarse` variants

### Migration
```bash
npx @tailwindcss/upgrade  # Handles most mechanical changes
# WARNING: aggressive text replacement — always regression-test
```

---

## 4. Chat UI Libraries

### Comparison
| Library | Type | Best For |
|---------|------|----------|
| **Vercel AI SDK 6** | Data layer + hooks | Streaming AI chat (THE STANDARD) |
| **shadcn/ui + Prompt Kit** | Composable UI | Custom AI chat UI |
| **AI Elements** (elements.ai-sdk.dev) | AI-specific components | Deep AI SDK integration |
| **assistant-ui** | AI chat components | AI-optimized patterns |
| **Stream Chat React** | Full platform | Enterprise scale (100M+ users) |
| **CopilotKit** | In-app AI assistant | Embedded AI agents |

### Vercel AI SDK 6 (THE STANDARD)
- 20M+ monthly npm downloads
- Key hooks: `useChat`, `useCompletion`, `useObject`, `useAssistant`
- New in v6: `ToolLoopAgent`, enhanced tool calling, MCP support, HITL approval
- 25+ LLM providers (OpenAI, Anthropic, Google, etc.)

```typescript
// Server: app/api/chat/route.ts
import { streamText } from 'ai';
import { google } from '@ai-sdk/google';

export async function POST(req: Request) {
  const { messages } = await req.json();
  const result = streamText({
    model: google('gemini-2.5-flash'),
    messages,
  });
  return result.toUIMessageStreamResponse();
}

// Client: components/Chat.tsx
'use client';
import { useChat } from '@ai-sdk/react';

export function Chat() {
  const { messages, input, handleInputChange, handleSubmit, status } = useChat();
  return (
    <div>
      {messages.map(msg => (
        <div key={msg.id}>{msg.role}: {msg.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
        <button disabled={status === 'streaming'}>Send</button>
      </form>
    </div>
  );
}
```

### shadcn/ui Chat Ecosystem (Feb 2026)
- **Prompt Kit**: Chat Input, Message, Markdown, CodeBlock, Response Stream, File Upload, Reasoning, Loader
- **AI Elements** (elements.ai-sdk.dev): Chatbot, Attachments, Chain of Thought, Confirmation, Citations, Suggestion, Task components
- **Zola**: Open-source multi-model chat UI
- **Recent**: Unified `radix-ui` package, RTL support

---

## 5. Streaming Architecture

### Protocol Selection
| Protocol | Direction | Best For |
|----------|-----------|----------|
| **SSE** | Server -> Client | LLM streaming (RECOMMENDED) |
| **WebSocket** | Bidirectional | Multi-user real-time chat |

### SSE Headers (Required)
```typescript
{
  'Content-Type': 'text/event-stream; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
  'Connection': 'keep-alive',
  'X-Accel-Buffering': 'no'  // CRITICAL for nginx/Cloud Run
}
```

### Best Practices
- Vercel AI SDK handles SSE, parsing, state automatically via `useChat`
- Heartbeat messages every 30s to prevent proxy timeouts
- Edge Runtime has 30s timeout — delegate LLM calls to Node.js runtime
- ReadableStream + TransformStream for custom pipelines

---

## 6. Animation Libraries

| Library | Bundle | Best For |
|---------|--------|----------|
| **Motion** (framer-motion successor) | 85KB | Standard choice, comprehensive |
| **Tailwind CSS Motion** | 5KB | Lightweight, CSS-native |
| **React Spring** | 45KB | Data viz, interactive |
| **GSAP** | 78KB | Premium marketing |

### Recommendation
Default to Motion. For ultra-lightweight: Tailwind CSS Motion (5KB, GPU-accelerated).

---

## 7. Architecture Decision: Our Frontend

For the casino host chat UI:

### Pattern A: Full Vercel AI SDK (Recommended if Next.js frontend)
- `useChat` hook handles streaming, state, error recovery
- Server route calls our LangGraph backend
- SSE for token-by-token display
- shadcn/ui + Prompt Kit for chat components

### Pattern B: Custom SSE (If FastAPI-direct)
- EventSource API from browser
- Custom state management (Zustand)
- More control, more code
- Better for non-standard streaming patterns

### Pattern C: Hybrid
- Next.js frontend with API route proxy
- API route calls FastAPI backend
- `useChat` on frontend, FastAPI streams from LangGraph
- **Best of both worlds — likely our approach**

---

## Key Takeaways for Assignment

1. **Use Vercel AI SDK `useChat`** for the chat interface — it's the industry standard
2. **SSE, not WebSocket** for LLM streaming
3. **Tailwind CSS 4** if starting fresh, but v3 is fine if already set up
4. **shadcn/ui + Prompt Kit** for chat components (copy-paste, no dependency)
5. **React 19.2 `useOptimistic`** for instant message feedback
6. **`<Activity>` component** if we need tab switching (chat vs dashboard)
7. **Motion** for entrance/exit animations on messages
8. **Server Components** for initial page load (player context, conversation history)
