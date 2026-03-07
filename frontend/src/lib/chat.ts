/**
 * SSE streaming chat client for Hey Seven backend.
 *
 * Connects to POST /chat with { message, thread_id } and streams
 * SSE events: metadata, token, sources, node, done, error.
 */

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onMetadata: (threadId: string) => void;
  onSources: (sources: string[]) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

/**
 * Send a message and stream the response via SSE.
 * Uses fetch + ReadableStream (not EventSource, since POST is needed).
 */
export async function streamChat(
  message: string,
  threadId: string | null,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<string | null> {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId,
    }),
    signal,
  });

  if (!response.ok) {
    const text = await response.text();
    callbacks.onError(`Server error: ${response.status} — ${text}`);
    return threadId;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return threadId;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentThreadId = threadId;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let eventType = "";
    let eventData = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        eventData = line.slice(6);
      } else if (line === "" && eventType && eventData) {
        try {
          const parsed = JSON.parse(eventData);

          switch (eventType) {
            case "metadata":
              currentThreadId = parsed.thread_id;
              callbacks.onMetadata(parsed.thread_id);
              break;
            case "token":
              callbacks.onToken(parsed.content);
              break;
            case "sources":
              callbacks.onSources(parsed.sources || []);
              break;
            case "done":
              callbacks.onDone();
              break;
            case "error":
              callbacks.onError(parsed.message || "Unknown error");
              break;
          }
        } catch {
          // Non-JSON data line, skip
        }
        eventType = "";
        eventData = "";
      }
    }
  }

  return currentThreadId;
}
