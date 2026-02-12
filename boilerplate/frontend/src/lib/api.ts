import type { ChatResponse, ChatMessage, MessageType, PlayerProfile, CompResult } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

/** Backend chat response shape (different from frontend ChatMessage) */
interface BackendChatResponse {
  response: string;
  thread_id: string;
  player_id: string | null;
  escalation: boolean;
  compliance_flags: string[];
  /** Optional: backend may include a message type hint */
  message_type?: string;
}

/**
 * Detect the appropriate message type from the backend response.
 * Prefers an explicit backend field, falls back to content heuristics.
 */
function detectMessageType(backend: BackendChatResponse): MessageType {
  // Explicit backend type field takes priority
  if (backend.message_type === "comp_offer") return "comp_offer";
  if (backend.message_type === "reservation") return "reservation_confirmation";
  if (backend.message_type === "escalation") return "escalation_notice";

  // Escalation flag from backend
  if (backend.escalation) return "escalation_notice";

  // Content-based heuristics as fallback
  const lower = backend.response.toLowerCase();
  if (
    lower.includes("comp") &&
    (lower.includes("offer") || lower.includes("complimentary") || lower.includes("credit"))
  ) {
    return "comp_offer";
  }
  if (
    lower.includes("reservation") &&
    (lower.includes("confirmed") || lower.includes("booked"))
  ) {
    return "reservation_confirmation";
  }

  return "text";
}

/** Convert backend response into frontend ChatResponse */
function adaptChatResponse(backend: BackendChatResponse): ChatResponse {
  return {
    message: {
      id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      role: "assistant",
      content: backend.response,
      type: detectMessageType(backend),
      timestamp: new Date(),
    },
  };
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "Unknown error");
    throw new Error(
      `API error ${response.status} on ${path}: ${errorBody}`
    );
  }

  return response.json() as Promise<T>;
}

/**
 * Send a chat message to the AI host agent.
 * Maps backend response shape to frontend ChatResponse.
 */
export async function chat(
  message: string,
  playerId?: string
): Promise<ChatResponse> {
  const backend = await request<BackendChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({ message, thread_id: playerId }),
  });
  return adaptChatResponse(backend);
}

/**
 * Fetch a player's full profile by ID.
 */
export async function getPlayer(playerId: string): Promise<PlayerProfile> {
  return request<PlayerProfile>(
    `/api/v1/player/${encodeURIComponent(playerId)}`
  );
}

/**
 * Request a comp calculation for a player.
 */
export async function calculateComp(
  playerId: string,
  compType: string
): Promise<CompResult> {
  return request<CompResult>("/api/v1/comp/calculate", {
    method: "POST",
    body: JSON.stringify({ player_id: playerId, comp_type: compType }),
  });
}

/**
 * Stream a chat response using Server-Sent Events.
 * Calls onChunk for each text fragment as it arrives.
 * Returns the full assembled message when the stream completes.
 */
export async function chatStream(
  message: string,
  playerId: string | undefined,
  onChunk: (chunk: string) => void,
  signal?: AbortSignal
): Promise<ChatMessage> {
  const url = `${BASE_URL}/api/v1/chat/stream`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: playerId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream error ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      fullContent += text;
      onChunk(text);
    }
  } finally {
    reader.releaseLock();
  }

  // Use detectMessageType instead of hardcoding "text" so streaming
  // responses can still render as comp_offer / reservation / escalation cards.
  const syntheticBackend: BackendChatResponse = {
    response: fullContent,
    thread_id: playerId ?? "",
    player_id: null,
    escalation: false,
    compliance_flags: [],
  };

  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role: "assistant",
    content: fullContent,
    type: detectMessageType(syntheticBackend),
    timestamp: new Date(),
  };
}
