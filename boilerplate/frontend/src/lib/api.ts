import type { ChatResponse, PlayerProfile, CompResult } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
 * Returns the assistant's response message.
 */
export async function chat(
  message: string,
  playerId?: string
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, player_id: playerId }),
  });
}

/**
 * Fetch a player's full profile by ID.
 */
export async function getPlayer(playerId: string): Promise<PlayerProfile> {
  return request<PlayerProfile>(`/api/players/${encodeURIComponent(playerId)}`);
}

/**
 * Request a comp calculation for a player.
 */
export async function calculateComp(
  playerId: string,
  compType: string
): Promise<CompResult> {
  return request<CompResult>("/api/comps/calculate", {
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
  onChunk: (chunk: string) => void
): Promise<ChatResponse> {
  const url = `${BASE_URL}/api/chat/stream`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, player_id: playerId }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream error ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    fullContent += text;
    onChunk(text);
  }

  return {
    message: {
      id: crypto.randomUUID(),
      role: "assistant",
      content: fullContent,
      type: "text",
      timestamp: new Date(),
    },
  };
}
