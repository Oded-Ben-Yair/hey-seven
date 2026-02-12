"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Gift,
  CalendarCheck,
  AlertTriangle,
  Bot,
  User,
} from "lucide-react";
import clsx from "clsx";
import type {
  ChatMessage,
  ConnectionStatus,
  CompOfferMetadata,
  ReservationMetadata,
  EscalationMetadata,
} from "@/lib/types";
import { chatStream, chat as chatFallback } from "@/lib/api";

interface ChatInterfaceProps {
  playerId: string;
  onStatusChange: (status: ConnectionStatus) => void;
}

/** Generate a unique message ID */
function msgId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Welcome message shown at the start of every session */
const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hello, I'm your AI Casino Host. I can help with comp offers, restaurant reservations, room bookings, and player insights. What can I do for you today?",
  type: "text",
  timestamp: new Date(),
};

export function ChatInterface({ playerId, onStatusChange }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  /* Auto-scroll to bottom on new messages or streaming content */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  /* Focus input on mount */
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  /* Auto-grow textarea as user types (C5) */
  useEffect(() => {
    const textarea = inputRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 128)}px`;
    }
  }, [input]);

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatMessage = {
      id: msgId(),
      role: "user",
      content: trimmed,
      type: "text",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStreamingContent("");
    onStatusChange("processing");

    try {
      // Try streaming first, fall back to regular chat if stream endpoint unavailable
      const streamMsgId = msgId();
      let usedStream = false;

      try {
        const finalMessage = await chatStream(
          trimmed,
          playerId,
          (chunk) => {
            usedStream = true;
            setStreamingContent((prev) => prev + chunk);
          }
        );
        // Stream completed -- add final message and clear streaming state
        setStreamingContent("");
        setMessages((prev) => [...prev, { ...finalMessage, id: streamMsgId }]);
      } catch {
        // Stream endpoint not available -- fall back to regular chat
        if (!usedStream) {
          setStreamingContent("");
          const response = await chatFallback(trimmed, playerId);
          setMessages((prev) => [...prev, response.message]);
        } else {
          // Stream was partially received then failed
          throw new Error("Stream interrupted");
        }
      }

      onStatusChange("online");
    } catch (error) {
      setStreamingContent("");
      const errorMsg =
        error instanceof Error ? error.message : "Unknown error";
      const fallback: ChatMessage = {
        id: msgId(),
        role: "assistant",
        content: `NOT CONNECTED -- API error: ${errorMsg}. Ensure the backend is running.`,
        type: "text",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, fallback]);
      onStatusChange("offline");
    } finally {
      setIsLoading(false);
      // Return focus to input after sending (C6: focus management)
      inputRef.current?.focus();
    }
  }, [input, isLoading, playerId, onStatusChange]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    // Shift+Enter allows newline (default behavior)
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages area with accessibility: role="log" and aria-live */}
      <div
        className="scrollbar-thin flex-1 overflow-y-auto px-6 py-4"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Streaming response in progress */}
          {isLoading && streamingContent && (
            <div className={clsx("flex gap-3", "flex-row")}>
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-hs-gold/15">
                <Bot className="h-4 w-4 text-hs-gold" aria-hidden="true" />
              </div>
              <div className="max-w-[75%] rounded-xl px-4 py-3 card-surface text-hs-text-secondary">
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{streamingContent}</p>
              </div>
            </div>
          )}

          {/* Thinking indicator when no stream content yet */}
          {isLoading && !streamingContent && (
            <div className="flex items-center gap-2 text-xs text-hs-text-muted" aria-live="polite">
              <Bot className="h-4 w-4 animate-pulse text-hs-gold" aria-hidden="true" />
              <span>Hey Seven is thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <div className="border-t border-hs-border bg-hs-cream px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-3">
          <label htmlFor="chat-input" className="sr-only">
            Type your message
          </label>
          <textarea
            id="chat-input"
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message the AI Host..."
            rows={1}
            disabled={isLoading}
            aria-label="Type your message"
            className={clsx(
              "flex-1 resize-none rounded-lg border border-hs-border bg-hs-elevated px-4 py-3",
              "text-sm text-hs-dark placeholder:text-hs-text-muted",
              "focus:border-hs-gold/50 focus:outline-none focus:ring-1 focus:ring-hs-gold/30",
              "disabled:opacity-50",
              "max-h-32"
            )}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
            className={clsx(
              "flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-lg",
              "bg-gradient-to-r from-hs-gold-light to-hs-gold text-white transition-colors",
              "hover:opacity-90 disabled:opacity-40"
            )}
          >
            <Send className="h-4 w-4" aria-hidden="true" />
          </button>
        </div>
        <p className="mx-auto mt-1.5 max-w-3xl text-[10px] text-hs-text-muted">
          Press Enter to send, Shift+Enter for a new line
        </p>
      </div>
    </div>
  );
}

/* ---- Message rendering ---- */

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={clsx("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {/* Avatar */}
      <div
        className={clsx(
          "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-hs-dark/10" : "bg-hs-gold/15"
        )}
        aria-hidden="true"
      >
        {isUser ? (
          <User className="h-4 w-4 text-hs-text-secondary" />
        ) : (
          <Bot className="h-4 w-4 text-hs-gold" />
        )}
      </div>

      {/* Content */}
      <div
        className={clsx(
          "max-w-[75%] rounded-xl px-4 py-3",
          isUser
            ? "bg-hs-dark text-white"
            : "card-surface text-hs-text-secondary"
        )}
      >
        <SpecializedContent message={message} />
        <time
          className={clsx(
            "mt-1.5 block text-[10px]",
            isUser ? "text-white/50" : "text-hs-text-muted"
          )}
          dateTime={message.timestamp.toISOString()}
        >
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </time>
      </div>
    </div>
  );
}

/** Renders specialized content blocks based on message type */
function SpecializedContent({ message }: { message: ChatMessage }) {
  switch (message.type) {
    case "comp_offer":
      return <CompOfferCard content={message.content} metadata={message.metadata as CompOfferMetadata | undefined} />;
    case "reservation_confirmation":
      return <ReservationCard content={message.content} metadata={message.metadata as ReservationMetadata | undefined} />;
    case "escalation_notice":
      return <EscalationCard content={message.content} metadata={message.metadata as EscalationMetadata | undefined} />;
    default:
      return <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>;
  }
}

function CompOfferCard({ content, metadata }: { content: string; metadata?: CompOfferMetadata }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-hs-gold">
        <Gift className="h-4 w-4" aria-hidden="true" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Comp Offer
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      {metadata && (
        <div className="mt-1 rounded-md bg-hs-gold/10 px-3 py-2 text-xs text-hs-gold">
          {metadata.compType} -- {metadata.currency}
          {metadata.value.toLocaleString()}
          {metadata.expiresAt && ` (expires ${metadata.expiresAt})`}
        </div>
      )}
    </div>
  );
}

function ReservationCard({ content, metadata }: { content: string; metadata?: ReservationMetadata }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-emerald-600">
        <CalendarCheck className="h-4 w-4" aria-hidden="true" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Reservation Confirmed
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      {metadata && (
        <div className="mt-1 rounded-md bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
          {metadata.venue} -- {metadata.date} at {metadata.time} -- Party of{" "}
          {metadata.partySize}
          <br />
          Confirmation: {metadata.confirmationId}
        </div>
      )}
    </div>
  );
}

function EscalationCard({ content, metadata }: { content: string; metadata?: EscalationMetadata }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-amber-600">
        <AlertTriangle className="h-4 w-4" aria-hidden="true" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Escalation Notice
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      {metadata && (
        <div className="mt-1 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Reason: {metadata.reason} -- Priority: {metadata.priority} -- Assigned
          to: {metadata.assignedTo}
        </div>
      )}
    </div>
  );
}
