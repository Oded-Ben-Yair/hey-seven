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
  MessageType,
  ConnectionStatus,
  CompOfferMetadata,
  ReservationMetadata,
  EscalationMetadata,
} from "@/lib/types";
import { chat as chatApi } from "@/lib/api";

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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  /* Auto-scroll to bottom on new messages */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* Focus input on mount */
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

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
    onStatusChange("processing");

    try {
      const response = await chatApi(trimmed, playerId);
      setMessages((prev) => [...prev, response.message]);
      onStatusChange("online");
    } catch (error) {
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
    }
  }, [input, isLoading, playerId, onStatusChange]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="scrollbar-thin flex-1 overflow-y-auto px-6 py-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isLoading && (
            <div className="flex items-center gap-2 text-xs text-hs-text-muted">
              <Bot className="h-4 w-4 animate-pulse text-hs-gold" />
              <span>Hey Seven is thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-hs-border bg-white px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message the AI Host..."
            rows={1}
            disabled={isLoading}
            className={clsx(
              "flex-1 resize-none rounded-lg border border-hs-border bg-hs-elevated px-4 py-3",
              "text-sm text-hs-dark placeholder:text-hs-text-muted",
              "focus:border-hs-gold/50 focus:outline-none focus:ring-1 focus:ring-hs-gold/30",
              "disabled:opacity-50"
            )}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className={clsx(
              "flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-lg",
              "bg-gradient-to-r from-hs-gold-light to-hs-gold text-white transition-colors",
              "hover:opacity-90 disabled:opacity-40"
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
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
        <time className="mt-1.5 block text-[10px] text-hs-text-muted">
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
        <Gift className="h-4 w-4" />
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
      <div className="flex items-center gap-2 text-emerald-400">
        <CalendarCheck className="h-4 w-4" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Reservation Confirmed
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      {metadata && (
        <div className="mt-1 rounded-md bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
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
      <div className="flex items-center gap-2 text-amber-400">
        <AlertTriangle className="h-4 w-4" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Escalation Notice
        </span>
      </div>
      <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      {metadata && (
        <div className="mt-1 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          Reason: {metadata.reason} -- Priority: {metadata.priority} -- Assigned
          to: {metadata.assignedTo}
        </div>
      )}
    </div>
  );
}
