"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type ChatMessage, streamChat } from "@/lib/chat";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    setInput("");
    setSources([]);
    const userMsg: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);

    // Add empty assistant message for streaming
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    abortRef.current = new AbortController();

    try {
      const newThreadId = await streamChat(
        trimmed,
        threadId,
        {
          onToken: (token) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + token,
                };
              }
              return updated;
            });
          },
          onMetadata: (tid) => setThreadId(tid),
          onSources: (s) => setSources(s),
          onDone: () => setIsStreaming(false),
          onError: (err) => {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              if (last?.role === "assistant") {
                updated[updated.length - 1] = {
                  ...last,
                  content: `Sorry, something went wrong: ${err}`,
                };
              }
              return updated;
            });
            setIsStreaming(false);
          },
        },
        abortRef.current.signal
      );
      if (newThreadId) setThreadId(newThreadId);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setIsStreaming(false);
    }
  }, [input, isStreaming, threadId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    setThreadId(null);
    setSources([]);
    setIsStreaming(false);
    setInput("");
    inputRef.current?.focus();
  };

  return (
    <div className="flex h-screen flex-col bg-cream">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-warm-gray px-6 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gold text-white font-semibold text-sm">
            7
          </div>
          <div>
            <h1
              className="text-lg font-semibold text-dark-brown"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              Hey Seven
            </h1>
            <p className="text-xs text-dark-brown-light">
              AI Casino Host
            </p>
          </div>
        </div>
        <button
          onClick={handleNewChat}
          className="rounded-lg border border-warm-gray px-3 py-1.5 text-sm text-dark-brown-light hover:bg-warm-gray transition-colors"
        >
          New Chat
        </button>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-2xl space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gold/10 text-gold text-2xl font-bold">
                7
              </div>
              <h2
                className="mb-2 text-2xl font-semibold text-dark-brown"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Welcome to Hey Seven
              </h2>
              <p className="max-w-md text-dark-brown-light">
                Your AI casino host. Ask about dining, shows,
                hotel rooms, rewards, or anything else at the property.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {[
                  "What restaurants do you have?",
                  "Any shows tonight?",
                  "Tell me about the spa",
                  "What rewards can I earn?",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => {
                      setInput(q);
                      inputRef.current?.focus();
                    }}
                    className="rounded-full border border-gold/30 px-3 py-1.5 text-sm text-dark-brown hover:bg-gold/10 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed ${
                  msg.role === "user"
                    ? "bg-dark-brown text-cream rounded-br-md"
                    : "bg-white text-dark-brown border border-warm-gray rounded-bl-md shadow-sm"
                }`}
              >
                {msg.role === "assistant" && msg.content === "" && isStreaming ? (
                  <div className="flex gap-1 py-1">
                    <span className="typing-dot h-2 w-2 rounded-full bg-gold" />
                    <span className="typing-dot h-2 w-2 rounded-full bg-gold" />
                    <span className="typing-dot h-2 w-2 rounded-full bg-gold" />
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {/* Sources badge */}
          {sources.length > 0 && !isStreaming && (
            <div className="flex justify-start">
              <div className="flex flex-wrap gap-1">
                {sources.map((s) => (
                  <span
                    key={s}
                    className="inline-block rounded-full bg-gold/10 px-2.5 py-0.5 text-xs text-gold-light"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="border-t border-warm-gray bg-white px-4 py-3">
        <div className="mx-auto flex max-w-2xl items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about dining, shows, rooms, rewards..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-warm-gray bg-cream px-4 py-2.5 text-[15px] text-dark-brown placeholder:text-dark-brown-light/50 focus:border-gold focus:outline-none transition-colors"
            style={{ maxHeight: "120px" }}
            disabled={isStreaming}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gold text-white disabled:opacity-40 hover:bg-gold-light transition-colors"
            aria-label="Send message"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="h-5 w-5"
            >
              <path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" />
            </svg>
          </button>
        </div>
        {threadId && (
          <p className="mx-auto mt-1 max-w-2xl text-right text-[10px] text-dark-brown-light/40">
            Thread: {threadId.slice(0, 8)}...
          </p>
        )}
      </footer>
    </div>
  );
}
