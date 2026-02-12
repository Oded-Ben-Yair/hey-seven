"use client";

import { useState } from "react";
import { Header } from "@/components/Header";
import { ChatInterface } from "@/components/ChatInterface";
import type { ConnectionStatus } from "@/lib/types";

interface HomeClientProps {
  playerId: string;
  /** Server-rendered sidebar passed as children to preserve Server Component rendering */
  children: React.ReactNode;
}

/**
 * Client boundary for the home page.
 * Manages connection status state and renders the interactive chat interface.
 * Accepts server-rendered children (PlayerPanel sidebar) via the composition pattern.
 */
export function HomeClient({ playerId, children }: HomeClientProps) {
  const [status, setStatus] = useState<ConnectionStatus>("online");

  return (
    <>
      <Header status={status} />

      <main className="flex flex-1 overflow-hidden">
        {/* Server-rendered sidebar injected via children */}
        {children}

        {/* Chat area */}
        <section className="flex flex-1 flex-col overflow-hidden" aria-label="Chat with AI Host">
          <ChatInterface playerId={playerId} onStatusChange={setStatus} />
        </section>
      </main>
    </>
  );
}
