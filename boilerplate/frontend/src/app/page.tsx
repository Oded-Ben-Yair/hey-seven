"use client";

import { useState } from "react";
import { Header } from "@/components/Header";
import { ChatInterface } from "@/components/ChatInterface";
import { PlayerPanel } from "@/components/PlayerPanel";
import type { ConnectionStatus, PlayerProfile } from "@/lib/types";

/** Placeholder player data for development. Replace with real API call. */
const DEMO_PLAYER: PlayerProfile = {
  id: "PLR-88421",
  name: "Michael Chen",
  tier: "Platinum",
  photoUrl: null,
  adt: 1250,
  compBalance: 3400,
  lastVisit: "2025-01-28",
  totalVisits: 47,
  memberSince: "2022-03-15",
  preferences: [
    "Blackjack (6-deck, $100 min)",
    "Steakhouse reservations",
    "Corner suite, high floor",
    "Prefers morning play sessions",
    "No shellfish (allergy)",
  ],
  recentActivity: [
    {
      date: "2025-01-28",
      duration: "4h 20m",
      spend: 5200,
      primaryGame: "Blackjack",
    },
    {
      date: "2025-01-15",
      duration: "6h 10m",
      spend: 8400,
      primaryGame: "Blackjack",
    },
    {
      date: "2024-12-31",
      duration: "8h 45m",
      spend: 12100,
      primaryGame: "Baccarat",
    },
  ],
};

export default function HomePage() {
  const [status, setStatus] = useState<ConnectionStatus>("online");
  const [player] = useState<PlayerProfile>(DEMO_PLAYER);

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header status={status} />

      <main className="flex flex-1 overflow-hidden">
        {/* Player info sidebar */}
        <aside className="hidden w-80 flex-shrink-0 overflow-y-auto border-r border-hs-border bg-hs-dark lg:block">
          <PlayerPanel player={player} />
        </aside>

        {/* Chat area */}
        <section className="flex flex-1 flex-col overflow-hidden">
          <ChatInterface
            playerId={player.id}
            onStatusChange={setStatus}
          />
        </section>
      </main>
    </div>
  );
}
