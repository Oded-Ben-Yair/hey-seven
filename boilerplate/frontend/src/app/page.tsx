import { Suspense } from "react";
import type { PlayerProfile } from "@/lib/types";
import { PlayerPanel } from "@/components/PlayerPanel";
import { HomeClient } from "@/components/HomeClient";

/**
 * Demo player data for development.
 * In production, this would be fetched server-side from the API.
 * Clearly labeled as DEMO data.
 */
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

/**
 * Server-side data fetch for player profile.
 * Currently returns DEMO data; replace with real API call in production.
 */
async function getPlayerData(): Promise<PlayerProfile> {
  // In production: return await getPlayer("PLR-88421");
  return DEMO_PLAYER;
}

/** Loading skeleton for the player sidebar */
function PlayerPanelSkeleton() {
  return (
    <div className="flex flex-col gap-5 p-5 animate-pulse" aria-label="Loading player data">
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-full bg-hs-warm-gray" />
        <div className="flex flex-col gap-1.5">
          <div className="h-4 w-28 rounded bg-hs-warm-gray" />
          <div className="h-3 w-20 rounded bg-hs-warm-gray" />
        </div>
      </div>
      <div className="h-10 rounded-lg bg-hs-warm-gray" />
      <div className="grid grid-cols-2 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 rounded-lg bg-hs-warm-gray" />
        ))}
      </div>
    </div>
  );
}

/**
 * Home page -- Server Component.
 * Fetches player data on the server, renders PlayerPanel as a server component,
 * and wraps the interactive ChatInterface in a client boundary (HomeClient).
 */
export default async function HomePage() {
  const player = await getPlayerData();

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      {/* Chat area with header -- client boundary for interactivity */}
      <HomeClient playerId={player.id}>
        {/* Player info sidebar -- rendered server-side, passed as children */}
        <aside
          className="hidden w-80 flex-shrink-0 overflow-y-auto border-r border-hs-border bg-hs-cream lg:block"
          aria-label="Player information"
        >
          <Suspense fallback={<PlayerPanelSkeleton />}>
            <PlayerPanel player={player} />
          </Suspense>
        </aside>
      </HomeClient>
    </div>
  );
}
