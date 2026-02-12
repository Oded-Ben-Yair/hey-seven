"use client";

import {
  User,
  Crown,
  DollarSign,
  Calendar,
  Star,
  Clock,
  Gamepad2,
} from "lucide-react";
import clsx from "clsx";
import type { PlayerProfile, PlayerTier } from "@/lib/types";

interface PlayerPanelProps {
  player: PlayerProfile;
}

const TIER_STYLES: Record<PlayerTier, { bg: string; text: string; border: string }> = {
  Gold: {
    bg: "bg-yellow-500/10",
    text: "text-yellow-400",
    border: "border-yellow-500/30",
  },
  Platinum: {
    bg: "bg-slate-300/10",
    text: "text-slate-300",
    border: "border-slate-300/30",
  },
  Diamond: {
    bg: "bg-cyan-400/10",
    text: "text-cyan-300",
    border: "border-cyan-400/30",
  },
  "Seven Star": {
    bg: "bg-hs-red/10",
    text: "text-hs-red",
    border: "border-hs-red/30",
  },
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function PlayerPanel({ player }: PlayerPanelProps) {
  const tierStyle = TIER_STYLES[player.tier];

  return (
    <div className="flex flex-col gap-5 p-5">
      {/* Player identity */}
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-hs-elevated">
          {player.photoUrl ? (
            <img
              src={player.photoUrl}
              alt={player.name}
              className="h-12 w-12 rounded-full object-cover"
            />
          ) : (
            <User className="h-6 w-6 text-hs-text-secondary" />
          )}
        </div>
        <div>
          <p className="text-sm font-semibold text-white">{player.name}</p>
          <p className="text-xs text-hs-text-muted">ID: {player.id}</p>
        </div>
      </div>

      {/* Tier badge */}
      <div
        className={clsx(
          "card-surface flex items-center gap-2 rounded-lg px-3 py-2",
          tierStyle.border
        )}
      >
        <Crown className={clsx("h-4 w-4", tierStyle.text)} />
        <span className={clsx("text-sm font-semibold", tierStyle.text)}>
          {player.tier}
        </span>
        <span className="text-xs text-hs-text-muted">
          since {formatDate(player.memberSince)}
        </span>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          icon={<DollarSign className="h-4 w-4 text-hs-gold" />}
          label="ADT"
          value={formatCurrency(player.adt)}
        />
        <MetricCard
          icon={<Star className="h-4 w-4 text-hs-red" />}
          label="Comp Balance"
          value={formatCurrency(player.compBalance)}
        />
        <MetricCard
          icon={<Calendar className="h-4 w-4 text-hs-text-secondary" />}
          label="Last Visit"
          value={formatDate(player.lastVisit)}
        />
        <MetricCard
          icon={<Clock className="h-4 w-4 text-hs-text-secondary" />}
          label="Total Visits"
          value={String(player.totalVisits)}
        />
      </div>

      {/* Preferences */}
      <Section title="Player Preferences">
        <ul className="flex flex-col gap-1.5">
          {player.preferences.map((pref) => (
            <li
              key={pref}
              className="flex items-start gap-2 text-xs text-hs-text-secondary"
            >
              <span className="mt-1 h-1 w-1 flex-shrink-0 rounded-full bg-hs-gold" />
              {pref}
            </li>
          ))}
        </ul>
      </Section>

      {/* Recent visits */}
      <Section title="Recent Activity">
        <div className="flex flex-col gap-2">
          {player.recentActivity.map((visit) => (
            <div
              key={visit.date}
              className="card-surface flex items-center justify-between rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <Gamepad2 className="h-3.5 w-3.5 text-hs-text-muted" />
                <div>
                  <p className="text-xs font-medium text-white">
                    {visit.primaryGame}
                  </p>
                  <p className="text-[11px] text-hs-text-muted">
                    {formatDate(visit.date)} -- {visit.duration}
                  </p>
                </div>
              </div>
              <span className="text-xs font-medium text-hs-gold">
                {formatCurrency(visit.spend)}
              </span>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

/* ---- Sub-components ---- */

function MetricCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="card-surface flex flex-col gap-1 rounded-lg p-3">
      <div className="flex items-center gap-1.5">
        {icon}
        <span className="text-[11px] text-hs-text-muted">{label}</span>
      </div>
      <span className="text-sm font-semibold text-white">{value}</span>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-hs-text-muted">
        {title}
      </h3>
      {children}
    </div>
  );
}
