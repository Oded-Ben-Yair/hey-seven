import { Dice5, Wifi, Loader2, WifiOff } from "lucide-react";
import clsx from "clsx";
import type { ConnectionStatus } from "@/lib/types";

interface HeaderProps {
  status: ConnectionStatus;
}

const STATUS_CONFIG: Record<
  ConnectionStatus,
  { label: string; color: string; icon: typeof Wifi }
> = {
  online: { label: "Online", color: "bg-emerald-500", icon: Wifi },
  processing: { label: "Processing", color: "bg-hs-gold", icon: Loader2 },
  offline: { label: "Offline", color: "bg-red-500", icon: WifiOff },
};

export function Header({ status }: HeaderProps) {
  const statusInfo = STATUS_CONFIG[status];
  const StatusIcon = statusInfo.icon;

  return (
    <nav
      aria-label="Main navigation"
      className="flex h-14 flex-shrink-0 items-center justify-between border-b border-hs-border bg-hs-dark px-6"
    >
      {/* Logo + title */}
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-hs-gold/20">
          <Dice5 className="h-5 w-5 text-hs-gold" aria-hidden="true" />
        </div>
        <div className="flex items-baseline gap-2">
          <h1 className="font-serif text-lg font-semibold tracking-tight text-white">
            Hey Seven
          </h1>
          <span className="text-xs font-medium text-white/60">
            Pulse
          </span>
        </div>
      </div>

      {/* Status indicator */}
      <div
        className="flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5"
        role="status"
        aria-live="polite"
        aria-label={`Connection status: ${statusInfo.label}`}
      >
        <StatusIcon
          className={clsx(
            "h-3.5 w-3.5",
            status === "processing" && "animate-spin",
            status === "online" && "text-emerald-400",
            status === "processing" && "text-hs-gold",
            status === "offline" && "text-red-400"
          )}
          aria-hidden="true"
        />
        <span className="text-xs font-medium text-white/70">
          {statusInfo.label}
        </span>
        <span
          className={clsx(
            "h-2 w-2 rounded-full",
            statusInfo.color,
            status === "online" && "animate-pulse"
          )}
          aria-hidden="true"
        />
      </div>
    </nav>
  );
}
