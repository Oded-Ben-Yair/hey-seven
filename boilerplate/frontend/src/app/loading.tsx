import { Dice5 } from "lucide-react";

export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-hs-cream">
      <div className="flex flex-col items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-hs-gold/20">
          <Dice5 className="h-7 w-7 animate-pulse text-hs-gold" />
        </div>
        <p className="font-serif text-sm text-hs-text-muted">
          Loading Hey Seven Pulse...
        </p>
      </div>
    </div>
  );
}
