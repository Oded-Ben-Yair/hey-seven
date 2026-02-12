"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-screen items-center justify-center bg-hs-cream">
      <div className="mx-auto max-w-md text-center">
        <h2 className="font-serif text-xl font-semibold text-hs-dark">
          Something went wrong
        </h2>
        <p className="mt-2 text-sm text-hs-text-secondary">
          {error.message || "An unexpected error occurred."}
        </p>
        <button
          onClick={reset}
          className="mt-6 rounded-lg bg-hs-gold px-6 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
