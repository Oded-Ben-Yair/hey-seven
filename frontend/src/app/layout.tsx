import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hey Seven — AI Casino Host",
  description: "The Autonomous Casino Host That Never Sleeps",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-cream text-deep-black">{children}</body>
    </html>
  );
}
