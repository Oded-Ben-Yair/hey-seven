import type { Metadata } from "next";
import { Playfair_Display } from "next/font/google";
import "./globals.css";

const playfair = Playfair_Display({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-playfair",
});

export const metadata: Metadata = {
  title: "Hey Seven Pulse - AI Casino Host",
  description:
    "The Autonomous Casino Host That Never Sleeps. AI-powered player relationship management for modern casinos.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={playfair.variable}>
      <body className="min-h-screen bg-hs-bg font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
