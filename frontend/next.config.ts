import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8080/api/:path*",
      },
      {
        source: "/chat",
        destination: "http://localhost:8080/chat",
      },
      {
        source: "/health",
        destination: "http://localhost:8080/health",
      },
      {
        source: "/live",
        destination: "http://localhost:8080/live",
      },
    ];
  },
};

export default nextConfig;
