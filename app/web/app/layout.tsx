import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BridgeSight — Inspection Studio",
  description: "Local-first bridge inspection photo review and SAM analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
