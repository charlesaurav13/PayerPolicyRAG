import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PayerPolicy RAG — Access Score Analyzer",
  description: "Extract and score prior authorization parameters from payer policy PDFs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col antialiased">{children}</body>
    </html>
  );
}
