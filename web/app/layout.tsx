import type { Metadata } from "next";

import Footer from "@/components/Footer";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "PERM Predictor — Processing Time & Outcome",
  description:
    "Explainable, fairness-audited machine learning for U.S. PERM labour certification: predicted processing time, case outcome, and per-case SHAP explanations.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col">
        <Nav />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
