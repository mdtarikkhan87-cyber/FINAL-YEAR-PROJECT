"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/predict", label: "Predict" },
  { href: "/about", label: "Methodology" },
];

type Status = "checking" | "online" | "offline";

export default function Nav() {
  const pathname = usePathname();
  const [status, setStatus] = useState<Status>("checking");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api
        .health()
        .then(() => !cancelled && setStatus("online"))
        .catch(() => !cancelled && setStatus("offline"));
    check();
    const timer = setInterval(check, 20_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const dot =
    status === "online"
      ? "bg-emerald-500"
      : status === "offline"
        ? "bg-red-500"
        : "bg-ink-300";

  return (
    <header className="sticky top-0 z-40 border-b border-ink-200 bg-white/85 backdrop-blur">
      <div className="container-page flex h-16 items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2.5" onClick={() => setOpen(false)}>
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-accent-600 text-sm font-bold text-white">
            P
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-ink-900">
            PERM Predictor
          </span>
        </Link>

        <nav className="hidden items-center gap-1 sm:flex">
          {LINKS.map((link) => {
            const active =
              link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                  active
                    ? "bg-accent-50 text-accent-700"
                    : "text-ink-600 hover:bg-ink-50 hover:text-ink-900"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-3">
          <span
            className="hidden items-center gap-2 text-xs text-ink-500 sm:flex"
            title={
              status === "offline"
                ? "Start the API with: uvicorn api.main:app --port 8000"
                : "FastAPI backend status"
            }
          >
            <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
            API {status}
          </span>
          <button
            type="button"
            aria-label="Toggle navigation"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="rounded-lg border border-ink-300 px-2.5 py-1.5 text-ink-700 sm:hidden"
          >
            <span aria-hidden>{open ? "✕" : "☰"}</span>
          </button>
        </div>
      </div>

      {open && (
        <nav className="border-t border-ink-200 bg-white sm:hidden">
          <div className="container-page flex flex-col py-2">
            {LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2.5 text-sm font-medium text-ink-700 hover:bg-ink-50"
              >
                {link.label}
              </Link>
            ))}
            <span className="flex items-center gap-2 px-3 py-2 text-xs text-ink-500">
              <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
              API {status}
            </span>
          </div>
        </nav>
      )}
    </header>
  );
}
