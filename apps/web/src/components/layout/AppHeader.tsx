import Link from "next/link";
import { BackendStatus } from "@/features/system/BackendStatus";

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
      <rect x="1.5" y="5.5" width="29" height="21" rx="3" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <line x1="16" y1="5.5" x2="16" y2="26.5" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="16" cy="16" r="4" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <rect x="1.5" y="11" width="4.5" height="10" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <rect x="26" y="11" width="4.5" height="10" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-slate-900 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-6 px-6">
        <Link href="/" className="flex items-center gap-2.5 text-cyan-400">
          <Logo />
          <span className="text-sm font-semibold tracking-tight text-slate-100">
            SoccerVision <span className="text-cyan-400">3D</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          <Link href="/" className="rounded-md px-3 py-1.5 text-slate-300 hover:bg-slate-800 hover:text-slate-100">
            Dashboard
          </Link>
          <Link
            href="/upload"
            className="rounded-md px-3 py-1.5 text-slate-300 hover:bg-slate-800 hover:text-slate-100"
          >
            Upload
          </Link>
        </nav>
        <BackendStatus />
      </div>
    </header>
  );
}
