import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost";

const buttonBase =
  "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-400 " +
  "disabled:cursor-not-allowed disabled:opacity-50";

const buttonVariants: Record<ButtonVariant, string> = {
  primary: "bg-cyan-400 text-slate-950 hover:bg-cyan-300",
  secondary: "border border-slate-700 bg-slate-900 text-slate-100 hover:border-slate-500 hover:bg-slate-800",
  ghost: "text-slate-300 hover:bg-slate-800 hover:text-slate-100",
};

export function Button({
  variant = "primary",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  return <button className={cx(buttonBase, buttonVariants[variant], className)} {...props} />;
}

export function ButtonLink({
  variant = "primary",
  className,
  ...props
}: ComponentProps<typeof Link> & { variant?: ButtonVariant }) {
  return <Link className={cx(buttonBase, buttonVariants[variant], className)} {...props} />;
}

export function Card({ className, ...props }: ComponentProps<"section">) {
  return <section className={cx("rounded-xl border border-slate-800 bg-slate-900/60 p-5", className)} {...props} />;
}

export function CardHeader({ title, subtitle, action }: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div>
        <h2 className="text-sm font-semibold tracking-wide text-slate-100">{title}</h2>
        {subtitle ? <p className="mt-1 text-xs text-slate-400">{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

type Tone = "neutral" | "info" | "success" | "warning" | "danger";

const toneClasses: Record<Tone, string> = {
  neutral: "border-slate-700 bg-slate-800/60 text-slate-300",
  info: "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
  success: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  warning: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  danger: "border-rose-500/40 bg-rose-500/10 text-rose-300",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        toneClasses[tone],
      )}
    >
      {children}
    </span>
  );
}

const statusTone: Record<string, Tone> = {
  uploaded: "neutral",
  queued: "info",
  pending: "info",
  processing: "info",
  completed: "success",
  failed: "danger",
};

export function StatusBadge({ status }: { status: string }) {
  const active = status === "queued" || status === "pending" || status === "processing";
  return (
    <Badge tone={statusTone[status] ?? "neutral"}>
      {active ? <span className="size-1.5 animate-pulse rounded-full bg-current" aria-hidden /> : null}
      {status}
    </Badge>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label={label}
      className="h-2 w-full overflow-hidden rounded-full bg-slate-800"
    >
      <div className="h-full rounded-full bg-cyan-400 transition-[width] duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
      <dt className="text-xs uppercase tracking-wider text-slate-500">{label}</dt>
      <dd className="mt-1 font-mono text-xl tabular-nums text-slate-100">{value}</dd>
      {hint ? <p className="mt-1 text-xs text-slate-500">{hint}</p> : null}
    </div>
  );
}

export function Alert({ tone = "danger", title, children }: { tone?: Tone; title: string; children?: ReactNode }) {
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={cx("rounded-lg border p-4 text-sm", toneClasses[tone])}>
      <p className="font-semibold">{title}</p>
      {children ? <div className="mt-1 text-slate-300">{children}</div> : null}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx("inline-block size-4 animate-spin rounded-full border-2 border-slate-600 border-t-cyan-400", className)}
    />
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx("animate-pulse rounded-md bg-slate-800/70", className)} />;
}

export function DefinitionList({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
      {items.map(([term, value]) => (
        <div key={term} className="contents">
          <dt className="text-slate-500">{term}</dt>
          <dd className="truncate font-mono text-slate-200">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
