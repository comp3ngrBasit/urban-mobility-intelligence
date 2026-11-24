import type { ReactNode } from "react";
import { ApiError, type ApiState } from "../lib/api";
import { fmtTs } from "../lib/format";

export function Card({ title, subtitle, right, children, className = "" }: {
  title?: string; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`panel p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-sm font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, accent = "text-slate-50" }: { label: string; value: ReactNode; sub?: ReactNode; accent?: string }) {
  return (
    <div className="panel px-4 py-3.5">
      <div className="label">{label}</div>
      <div className={`num mt-1.5 text-2xl font-semibold ${accent}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
    </div>
  );
}

export function PageHeader({ title, description, right }: { title: string; description: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-white">{title}</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">{description}</p>
      </div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

export function Segmented<T extends string | number>({ value, options, onChange }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-ink-600 bg-ink-800 p-0.5">
      {options.map((o) => (
        <button key={String(o.value)} onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
            o.value === value ? "bg-electric-500/20 text-electric-400" : "text-slate-400 hover:text-slate-200"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

type Mode = "live" | "latest_available" | "historical" | "unavailable";
const MODE: Record<Mode, { label: string; cls: string }> = {
  live: { label: "LIVE", cls: "bg-signal-green/15 text-signal-green ring-signal-green/30" },
  latest_available: { label: "LATEST AVAILABLE", cls: "bg-signal-amber/15 text-signal-amber ring-signal-amber/30" },
  historical: { label: "HISTORICAL", cls: "bg-signal-violet/15 text-violet-300 ring-signal-violet/30" },
  unavailable: { label: "NO DATA", cls: "bg-slate-500/15 text-slate-400 ring-slate-500/30" },
};

export function FreshnessBadge({ mode, asOf }: { mode: Mode; asOf?: string | null }) {
  const m = MODE[mode];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wider ring-1 ${m.cls}`}>
      {mode === "live" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal-green" />}
      {m.label}
      {asOf && <span className="font-normal normal-case tracking-normal opacity-80">· {fmtTs(asOf)}</span>}
    </span>
  );
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  const cls = tone === "warn" ? "border-signal-amber/30 bg-signal-amber/5 text-amber-200" : "border-electric-500/25 bg-electric-500/5 text-slate-300";
  return <div className={`rounded-lg border px-3 py-2 text-xs ${cls}`}>{children}</div>;
}

export function EmptyState({ what, hint }: { what: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-ink-600 px-6 py-10 text-center">
      <div className="text-sm font-medium text-slate-300">No data yet for {what}</div>
      <p className="mt-1 text-xs text-slate-500">This view shows only computed results. Produce them with:</p>
      <code className="mt-3 rounded bg-ink-800 px-3 py-1.5 font-mono text-xs text-signal-cyan">{hint}</code>
    </div>
  );
}

export function Skeleton({ h = 240 }: { h?: number }) {
  return <div className="animate-pulse rounded-lg bg-ink-800/70" style={{ height: h }} />;
}

/** Standard loading / not-ready / error handling around any API-backed block. */
export function Async<T>({ state, children, h = 240 }: { state: ApiState<T>; children: (d: T) => ReactNode; h?: number }) {
  if (state.error) {
    const nr = state.error instanceof ApiError ? state.error.notReady : null;
    if (nr) return <EmptyState what={nr.what} hint={nr.hint} />;
    return (
      <div className="rounded-lg border border-signal-red/30 bg-signal-red/5 p-4 text-sm text-red-300">
        Request failed: {state.error.message}{" "}
        <button className="ml-2 underline" onClick={state.reload}>retry</button>
      </div>
    );
  }
  if (state.loading && state.data === undefined) return <Skeleton h={h} />;
  if (state.data === undefined) return null;
  return <>{children(state.data)}</>;
}
