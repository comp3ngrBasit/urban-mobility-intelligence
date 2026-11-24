import { useState } from "react";
import { Async, Card, PageHeader } from "../components/ui";
import { qs, useApi } from "../lib/api";
import { fmt, fmtInt, fmtTs } from "../lib/format";

interface Status {
  sources: { source: string; last_status: string; last_run: string; last_success: string | null; data_max_ts: string | null; records_last_success: number; total_records: number; last_error: string | null; runs: number; failures: number }[];
  verification: { source: string; status: string; dataset_id?: string; endpoint: string; checked_at: string; rows_updated_at?: string; deprecated?: boolean; is_live?: boolean; last_observation?: string; missing_columns?: string[]; error?: string }[] | null;
}
interface Quality { dataset: string; checked_at: string; rows: number; rows_after_cleaning?: number; duplicates?: number; invalid_coordinates?: number; missing_coordinates?: number; issues?: string[]; outliers?: Record<string, number>; columns?: Record<string, { missing_pct: number; out_of_range?: number }>; [k: string]: unknown }
interface Run { source: string; started_at: string; status: string; records: number; duration_s: number; error: string | null }

const dot = (s: string) => ({ success: "bg-signal-green", ok: "bg-signal-green", partial: "bg-signal-amber", failed: "bg-signal-red", unreachable: "bg-signal-red", error: "bg-signal-red", running: "bg-electric-500" }[s] ?? "bg-slate-500");

export default function Pipeline() {
  const st = useApi<Status>("/api/pipeline/status", 60_000);
  const q = useApi<Quality[]>("/api/pipeline/quality");
  const [src, setSrc] = useState("");
  const runs = useApi<Run[]>(`/api/pipeline/runs${qs({ limit: 50, source: src })}`, 60_000);
  return (
    <>
      <PageHeader title="Data & Pipeline Monitoring" description="Ingestion ledger, source verification and data-quality reports. When a source is unavailable the platform keeps serving the last successful ingestion, labelled with its real timestamp." />
      <div className="space-y-5">
        <Card title="Sources" subtitle="Status of the most recent run and freshness of the last successful one">
          <Async state={st} h={200}>{(d) => (
            <div className="overflow-auto"><table className="data"><thead><tr><th>Source</th><th>Last run</th><th>Last success</th><th>Data up to</th><th className="text-right">Records (last)</th><th className="text-right">Runs / failures</th><th>Last error</th></tr></thead>
              <tbody>{d.sources.map((s) => <tr key={s.source} className="cursor-pointer hover:bg-ink-800" onClick={() => setSrc(s.source)}>
                <td><span className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${dot(s.last_status)}`} />{s.source}</span></td>
                <td className="text-xs">{fmtTs(s.last_run)}</td><td className="text-xs">{fmtTs(s.last_success)}</td><td className="text-xs">{fmtTs(s.data_max_ts)}</td>
                <td className="num text-right">{fmtInt(s.records_last_success)}</td><td className="num text-right">{s.runs} / <span className={s.failures ? "text-signal-red" : ""}>{s.failures}</span></td>
                <td className="max-w-[280px] truncate text-xs text-red-300" title={s.last_error ?? ""}>{s.last_error ?? ""}</td></tr>)}
                {!d.sources.length && <tr><td colSpan={7} className="text-slate-500">No ingestion runs recorded yet — run <code>python -m src.pipeline ingest</code>.</td></tr>}</tbody></table></div>)}</Async>
        </Card>
        {st.data?.verification && (
          <Card title="API verification" subtitle="Endpoint reachability, schema and publisher-reported freshness (python -m src.pipeline verify)">
            <table className="data"><thead><tr><th>Source</th><th>Endpoint</th><th>Publisher updated</th><th>Last observation</th><th>Flags</th></tr></thead>
              <tbody>{st.data.verification.map((v) => <tr key={v.source}>
                <td><span className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${dot(v.status)}`} />{v.source}</span></td>
                <td className="font-mono text-[11px] text-slate-400">{v.endpoint}</td><td className="text-xs">{fmtTs(v.rows_updated_at)}</td><td className="text-xs">{fmtTs(v.last_observation)}</td>
                <td className="text-xs">{v.deprecated && <span className="mr-2 rounded bg-signal-amber/15 px-1.5 text-signal-amber">deprecated</span>}{v.is_live && <span className="mr-2 rounded bg-signal-green/15 px-1.5 text-signal-green">live</span>}{v.missing_columns?.length ? `missing: ${v.missing_columns.join(", ")}` : ""}{v.error}</td></tr>)}</tbody></table>
          </Card>)}
        <Card title="Data quality" subtitle="Validation report produced by each cleaner">
          <Async state={q} h={200}>{(qs_) => (
            <div className="grid gap-4 lg:grid-cols-2">{qs_.map((r) => (
              <div key={r.dataset} className="rounded-lg border border-ink-700 bg-ink-800/50 p-3">
                <div className="flex items-center justify-between"><span className="font-medium text-white">{r.dataset}</span><span className="text-xs text-slate-500">{fmtTs(r.checked_at)}</span></div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div><div className="label">Raw rows</div><div className="num">{fmtInt(r.rows)}</div></div>
                  <div><div className="label">After cleaning</div><div className="num">{fmtInt(r.rows_after_cleaning)}</div></div>
                  <div><div className="label">Duplicates</div><div className="num">{fmtInt(r.duplicates)}</div></div>
                  {r.invalid_coordinates != null && <div><div className="label">Bad coords</div><div className="num">{fmtInt(r.invalid_coordinates)}</div></div>}
                  {r.missing_coordinates != null && <div><div className="label">Missing coords</div><div className="num">{fmtInt(r.missing_coordinates)}</div></div>}
                  {r.outliers && Object.entries(r.outliers).map(([k, v]) => <div key={k}><div className="label">Outliers · {k}</div><div className="num">{fmtInt(v)}</div></div>)}
                </div>
                {r.columns && <div className="mt-2 flex flex-wrap gap-1">{Object.entries(r.columns).filter(([, c]) => c.missing_pct > 0).map(([k, c]) => <span key={k} className="rounded bg-ink-700 px-1.5 py-0.5 text-[10px] text-slate-400">{k}: {fmt(c.missing_pct, 1)}% missing</span>)}</div>}
                {r.issues?.length ? <ul className="mt-2 list-disc pl-4 text-xs text-signal-amber">{r.issues.map((i) => <li key={i}>{i}</li>)}</ul> : null}
              </div>))}</div>)}</Async>
        </Card>
        <Card title="Recent runs" subtitle={src ? `Filtered to ${src}` : "All sources"} right={src && <button className="text-xs text-slate-400 hover:text-white" onClick={() => setSrc("")}>clear filter</button>}>
          <Async state={runs} h={200}>{(rs) => (
            <div className="max-h-[400px] overflow-auto"><table className="data"><thead><tr><th>Started</th><th>Source</th><th>Status</th><th className="text-right">Records</th><th className="text-right">Duration</th><th>Error</th></tr></thead>
              <tbody>{rs.map((r, i) => <tr key={i}><td className="text-xs">{fmtTs(r.started_at)}</td><td>{r.source}</td><td><span className="flex items-center gap-2"><span className={`h-2 w-2 rounded-full ${dot(r.status)}`} />{r.status}</span></td>
                <td className="num text-right">{fmtInt(r.records)}</td><td className="num text-right">{fmt(r.duration_s, 1)} s</td><td className="max-w-[320px] truncate text-xs text-red-300">{r.error}</td></tr>)}</tbody></table></div>)}</Async>
        </Card>
      </div>
    </>
  );
}
