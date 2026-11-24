import { useMemo, useState } from "react";
import { DowHourHeatmap, HEAT_SCALE, HourlyCI } from "../components/charts";
import MapView, { CENTERS, fc, kv, line } from "../components/MapView";
import Plot from "../components/Plot";
import { Async, Card, FreshnessBadge, Notice, PageHeader, Segmented, Stat } from "../components/ui";
import { qs, useApi } from "../lib/api";
import { fmt, fmtInt, fmtTs, LEVEL_COLORS, ratioColor } from "../lib/format";
import type { CI, Freshness, Heatmap, Segment, TwoGroup } from "../lib/types";

interface NycLink { id: string; link_name: string; borough: string; speed: number; travel_time: number; data_as_of: string; typical_speed: number | null; relative_speed: number | null; anomaly: boolean; path?: [number, number][] | null }
interface NycLive { freshness: Freshness; links: NycLink[]; summary: { links: number; median_speed_mph: number; slow_links: number } }
interface ChiLive { freshness: Freshness; segments: (Segment & { speed: number; time: string; congestion_level?: string; free_flow_speed?: number })[] }
interface Anoms { city: string; mode: string; summary: { rows_scored: number; anomalies: number; rate: number; window_end: string } | null; anomalies: Record<string, unknown>[] }

export default function LiveTraffic() {
  const [city, setCity] = useState<"nyc" | "chicago">("nyc");
  const [borough, setBorough] = useState("");
  const [link, setLink] = useState<string | null>(null);
  const nyc = useApi<NycLive>(city === "nyc" ? "/api/live/nyc" : null, 60_000);
  const chi = useApi<ChiLive>(city === "chicago" ? "/api/live/chicago" : null);
  const an = useApi<Anoms>(`/api/live/anomalies${qs({ city, limit: 100 })}`);
  const hist = useApi<{ link_name: string; series: { data_as_of: string; speed: number }[] }>(link ? `/api/live/nyc/${link}/history?hours=72` : null);

  const nycLayer = useMemo(() => {
    const links = (nyc.data?.links ?? []).filter((l) => l.path && (!borough || l.borough === borough));
    return [{
      id: "nyc-links",
      data: fc(links.map((l) => line(l.path!, { id: l.id, name: l.link_name, speed: l.speed, typical: l.typical_speed, ratio: l.relative_speed, color: ratioColor(l.relative_speed), anomaly: l.anomaly }))),
      layers: [
        { id: "nyc-links-glow", type: "line", paint: { "line-color": ["get", "color"], "line-width": 7, "line-opacity": 0.15 } },
        { id: "nyc-links-line", type: "line", paint: { "line-color": ["get", "color"], "line-width": 3 } },
      ],
      popup: (p: Record<string, unknown>) => `<b>${p.name}</b>${kv([["Speed", `${fmt(Number(p.speed), 1)} mph`], ["Typical (same hour)", `${fmt(Number(p.typical), 1)} mph`], ["Anomaly", p.anomaly ? "yes" : "no"]])}`,
      onClick: (p: Record<string, unknown>) => setLink(String(p.id)),
    }];
  }, [nyc.data, borough]);

  const chiLayer = useMemo(() => [{
    id: "chi-segs",
    data: fc((chi.data?.segments ?? []).map((s) => line([[s.start_longitude, s.start_latitude], [s.end_longitude, s.end_latitude]],
      { id: s.segment_id, street: s.street, speed: s.speed, level: s.congestion_level ?? "n/a", color: LEVEL_COLORS[s.congestion_level ?? ""] ?? "#475569" }))),
    layers: [{ id: "chi-segs-line", type: "line", paint: { "line-color": ["get", "color"], "line-width": 3.5 } }],
    popup: (p: Record<string, unknown>) => `<b>${p.street}</b>${kv([["Speed", `${fmt(Number(p.speed), 0)} mph`], ["Level", p.level]])}`,
  }], [chi.data]);

  const fr = city === "nyc" ? nyc.data?.freshness : chi.data?.freshness;
  const boroughs = [...new Set((nyc.data?.links ?? []).map((l) => l.borough).filter(Boolean))].sort();

  return (
    <>
      <PageHeader title="Live Traffic Intelligence"
        description="NYC DOT sensor speeds refresh every few minutes and are the platform's live feed. Chicago Traffic Tracker is archived and is always shown as historical."
        right={<>
          <Segmented value={city} onChange={(v) => { setCity(v); setLink(null); }} options={[{ value: "nyc", label: "New York (DOT)" }, { value: "chicago", label: "Chicago (archive)" }]} />
          {city === "nyc" && <select value={borough} onChange={(e) => setBorough(e.target.value)}><option value="">All boroughs</option>{boroughs.map((b) => <option key={b}>{b}</option>)}</select>}
        </>} />

      {fr && (
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <FreshnessBadge mode={fr.mode} asOf={fr.data_as_of} />
          {fr.last_ingestion && <span className="text-xs text-slate-500">last ingestion {fmtTs(fr.last_ingestion)} · {fr.last_ingestion_status}</span>}
        </div>
      )}
      {fr?.note && <div className="mb-4"><Notice tone={fr.mode === "live" ? "info" : "warn"}>{fr.note}</Notice></div>}

      <div className="grid gap-5 xl:grid-cols-[1fr_340px]">
        <Card title={city === "nyc" ? "Link speed vs. typical for this hour" : "Last archived snapshot · congestion level"}
          subtitle={city === "nyc" ? "Colour = current speed ÷ median speed for the same hour on that link (click a link for its history)" : "Level = speed ÷ segment free-flow speed (85th pct. night-time, training period)"}>
          {city === "nyc"
            ? <Async state={nyc} h={520}>{() => <MapView layers={nycLayer} {...CENTERS.nyc} />}</Async>
            : <Async state={chi} h={520}>{() => <MapView layers={chiLayer} {...CENTERS.chicago} />}</Async>}
          <Legend city={city} />
        </Card>
        <div className="space-y-4">
          {city === "nyc" && nyc.data && (
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Links reporting" value={fmtInt(nyc.data.summary.links)} />
              <Stat label="Median speed" value={`${fmt(nyc.data.summary.median_speed_mph, 1)}`} sub="mph" accent="text-signal-cyan" />
              <Stat label="Slower than usual" value={fmtInt(nyc.data.summary.slow_links)} sub="< 60% of typical" accent="text-signal-amber" />
              <Stat label="Anomalous links" value={fmtInt(nyc.data.links.filter((l) => l.anomaly).length)} sub="last hour" accent="text-signal-red" />
            </div>
          )}
          <Card title="Traffic anomalies" subtitle={an.data?.summary ? `${fmtInt(an.data.summary.anomalies)} of ${fmtInt(an.data.summary.rows_scored)} observations flagged (${fmt(an.data.summary.rate * 100, 2)}%)` : "Robust z-score + Isolation Forest ∧ LOF"}>
            <Async state={an} h={300}>{(d) => (
              <div className="max-h-[380px] overflow-auto">
                <table className="data"><thead><tr><th>{city === "nyc" ? "Link" : "Segment"}</th><th>Time</th><th>Speed</th><th>z</th></tr></thead>
                  <tbody>{d.anomalies.slice(0, 60).map((a, i) => (
                    <tr key={i} className="cursor-pointer hover:bg-ink-800" onClick={() => city === "nyc" && setLink(String(a.id))}>
                      <td className="num">{String(a.id ?? a.segment_id)}</td>
                      <td className="text-xs text-slate-400">{fmtTs(String(a.data_as_of ?? a.time))}</td>
                      <td className="num">{fmt(Number(a.speed), 0)} <span className="text-xs text-slate-500">/ {fmt(Number(a.profile_median), 0)}</span></td>
                      <td className={`num ${Number(a.robust_z) < 0 ? "text-signal-red" : "text-signal-cyan"}`}>{fmt(Number(a.robust_z), 1)}</td>
                    </tr>))}</tbody></table>
              </div>)}</Async>
          </Card>
        </div>
      </div>

      {city === "chicago" && <ChicagoPatterns />}

      {link && (
        <Card className="mt-5" title={`Link ${link} · ${hist.data?.link_name ?? ""}`} subtitle="Observed speed over the last 72 hours of stored data"
          right={<button className="text-xs text-slate-400 hover:text-white" onClick={() => setLink(null)}>close</button>}>
          <Async state={hist} h={240}>{(d) => (
            <Plot height={240} data={[{ x: d.series.map((s) => s.data_as_of), y: d.series.map((s) => s.speed), type: "scatter", mode: "lines", line: { color: "#22D3EE", width: 1.5 }, name: "speed" }]}
              layout={{ yaxis: { title: { text: "mph" } } }} />)}</Async>
        </Card>
      )}
    </>
  );
}

function Legend({ city }: { city: "nyc" | "chicago" }) {
  const items = city === "nyc"
    ? [["≥ 80% of typical", "#22D3EE"], ["60–80%", "#F5A524"], ["40–60%", "#FB7A3C"], ["< 40%", "#F2545B"]]
    : Object.entries(LEVEL_COLORS);
  return (
    <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-400">
      {items.map(([l, c]) => <span key={l} className="flex items-center gap-1.5"><span className="h-1 w-5 rounded" style={{ background: c }} />{l}</span>)}
    </div>
  );
}

interface TrafficSummary { period: [string, string]; segments: number; hourly_profile: (CI & { daytype: string; hour: number })[]; dow_hour_ratio_heatmap: Heatmap;
  weekday_vs_weekend_speed: TwoGroup; spatial: { morans_i?: { I: number; p_value: number } | null };
  bottlenecks: { segment_id: number; street: string; direction: string; from_street: string; to_street: string; congested_share: number; recurrence_pct_days: number; mean_peak_speed: number; free_flow_speed: number }[] }

function ChicagoPatterns() {
  const t = useApi<TrafficSummary>("/api/traffic?limit=25");
  return (
    <Async state={t} h={400}>{(d) => (
      <div className="mt-5 space-y-5">
        <div className="grid gap-5 xl:grid-cols-2">
          <Card title="When congestion occurs" subtitle="Mean segment speed by hour, 95% CI (historical archive)"><HourlyCI rows={d.hourly_profile} unit="mph" /></Card>
          <Card title="Speed ÷ free-flow by weekday and hour" subtitle={d.spatial.morans_i ? `Spatial clustering of congestion: Moran's I = ${fmt(d.spatial.morans_i.I, 3)} (p ${d.spatial.morans_i.p_value < 0.001 ? "< 0.001" : fmt(d.spatial.morans_i.p_value, 3)})` : ""}>
            <DowHourHeatmap h={d.dow_hour_ratio_heatmap} colorscale={HEAT_SCALE} zTitle="ratio" /></Card>
        </div>
        <Card title="Recurring bottlenecks" subtitle="Weekday peak-hour intervals at High/Severe congestion; recurrence = share of peak days with ≥1 congested interval">
          <div className="max-h-[420px] overflow-auto"><table className="data"><thead><tr><th>Segment</th><th>Street</th><th>From → To</th><th className="text-right">Congested share</th><th className="text-right">Recurrence (days)</th><th className="text-right">Peak / free-flow mph</th></tr></thead>
            <tbody>{d.bottlenecks.map((b) => <tr key={b.segment_id}><td className="num">{b.segment_id}</td><td>{b.street} {b.direction}</td><td className="text-xs text-slate-400">{b.from_street} → {b.to_street}</td>
              <td className="num text-right">{fmt(b.congested_share * 100, 1)}%</td><td className="num text-right">{fmt(b.recurrence_pct_days, 0)}%</td><td className="num text-right">{fmt(b.mean_peak_speed, 1)} / {fmt(b.free_flow_speed, 1)}</td></tr>)}</tbody></table></div>
        </Card>
      </div>)}</Async>
  );
}
