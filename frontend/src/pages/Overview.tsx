import { Link } from "react-router-dom";
import Plot from "../components/Plot";
import { Async, Card, FreshnessBadge, Notice, PageHeader, Stat } from "../components/ui";
import { useApi } from "../lib/api";
import { fmt, fmtDate, fmtInt, LEVEL_COLORS, pct, SEVERITY_COLORS } from "../lib/format";
import type { Overview as O } from "../lib/types";

const LEVELS = ["Low", "Moderate", "High", "Severe"];

export default function Overview() {
  const s = useApi<O>("/api/overview", 120_000);
  return (
    <>
      <PageHeader title="Executive Overview"
        description="Headline indicators across the three independent data domains: NYC taxi demand, road speeds (NYC live, Chicago archived) and Chicago crash safety." />
      <Async state={s} h={500}>
        {(d) => (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
              <Stat label="Taxi trips analysed" value={fmtInt(d.mobility?.total_trips)}
                sub={d.mobility ? `${fmtDate(d.mobility.period[0])} – ${fmtDate(d.mobility.period[1])}` : "not ingested"} />
              <Stat label="NYC median link speed" value={d.nyc_live ? `${fmt(d.nyc_live.median_speed_mph, 1)} mph` : "—"}
                sub={d.nyc_live ? <FreshnessBadge mode={d.nyc_live.freshness.mode} asOf={d.nyc_live.freshness.data_as_of} /> : "no live feed"}
                accent="text-signal-cyan" />
              <Stat label="NYC links slower than usual" value={d.nyc_live ? fmtInt(d.nyc_live.slow_links) : "—"}
                sub={d.nyc_live ? `of ${fmtInt(d.nyc_live.links)} links (< 60% of typical)` : undefined} />
              <Stat label="Chicago mean arterial speed" value={d.chicago_traffic?.mean_speed.mean != null ? `${fmt(d.chicago_traffic.mean_speed.mean, 1)} mph` : "—"}
                sub={d.chicago_traffic ? <FreshnessBadge mode="historical" asOf={d.chicago_traffic.period[1]} /> : "not ingested"} />
              <Stat label="Chicago crashes / day" value={fmt(d.accidents?.crashes_per_day.mean, 1)}
                sub={d.accidents ? `95% CI ${fmt(d.accidents.crashes_per_day.lo, 1)}–${fmt(d.accidents.crashes_per_day.hi, 1)}` : "not ingested"}
                accent="text-signal-amber" />
            </div>

            <div className="grid gap-5 xl:grid-cols-3">
              <Card title="Taxi demand trend" subtitle="Daily NYC TLC pickups" className="xl:col-span-2">
                {d.mobility ? (
                  <Plot height={260} data={[{ x: d.mobility.daily.ts, y: d.mobility.daily.pickups, type: "scatter", mode: "lines",
                    line: { color: "#3B82F6", width: 2 }, fill: "tozeroy", fillcolor: "rgba(59,130,246,0.08)", name: "pickups" }]} />
                ) : <Notice>Taxi data not yet ingested.</Notice>}
              </Card>
              <Card title="Congestion distribution" subtitle="Share of segment-intervals by level · Chicago (historical) vs model forecast">
                {d.chicago_traffic ? (
                  <Plot height={260} data={[
                    { x: LEVELS, y: LEVELS.map((l) => d.chicago_traffic!.congestion_distribution[l] ?? 0), type: "bar", name: "Observed",
                      marker: { color: LEVELS.map((l) => LEVEL_COLORS[l]) } },
                    ...(d.predicted_congestion ? [{ x: LEVELS, y: LEVELS.map((l) => d.predicted_congestion!.distribution[l] ?? 0),
                      type: "bar" as const, name: "Forecast (+60 min)", marker: { color: "rgba(139,92,246,0.55)" } }] : []),
                  ]} layout={{ barmode: "group", yaxis: { tickformat: ".0%" } }} />
                ) : <Notice>Chicago traffic not yet ingested.</Notice>}
              </Card>
            </div>

            <div className="grid gap-5 xl:grid-cols-3">
              <Card title="Crash trend" subtitle="Monthly Chicago crashes and serious/fatal injuries" className="xl:col-span-2">
                {d.accidents ? (
                  <Plot height={250} data={[
                    { x: d.accidents.monthly.map((m) => m.month), y: d.accidents.monthly.map((m) => m.crashes), type: "bar", name: "Crashes", marker: { color: "rgba(59,130,246,0.6)" } },
                    { x: d.accidents.monthly.map((m) => m.month), y: d.accidents.monthly.map((m) => m.serious), type: "scatter", mode: "lines+markers", name: "Serious + fatal", yaxis: "y2", line: { color: "#F5A524" } },
                  ]} layout={{ yaxis2: { overlaying: "y", side: "right", showgrid: false, tickfont: { color: "#F5A524" } } }} />
                ) : <Notice>Crash data not yet ingested.</Notice>}
              </Card>
              <Card title="Crash severity mix" subtitle={d.accidents ? `${fmtInt(d.accidents.total_crashes)} crashes · ${fmtInt(d.accidents.fatalities)} fatalities` : ""}>
                {d.accidents ? (
                  <div className="space-y-2.5 pt-1">
                    {Object.entries(d.accidents.severity_counts).map(([k, v]) => (
                      <div key={k}>
                        <div className="flex justify-between text-xs"><span className="text-slate-300">{k}</span>
                          <span className="num text-slate-400">{fmtInt(v)} · {pct(v / d.accidents!.total_crashes)}</span></div>
                        <div className="mt-1 h-1.5 rounded-full bg-ink-700">
                          <div className="h-1.5 rounded-full" style={{ width: `${Math.max(0.5, (v / d.accidents!.total_crashes) * 100)}%`, background: SEVERITY_COLORS[k] }} />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : <Notice>Crash data not yet ingested.</Notice>}
              </Card>
            </div>

            {d.mobility && d.mobility.borough_peak_hours.some((b) => b.borough && !["Unknown", "N/A"].includes(b.borough)) && (
              <Card title="Peak taxi demand hour by borough" subtitle="Hour of day with the most pickups over the analysed period">
                <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
                  {d.mobility.borough_peak_hours.filter((b) => b.borough && b.borough !== "Unknown" && b.borough !== "N/A").map((b) => (
                    <div key={b.borough} className="rounded-lg bg-ink-800 px-3 py-2.5">
                      <div className="text-xs text-slate-400">{b.borough}</div>
                      <div className="num text-lg font-semibold text-white">{String(b.peak_hour).padStart(2, "0")}:00</div>
                    </div>
                  ))}
                </div>
              </Card>
            )}
            <Notice>
              {d.mobility?.scope_note} {d.chicago_traffic?.scope_note} See <Link className="underline" to="/pipeline">Data & Pipeline</Link> for source freshness.
            </Notice>
          </div>
        )}
      </Async>
    </>
  );
}
