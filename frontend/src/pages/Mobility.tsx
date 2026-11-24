import { useMemo, useState } from "react";
import { DowHourHeatmap, HourlyCI, NAVY_SCALE } from "../components/charts";
import MapView, { CENTERS, fc, kv, line } from "../components/MapView";
import Plot from "../components/Plot";
import { Async, Card, Field, Notice, PageHeader, Segmented, Stat } from "../components/ui";
import { qs, useApi } from "../lib/api";
import { fmt, fmtDate, fmtInt, pval } from "../lib/format";
import type { CI, Heatmap, TwoGroup } from "../lib/types";

interface Mob {
  scope_note: string; period: [string, string]; total_trips: number; hourly_profile: (CI & { daytype: string; hour: number })[];
  dow_hour_heatmap: Heatmap; zones: { zone: number; zone_name: string; borough: string; pickups_per_day: number; net_flow_per_day: number; cluster?: number }[];
  demand_clusters: { cluster: number; n_zones: number; peak_hour: number; profile: number[]; example_zones: string[] }[];
  weekday_vs_weekend: TwoGroup; holiday_vs_weekday: TwoGroup;
}
interface Flow { pu: number; do: number; trips: number; median_duration_min: number; pu_zone_name?: string; do_zone_name?: string; pu_lon?: number; pu_lat?: number; do_lon?: number; do_lat?: number }

export default function Mobility() {
  const [borough, setBorough] = useState("");
  const [hour, setHour] = useState<number | "">("");
  const [metric, setMetric] = useState<"pickups_per_day" | "net_flow_per_day">("pickups_per_day");
  const mob = useApi<Mob>(`/api/mobility${qs({ borough })}`);
  const od = useApi<{ flows: Flow[] }>(`/api/mobility/od${qs({ borough, hour: hour === "" ? null : hour, limit: 120 })}`);
  const zones = useApi<GeoJSON.FeatureCollection>(`/api/geo/taxi-zones`);
  const dec = useApi<{ ts: string[]; observed: number[]; trend: number[]; seasonal: number[]; strength_seasonal: number; strength_trend: number }>("/api/mobility/decomposition");

  const layers = useMemo(() => {
    const out = [];
    if (zones.data) {
      const vals = zones.data.features.map((f) => Number(f.properties?.[metric] ?? 0));
      const max = Math.max(...vals.map(Math.abs), 1);
      const feats = zones.data.features.filter((f) => !borough || f.properties?.borough === borough)
        .map((f) => ({ ...f, properties: { ...f.properties, v: Number(f.properties?.[metric] ?? 0) / max } }));
      out.push({
        id: "zones", data: fc(feats),
        layers: [
          { id: "zones-fill", type: "fill", paint: { "fill-color": metric === "net_flow_per_day"
            ? ["interpolate", ["linear"], ["get", "v"], -1, "#8B5CF6", 0, "#0F1729", 1, "#22D3EE"]
            : ["interpolate", ["linear"], ["sqrt", ["max", ["get", "v"], 0]], 0, "#0F1729", 0.4, "#1E3A8A", 0.75, "#3B82F6", 1, "#22D3EE"], "fill-opacity": 0.7 } },
          { id: "zones-line", type: "line", paint: { "line-color": "#2A3858", "line-width": 0.5 } },
        ],
        popup: (p: Record<string, unknown>) => `<b>${p.zone}</b>${kv([["Borough", p.borough], ["Pickups / day", fmt(Number(p.pickups_per_day), 0)], ["Net inflow / day", fmt(Number(p.net_flow_per_day), 0)], ["Demand cluster", p.cluster]])}`,
      });
    }
    const flows = (od.data?.flows ?? []).filter((f) => f.pu_lon != null);
    if (flows.length) {
      const mx = Math.max(...flows.map((f) => f.trips));
      out.push({ id: "od", data: fc(flows.map((f) => line([[f.pu_lon!, f.pu_lat!], [f.do_lon!, f.do_lat!]], { w: 0.5 + 5 * (f.trips / mx), t: f.trips, a: f.pu_zone_name, b: f.do_zone_name, m: f.median_duration_min }))),
        layers: [{ id: "od-line", type: "line", paint: { "line-color": "#F5A524", "line-width": ["get", "w"], "line-opacity": 0.55 } }],
        popup: (p: Record<string, unknown>) => `<b>${p.a} → ${p.b}</b>${kv([["Trips", fmtInt(Number(p.t))], ["Median duration", `${fmt(Number(p.m), 1)} min`]])}` });
    }
    return out;
  }, [zones.data, od.data, metric, borough]);

  return (
    <>
      <PageHeader title="Mobility Analytics" description="NYC TLC taxi and for-hire trips: when and where demand happens, how zones connect, and how weekday, weekend and holiday patterns differ."
        right={<>
          <select value={borough} onChange={(e) => setBorough(e.target.value)}><option value="">All boroughs</option>{["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"].map((b) => <option key={b}>{b}</option>)}</select>
        </>} />
      <Async state={mob} h={600}>{(m) => (
        <div className="space-y-5">
          <Notice>{m.scope_note}</Notice>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat label="Trips analysed" value={fmtInt(m.total_trips)} sub={`${fmtDate(m.period[0])} – ${fmtDate(m.period[1])}`} />
            <Stat label="Weekday vs weekend (hourly)" value={`d = ${fmt(m.weekday_vs_weekend.cohens_d ?? null, 2)}`} sub={`Welch p ${pval(m.weekday_vs_weekend.welch_p)}`} />
            <Stat label="Holiday vs weekday" value={`d = ${fmt(m.holiday_vs_weekday.cohens_d ?? null, 2)}`} sub={m.holiday_vs_weekday.note ?? `Welch p ${pval(m.holiday_vs_weekday.welch_p)}`} />
            <Stat label="Zones in view" value={fmtInt(m.zones.length)} />
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Pickups by hour of day" subtitle="Citywide mean pickups per hour with 95% confidence interval"><HourlyCI rows={m.hourly_profile} unit="pickups / hour" /></Card>
            <Card title="Weekday × hour demand" subtitle="Mean citywide pickups"><DowHourHeatmap h={m.dow_hour_heatmap} colorscale={NAVY_SCALE} /></Card>
          </div>
          <Card title="Zone demand and origin–destination flows"
            subtitle="Choropleth = selected metric · amber arcs = top OD pairs (straight lines between zone centroids)"
            right={<div className="flex items-end gap-3">
              <Segmented value={metric} onChange={setMetric} options={[{ value: "pickups_per_day", label: "Pickups / day" }, { value: "net_flow_per_day", label: "Net inflow" }]} />
              <Field label="OD hour"><select value={hour} onChange={(e) => setHour(e.target.value === "" ? "" : Number(e.target.value))}><option value="">All hours</option>{Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{String(i).padStart(2, "0")}:00</option>)}</select></Field>
            </div>}>
            {zones.error && !zones.data ? <Notice tone="warn">Taxi zone boundaries not ingested; OD table shown below.</Notice> : <MapView layers={layers} {...CENTERS.nyc} height={500} />}
          </Card>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Top origin–destination pairs">
              <Async state={od} h={300}>{(d) => (
                <div className="max-h-[360px] overflow-auto"><table className="data"><thead><tr><th>From</th><th>To</th><th className="text-right">Trips</th><th className="text-right">Median min</th></tr></thead>
                  <tbody>{d.flows.slice(0, 40).map((f) => <tr key={`${f.pu}-${f.do}`}><td>{f.pu_zone_name ?? f.pu}</td><td>{f.do_zone_name ?? f.do}</td><td className="num text-right">{fmtInt(f.trips)}</td><td className="num text-right">{fmt(f.median_duration_min, 1)}</td></tr>)}</tbody></table></div>)}</Async>
            </Card>
            <Card title="Demand clusters" subtitle="K-Means on each zone's normalised weekday 24-hour pickup profile">
              {m.demand_clusters.length ? (
                <Plot height={330} data={m.demand_clusters.map((c) => ({ x: Array.from({ length: 24 }, (_, i) => i), y: c.profile, type: "scatter" as const, mode: "lines" as const, name: `C${c.cluster} · ${c.n_zones} zones · peak ${c.peak_hour}:00`, hovertext: c.example_zones.join(", ") }))}
                  layout={{ xaxis: { title: { text: "hour" }, dtick: 3 }, yaxis: { title: { text: "share of daily pickups" }, tickformat: ".0%" }, legend: { orientation: "v", x: 1.02, y: 1 } }} />
              ) : <Notice>Not enough zones for clustering.</Notice>}
            </Card>
          </div>
          <Card title="Seasonal decomposition (STL, weekly period)" subtitle={dec.data ? `Seasonal strength ${fmt(dec.data.strength_seasonal, 2)} · trend strength ${fmt(dec.data.strength_trend, 2)} · last 6 weeks of citywide hourly pickups` : ""}>
            <Async state={dec} h={260}>{(d) => d.ts ? (
              <Plot height={280} data={[
                { x: d.ts, y: d.observed, type: "scatter", mode: "lines", name: "observed", line: { color: "rgba(148,163,184,0.5)", width: 1 } },
                { x: d.ts, y: d.trend, type: "scatter", mode: "lines", name: "trend", line: { color: "#22D3EE", width: 2 } },
                { x: d.ts, y: d.seasonal, type: "scatter", mode: "lines", name: "weekly seasonal", line: { color: "#8B5CF6", width: 1 }, yaxis: "y2" },
              ]} layout={{ yaxis2: { overlaying: "y", side: "right", showgrid: false } }} />) : <Notice>Series too short for weekly decomposition.</Notice>}</Async>
          </Card>
        </div>)}</Async>
    </>
  );
}
