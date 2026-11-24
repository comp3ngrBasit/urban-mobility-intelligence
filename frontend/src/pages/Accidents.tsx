import { useMemo, useState } from "react";
import { DowHourHeatmap } from "../components/charts";
import MapView, { CENTERS, fc, kv, point } from "../components/MapView";
import Plot from "../components/Plot";
import { Async, Card, Field, Notice, PageHeader, Segmented, Stat } from "../components/ui";
import { qs, useApi } from "../lib/api";
import { fmt, fmtDate, fmtInt, pct, pval, SEVERITY_COLORS } from "../lib/format";
import type { CI, Heatmap } from "../lib/types";

interface Summary { scope_note: string; period: [string, string]; total_crashes: number; crashes_per_day: CI; severity_counts: Record<string, number>; injuries_total: number; fatalities: number;
  monthly: { month: string; crashes: number; serious: number; fatal: number }[]; dow_hour_heatmap: Heatmap; top_causes: { prim_contributory_cause: string; crashes: number; serious_rate: number }[];
  lighting_vs_serious: { chi2?: number; p?: number; cramers_v?: number }; surface_conditions: { roadway_surface_cond: string; crashes: number; serious_rate: number }[] }
interface Hot { n_crashes: number; morans_i: { I: number; p_value: number; z: number }; significant_hotspots: number; cells?: { h3: string; lat: number; lon: number; crashes: number; serious: number; gi_z: number; gi_class: string }[];
  clusters?: { cluster: number; crashes: number; lat: number; lon: number; serious_rate: number; top_cause?: string }[]; kde?: { lats: number[]; lons: number[]; density: number[][] } }
interface Pts { total_matching: number; returned: number; points: { latitude: number; longitude: number; severity: string; crash_date: string; weather_condition: string; prim_contributory_cause: string }[] }

type Method = "gi" | "dbscan" | "hdbscan" | "kmeans" | "kde";
const SEVS = ["No injury", "Minor", "Serious", "Fatal"];

export default function Accidents() {
  const [method, setMethod] = useState<Method>("gi");
  const [sev, setSev] = useState<string[]>(["Serious", "Fatal"]);
  const [weather, setWeather] = useState("");
  const [hours, setHours] = useState<[number, number]>([0, 23]);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const sum = useApi<Summary>("/api/accidents/summary");
  const hot = useApi<Hot>(`/api/accidents/hotspots?method=${method}`);
  const pts = useApi<Pts>(`/api/accidents/points${qs({ severity: sev, weather, hour_from: hours[0], hour_to: hours[1], start, end, limit: 15000 })}`);

  const layers = useMemo(() => {
    const out = [];
    const h = hot.data;
    if (h?.cells) out.push({ id: "gi", data: fc(h.cells.filter((c) => c.gi_class !== "Not significant").map((c) => point(c.lon, c.lat, { ...c, color: c.gi_class === "Hotspot" ? "#F2545B" : "#5B9BFF", r: 4 + Math.min(14, Math.abs(c.gi_z) * 2) }))),
      layers: [{ id: "gi-c", type: "circle", paint: { "circle-color": ["get", "color"], "circle-radius": ["get", "r"], "circle-opacity": 0.35, "circle-stroke-color": ["get", "color"], "circle-stroke-width": 1 } }],
      popup: (p: Record<string, unknown>) => `<b>${p.gi_class}</b>${kv([["Gi* z", fmt(Number(p.gi_z), 2)], ["Crashes", fmtInt(Number(p.crashes))], ["Serious/fatal", fmtInt(Number(p.serious))]])}` });
    if (h?.clusters) out.push({ id: "cl", data: fc(h.clusters.map((c) => point(c.lon, c.lat, { ...c, r: 4 + Math.sqrt(c.crashes) / 2 }))),
      layers: [{ id: "cl-c", type: "circle", paint: { "circle-color": "#8B5CF6", "circle-radius": ["get", "r"], "circle-opacity": 0.35, "circle-stroke-color": "#A78BFA", "circle-stroke-width": 1 } }],
      popup: (p: Record<string, unknown>) => `<b>Cluster ${p.cluster}</b>${kv([["Crashes", fmtInt(Number(p.crashes))], ["Serious rate", pct(Number(p.serious_rate))], ["Top cause", p.top_cause]])}` });
    if (h?.kde) {
      const f: GeoJSON.Feature[] = [];
      h.kde.lats.forEach((la, i) => h.kde!.lons.forEach((lo, j) => { const v = h.kde!.density[i][j]; if (v > 0.05) f.push(point(lo, la, { v })); }));
      out.push({ id: "kde", data: fc(f), layers: [{ id: "kde-h", type: "heatmap", paint: { "heatmap-weight": ["get", "v"], "heatmap-radius": 28, "heatmap-intensity": 1.1,
        "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(0,0,0,0)", 0.3, "#1E3A8A", 0.6, "#8B5CF6", 0.85, "#F5A524", 1, "#F2545B"] } }] });
    }
    if (pts.data) out.push({ id: "pts", data: fc(pts.data.points.map((p) => point(p.longitude, p.latitude, { ...p, color: SEVERITY_COLORS[p.severity] }))),
      layers: [{ id: "pts-c", type: "circle", paint: { "circle-color": ["get", "color"], "circle-radius": 2.2, "circle-opacity": 0.75 } }],
      popup: (p: Record<string, unknown>) => `<b>${p.severity}</b>${kv([["Date", fmtDate(String(p.crash_date))], ["Weather", p.weather_condition], ["Cause", p.prim_contributory_cause]])}` });
    return out;
  }, [hot.data, pts.data]);

  return (
    <>
      <PageHeader title="Accident Intelligence" description="Chicago police-reported crashes: statistically significant hotspots, cluster structure, severity drivers and temporal risk." />
      <Async state={sum} h={600}>{(s) => (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Stat label="Crashes" value={fmtInt(s.total_crashes)} sub={`${fmtDate(s.period[0])} – ${fmtDate(s.period[1])}`} />
            <Stat label="Per day" value={fmt(s.crashes_per_day.mean, 1)} sub={`95% CI ${fmt(s.crashes_per_day.lo, 1)}–${fmt(s.crashes_per_day.hi, 1)}`} />
            <Stat label="Injuries" value={fmtInt(s.injuries_total)} accent="text-signal-amber" />
            <Stat label="Fatalities" value={fmtInt(s.fatalities)} accent="text-signal-red" />
            <Stat label="Global Moran's I" value={hot.data ? fmt(hot.data.morans_i.I, 3) : "—"} sub={hot.data ? `p ${pval(hot.data.morans_i.p_value)} · ${hot.data.significant_hotspots} Gi* hotspots (FDR 5%)` : ""} />
          </div>
          <Card title="Hotspot map" subtitle="Hotspot layer + filtered crash points (sampled to 15k for display)"
            right={<Segmented value={method} onChange={setMethod} options={[{ value: "gi", label: "Gi* (significant)" }, { value: "dbscan", label: "DBSCAN" }, { value: "hdbscan", label: "HDBSCAN" }, { value: "kmeans", label: "K-Means" }, { value: "kde", label: "KDE" }]} />}>
            <div className="mb-3 flex flex-wrap items-end gap-3">
              <Field label="Severity">
                <div className="flex gap-1">{SEVS.map((v) => (
                  <button key={v} onClick={() => setSev((x) => (x.includes(v) ? x.filter((y) => y !== v) : [...x, v]))}
                    className={`rounded-md border px-2 py-1 text-xs ${sev.includes(v) ? "border-transparent text-ink-900" : "border-ink-600 text-slate-400"}`}
                    style={sev.includes(v) ? { background: SEVERITY_COLORS[v] } : undefined}>{v}</button>))}</div>
              </Field>
              <Field label="Reported weather"><select value={weather} onChange={(e) => setWeather(e.target.value)}><option value="">Any</option>{["CLEAR", "RAIN", "SNOW", "CLOUDY/OVERCAST", "FOG/SMOKE/HAZE", "SLEET/HAIL", "FREEZING RAIN/DRIZZLE"].map((w) => <option key={w}>{w}</option>)}</select></Field>
              <Field label="Hour from"><input type="number" min={0} max={23} value={hours[0]} onChange={(e) => setHours([Number(e.target.value), hours[1]])} className="w-20" /></Field>
              <Field label="to"><input type="number" min={0} max={23} value={hours[1]} onChange={(e) => setHours([hours[0], Number(e.target.value)])} className="w-20" /></Field>
              <Field label="From date"><input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
              <Field label="To date"><input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
              {pts.data && <span className="pb-2 text-xs text-slate-400">{fmtInt(pts.data.total_matching)} matching · {fmtInt(pts.data.returned)} shown</span>}
            </div>
            <Async state={hot} h={520}>{() => <MapView layers={layers} {...CENTERS.chicago} height={520} />}</Async>
          </Card>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Monthly trend"><Plot height={260} data={[
              { x: s.monthly.map((m) => m.month), y: s.monthly.map((m) => m.crashes), type: "scatter", mode: "lines", name: "All crashes", line: { color: "#3B82F6" } },
              { x: s.monthly.map((m) => m.month), y: s.monthly.map((m) => m.serious), type: "scatter", mode: "lines", name: "Serious + fatal", yaxis: "y2", line: { color: "#F2545B" } }]}
              layout={{ yaxis2: { overlaying: "y", side: "right", showgrid: false } }} /></Card>
            <Card title="When crashes happen" subtitle={s.dow_hour_heatmap.unit}><DowHourHeatmap h={s.dow_hour_heatmap} colorscale={[[0, "#0F1729"], [0.5, "#8B5CF6"], [1, "#F2545B"]]} /></Card>
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            <Card title="Primary contributory causes" subtitle="Crash count and share resulting in serious or fatal injury">
              <Plot height={380} data={[{ y: s.top_causes.map((c) => c.prim_contributory_cause.slice(0, 42)), x: s.top_causes.map((c) => c.crashes), type: "bar", orientation: "h", marker: { color: s.top_causes.map((c) => c.serious_rate), colorscale: [[0, "#3B82F6"], [1, "#F2545B"]], colorbar: { thickness: 8, tickformat: ".1%" } } }]}
                layout={{ margin: { l: 270, r: 10, t: 10, b: 30 }, yaxis: { autorange: "reversed", tickfont: { size: 10 } } }} />
            </Card>
            <Card title="Road surface and lighting" subtitle={s.lighting_vs_serious.chi2 ? `Lighting × serious injury: χ² = ${fmt(s.lighting_vs_serious.chi2, 1)}, p ${pval(s.lighting_vs_serious.p)}, Cramér's V = ${fmt(s.lighting_vs_serious.cramers_v, 3)}` : ""}>
              <table className="data"><thead><tr><th>Surface</th><th className="text-right">Crashes</th><th className="text-right">Serious rate</th></tr></thead>
                <tbody>{s.surface_conditions.slice(0, 8).map((r) => <tr key={r.roadway_surface_cond}><td>{r.roadway_surface_cond}</td><td className="num text-right">{fmtInt(r.crashes)}</td><td className="num text-right">{pct(r.serious_rate, 2)}</td></tr>)}</tbody></table>
            </Card>
          </div>
          <Notice>{s.scope_note} Gi* hotspots use H3 resolution-8 cells, 999 permutations and Benjamini–Hochberg FDR control; DBSCAN/HDBSCAN/K-Means clusters are descriptive, not significance tests.</Notice>
        </div>)}</Async>
    </>
  );
}
