import { useMemo, useState } from "react";
import MapView, { CENTERS, kv } from "../components/MapView";
import { Card, Notice, PageHeader, Segmented } from "../components/ui";
import { qs, useApi } from "../lib/api";
import { fmt, fmtInt, pct, pval } from "../lib/format";

type City = "chicago" | "nyc";
const LAYERS: Record<City, { key: string; label: string }[]> = {
  chicago: [
    { key: "roads", label: "OSM road network" }, { key: "segments", label: "Tracker segments (congestion share)" },
    { key: "h3-crashes", label: "H3 crash Gi*" }, { key: "h3-congestion", label: "H3 congestion Gi*" }, { key: "h3-priority", label: "Infrastructure priority" },
  ],
  nyc: [{ key: "roads", label: "OSM road network" }, { key: "zones", label: "Taxi zones (pickups/day)" }],
};
const GI = ["match", ["get", "gi_class"], "Hotspot", "#F2545B", "Coldspot", "#3B82F6", "rgba(100,116,139,0.25)"];

export default function GeoExplorer() {
  const [city, setCity] = useState<City>("chicago");
  const [on, setOn] = useState<Record<string, boolean>>({ roads: true, segments: true, "h3-crashes": false, "h3-congestion": false, "h3-priority": false, zones: true });
  const [minRank, setMinRank] = useState(3);
  const [sel, setSel] = useState<Record<string, unknown> | null>(null);
  const L = (k: string) => on[k] && LAYERS[city].some((l) => l.key === k);

  const roads = useApi<GeoJSON.FeatureCollection>(L("roads") ? `/api/geo/road-network${qs({ city, min_rank: minRank })}` : null);
  const segs = useApi<GeoJSON.FeatureCollection>(L("segments") ? "/api/geo/chicago-segments" : null);
  const hc = useApi<GeoJSON.FeatureCollection>(L("h3-crashes") ? "/api/geo/h3?layer=crashes" : null);
  const hg = useApi<GeoJSON.FeatureCollection>(L("h3-congestion") ? "/api/geo/h3?layer=congestion" : null);
  const hp = useApi<GeoJSON.FeatureCollection>(L("h3-priority") ? "/api/geo/h3?layer=priority" : null);
  const zones = useApi<GeoJSON.FeatureCollection>(L("zones") ? "/api/geo/taxi-zones" : null);
  const traffic = useApi<{ spatial: { morans_i?: { I: number; p_value: number } } }>(city === "chicago" ? "/api/traffic?limit=1" : null);
  const hotspots = useApi<{ morans_i: { I: number; p_value: number; z: number }; significant_hotspots: number }>(city === "chicago" ? "/api/accidents/hotspots?method=gi" : null);
  const prio = useApi<{ method: string; cells: { h3: string; priority_index: number; serious: number; congestion_share: number }[] }>(city === "chicago" ? "/api/geo/infrastructure-priority?limit=10" : null);

  const layers = useMemo(() => {
    const out = [];
    const pick = (p: Record<string, unknown>) => setSel(p);
    if (L("zones") && zones.data) out.push({ id: "zones", data: zones.data, onClick: pick,
      layers: [{ id: "z-f", type: "fill", paint: { "fill-color": ["interpolate", ["linear"], ["coalesce", ["get", "pickups_per_day"], 0], 0, "#0F1729", 200, "#1E3A8A", 2000, "#3B82F6", 8000, "#22D3EE"], "fill-opacity": 0.6 } },
        { id: "z-l", type: "line", paint: { "line-color": "#2A3858", "line-width": 0.5 } }] });
    if (L("h3-crashes") && hc.data) out.push({ id: "hc", data: hc.data, onClick: pick, layers: [{ id: "hc-f", type: "fill", paint: { "fill-color": GI, "fill-opacity": 0.45, "fill-outline-color": "#0B1220" } }] });
    if (L("h3-congestion") && hg.data) out.push({ id: "hg", data: hg.data, onClick: pick, layers: [{ id: "hg-f", type: "fill", paint: { "fill-color": GI, "fill-opacity": 0.45 } }] });
    if (L("h3-priority") && hp.data) out.push({ id: "hp", data: hp.data, onClick: pick, layers: [{ id: "hp-f", type: "fill", paint: { "fill-color": ["interpolate", ["linear"], ["get", "priority_index"], -1, "#0F1729", 0, "#8B5CF6", 1.5, "#F5A524", 3, "#F2545B"], "fill-opacity": 0.55 } }] });
    if (L("roads") && roads.data) out.push({ id: "roads", data: roads.data, onClick: pick,
      layers: [{ id: "r-l", type: "line", paint: { "line-color": ["match", ["get", "road_class"], "motorway", "#8B5CF6", "trunk", "#A78BFA", "primary", "#3B82F6", "secondary", "#5B9BFF", "#334155"], "line-width": ["interpolate", ["linear"], ["get", "road_rank"], 1, 0.4, 6, 2.2] } }] });
    if (L("segments") && segs.data) out.push({ id: "segs", data: segs.data, onClick: pick,
      layers: [{ id: "s-l", type: "line", paint: { "line-color": ["interpolate", ["linear"], ["coalesce", ["get", "p_congested"], 0], 0, "#22D3EE", 0.15, "#F5A524", 0.35, "#F2545B"], "line-width": 3.5 } }] });
    return out;
  }, [zones.data, hc.data, hg.data, hp.data, roads.data, segs.data, on, city]); // eslint-disable-line react-hooks/exhaustive-deps

  const missing = [roads, segs, hc, hg, hp, zones].filter((s) => s.error).length;

  return (
    <>
      <PageHeader title="Geospatial Explorer" description="Overlay the OSM road network, tracker segments, H3 spatial statistics and taxi zones. Click any feature to inspect its attributes."
        right={<Segmented value={city} onChange={(c) => { setCity(c); setSel(null); }} options={[{ value: "chicago", label: "Chicago" }, { value: "nyc", label: "New York" }]} />} />
      <div className="grid gap-5 xl:grid-cols-[260px_1fr_300px]">
        <Card title="Layers">
          <div className="space-y-2">
            {LAYERS[city].map((l) => (
              <label key={l.key} className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" className="accent-electric-500" checked={!!on[l.key]} onChange={(e) => setOn({ ...on, [l.key]: e.target.checked })} />{l.label}
              </label>))}
          </div>
          <div className="mt-4"><span className="label">Min road class</span>
            <select className="mt-1 w-full" value={minRank} onChange={(e) => setMinRank(Number(e.target.value))}>
              <option value={5}>Motorway / trunk</option><option value={4}>+ primary</option><option value={3}>+ secondary</option><option value={2}>+ tertiary</option><option value={1}>All drivable</option>
            </select></div>
          {missing > 0 && <div className="mt-4"><Notice tone="warn">{missing} layer(s) not available yet — run ingestion/analysis for them.</Notice></div>}
        </Card>
        <Card><MapView layers={layers} {...CENTERS[city]} height={620} /></Card>
        <div className="space-y-4">
          <Card title="Selection">{sel ? <div className="space-y-1 text-xs" dangerouslySetInnerHTML={{ __html: kv(Object.entries(sel).filter(([, v]) => v !== null && v !== "").slice(0, 14).map(([k, v]) => [k, typeof v === "number" ? fmt(v, 3) : String(v)])) }} /> : <p className="text-xs text-slate-500">Click a feature on the map.</p>}</Card>
          {city === "chicago" && hotspots.data && (
            <Card title="Spatial autocorrelation">
              <div className="text-xs text-slate-300">Crash counts per H3 cell: Moran's I = <b className="num">{fmt(hotspots.data.morans_i.I, 3)}</b> (z {fmt(hotspots.data.morans_i.z, 1)}, p {pval(hotspots.data.morans_i.p_value)}); {fmtInt(hotspots.data.significant_hotspots)} significant Gi* hotspot cells.</div>
              {traffic.data?.spatial?.morans_i && <div className="mt-2 text-xs text-slate-300">Congestion share: Moran's I = <b>{fmt(traffic.data.spatial.morans_i.I, 3)}</b></div>}
            </Card>)}
          {city === "chicago" && prio.data && (
            <Card title="Infrastructure priority" subtitle={prio.data.method}>
              <table className="data"><thead><tr><th>Cell</th><th className="text-right">Index</th><th className="text-right">Serious</th><th className="text-right">Cong.</th></tr></thead>
                <tbody>{prio.data.cells.map((c) => <tr key={c.h3}><td className="num text-[10px]">{c.h3.slice(0, 10)}…</td><td className="num text-right">{fmt(c.priority_index, 2)}</td><td className="num text-right">{fmtInt(c.serious)}</td><td className="num text-right">{pct(c.congestion_share, 0)}</td></tr>)}</tbody></table>
            </Card>)}
        </div>
      </div>
    </>
  );
}
