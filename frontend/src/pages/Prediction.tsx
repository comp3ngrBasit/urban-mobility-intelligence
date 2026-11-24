import { useEffect, useMemo, useState } from "react";
import MapView, { CENTERS, fc, kv, line } from "../components/MapView";
import Plot from "../components/Plot";
import { Async, Card, Field, Notice, PageHeader, Segmented, Stat } from "../components/ui";
import { api, ApiError, qs, useApi } from "../lib/api";
import { fmt, fmtTs, LEVEL_COLORS, pct } from "../lib/format";
import type { Segment } from "../lib/types";

interface Options { speed_horizons_min: number[]; demand_horizons_h: number[]; segments: Segment[]; zones: { zone: number; zone_name: string; borough: string }[]; travel_time_available: boolean }
interface Pt { ts: string; actual: number; predicted: number; lower?: number; upper?: number }
interface SegDetail { speed: Pt[]; congestion?: { ts: string; actual: string; predicted: string }[]; forecast?: (Pt & { segment_id: number })[]; segment_metrics: Record<string, number>; overall_test_metrics: Record<string, number>; intervals: { empirical_test_coverage: number; nominal: number } | null }
interface SegFc { target_time: string; issued_at: string; note: string; segments: (Segment & { predicted_speed: number; speed_lower: number; speed_upper: number; predicted_level?: string; confidence?: number })[] }
interface ZoneDetail { series: Pt[]; forecast: Pt[]; zone_metrics: Record<string, number>; intervals: { empirical_test_coverage: number } | null }

const hLabel = (m: number) => (m >= 60 ? `${m / 60} h` : `${m} min`);

export default function Prediction() {
  const [tab, setTab] = useState<"segment" | "zone" | "trip">("segment");
  const opts = useApi<Options>("/api/predict/options");
  return (
    <>
      <PageHeader title="Traffic Prediction"
        description="Select a road segment, taxi zone or trip. Charts compare model predictions with actual values on the held-out test period; forecasts beyond the data carry 80% quantile intervals."
        right={<Segmented value={tab} onChange={setTab} options={[{ value: "segment", label: "Road segment" }, { value: "zone", label: "Zone demand" }, { value: "trip", label: "Travel time" }]} />} />
      <Async state={opts} h={500}>{(o) => (
        tab === "segment" ? <SegmentView o={o} /> : tab === "zone" ? <ZoneView o={o} /> : <TripView o={o} />
      )}</Async>
    </>
  );
}

function band(d: Pt[], color: string) {
  if (!d.length || d[0].lower === undefined) return [];
  return [
    { x: d.map((p) => p.ts), y: d.map((p) => p.upper ?? null), type: "scatter" as const, mode: "lines" as const, line: { width: 0 }, showlegend: false, hoverinfo: "skip" as const },
    { x: d.map((p) => p.ts), y: d.map((p) => p.lower ?? null), type: "scatter" as const, mode: "lines" as const, line: { width: 0 }, fill: "tonexty" as const, fillcolor: color, name: "80% interval" },
  ];
}

function SegmentView({ o }: { o: Options }) {
  const [h, setH] = useState(o.speed_horizons_min.includes(60) ? 60 : o.speed_horizons_min[0]);
  const [seg, setSeg] = useState<number | null>(null);
  const fcst = useApi<SegFc>(h ? `/api/predict/segments${qs({ horizon: h })}` : null);
  const det = useApi<SegDetail>(h && seg ? `/api/predict/segment/${seg}${qs({ horizon: h })}` : null);
  useEffect(() => { if (!seg && o.segments.length) setSeg(o.segments[0].segment_id); }, [o.segments, seg]);

  const layer = useMemo(() => [{
    id: "fc", data: fc((fcst.data?.segments ?? []).map((s) => line([[s.start_longitude, s.start_latitude], [s.end_longitude, s.end_latitude]],
      { id: s.segment_id, street: s.street, sp: s.predicted_speed, lo: s.speed_lower, hi: s.speed_upper, level: s.predicted_level, conf: s.confidence, color: LEVEL_COLORS[s.predicted_level ?? ""] ?? "#5B9BFF" }))),
    layers: [{ id: "fc-line", type: "line", paint: { "line-color": ["get", "color"], "line-width": 3.5 } }],
    popup: (p: Record<string, unknown>) => `<b>${p.street}</b>${kv([["Predicted speed", `${fmt(Number(p.sp), 1)} mph`], ["80% interval", `${fmt(Number(p.lo), 1)}–${fmt(Number(p.hi), 1)}`], ["Level", p.level], ["Confidence", pct(Number(p.conf))]])}`,
    onClick: (p: Record<string, unknown>) => setSeg(Number(p.id)),
  }], [fcst.data]);

  if (!o.speed_horizons_min.length) return <Notice tone="warn">No speed models trained yet. Run <code>python -m src.pipeline train --model traffic</code>.</Notice>;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Horizon"><Segmented value={h} onChange={setH} options={o.speed_horizons_min.map((m) => ({ value: m, label: hLabel(m) }))} /></Field>
        <Field label="Road segment (Chicago)">
          <select value={seg ?? ""} onChange={(e) => setSeg(Number(e.target.value))} className="min-w-[320px]">
            {o.segments.map((s) => <option key={s.segment_id} value={s.segment_id}>{s.street} {s.direction} · {s.from_street} → {s.to_street} (#{s.segment_id})</option>)}
          </select>
        </Field>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
        <Card title={`Forecast map · +${hLabel(h)}`} subtitle={fcst.data ? `Issued at archive end ${fmtTs(fcst.data.issued_at)} → target ${fmtTs(fcst.data.target_time)}` : ""}>
          <Async state={fcst} h={440}>{(d) => (<><MapView layers={layer} {...CENTERS.chicago} height={440} /><p className="mt-2 text-xs text-slate-500">{d.note}</p></>)}</Async>
        </Card>
        <Card title="Actual vs predicted speed" subtitle="Held-out test period for the selected segment">
          <Async state={det} h={440}>{(d) => (
            <>
              <div className="mb-3 grid grid-cols-4 gap-2">
                <Stat label="Segment MAE" value={fmt(d.segment_metrics.mae, 2)} sub="mph" />
                <Stat label="Segment RMSE" value={fmt(d.segment_metrics.rmse, 2)} sub="mph" />
                <Stat label="All-segment MAE" value={fmt(d.overall_test_metrics.mae, 2)} sub="mph" />
                <Stat label="Interval coverage" value={d.intervals ? pct(d.intervals.empirical_test_coverage) : "—"} sub="nominal 80%" />
              </div>
              <Plot height={320} data={[...band(d.speed, "rgba(139,92,246,0.18)"),
                { x: d.speed.map((p) => p.ts), y: d.speed.map((p) => p.actual), type: "scatter", mode: "lines", name: "Actual", line: { color: "#E2E8F0", width: 1.4 } },
                { x: d.speed.map((p) => p.ts), y: d.speed.map((p) => p.predicted), type: "scatter", mode: "lines", name: "Predicted", line: { color: "#3B82F6", width: 1.6 } },
                ...(d.forecast?.length ? [{ x: d.forecast.map((p) => p.ts), y: d.forecast.map((p) => p.predicted), type: "scatter" as const, mode: "lines+markers" as const, name: "Forecast", line: { color: "#22D3EE", dash: "dot" as const } }] : []),
              ]} layout={{ yaxis: { title: { text: "mph" } } }} />
            </>)}</Async>
        </Card>
      </div>
      {det.data?.congestion && det.data.congestion.length > 0 && (
        <Card title="Congestion level: actual vs predicted" subtitle="Classification model on the same segment and horizon">
          <Plot height={220} data={[
            { x: det.data.congestion.map((c) => c.ts), y: det.data.congestion.map((c) => c.actual), type: "scatter", mode: "markers", name: "Actual", marker: { size: 5, color: "#E2E8F0" } },
            { x: det.data.congestion.map((c) => c.ts), y: det.data.congestion.map((c) => c.predicted), type: "scatter", mode: "markers", name: "Predicted", marker: { size: 5, color: "#3B82F6", symbol: "x" } },
          ]} layout={{ yaxis: { categoryorder: "array", categoryarray: ["Low", "Moderate", "High", "Severe"] } }} />
        </Card>
      )}
    </div>
  );
}

function ZoneView({ o }: { o: Options }) {
  const [h, setH] = useState(o.demand_horizons_h[0]);
  const [zone, setZone] = useState<number>(o.zones.find((z) => z.zone === 161)?.zone ?? o.zones[0]?.zone ?? 1);
  const [borough, setBorough] = useState("Manhattan");
  const det = useApi<ZoneDetail>(h ? `/api/predict/zone/${zone}${qs({ horizon: h, days: 14 })}` : null);
  if (!o.demand_horizons_h.length) return <Notice tone="warn">No demand models trained yet. Run <code>python -m src.pipeline train --model demand</code>.</Notice>;
  const boroughs = [...new Set(o.zones.map((z) => z.borough))].filter(Boolean).sort();
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Horizon"><Segmented value={h} onChange={setH} options={o.demand_horizons_h.map((x) => ({ value: x, label: `${x} h` }))} /></Field>
        <Field label="Borough"><select value={borough} onChange={(e) => setBorough(e.target.value)}>{boroughs.map((b) => <option key={b}>{b}</option>)}</select></Field>
        <Field label="Taxi zone">
          <select value={zone} onChange={(e) => setZone(Number(e.target.value))} className="min-w-[260px]">
            {o.zones.filter((z) => z.borough === borough).map((z) => <option key={z.zone} value={z.zone}>{z.zone_name} ({z.zone})</option>)}
          </select>
        </Field>
      </div>
      <Card title="Hourly pickups: actual vs predicted" subtitle="Last 14 days of the test period, then the forward forecast">
        <Async state={det} h={400}>{(d) => (
          <>
            <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-4">
              <Stat label="Zone MAE" value={fmt(d.zone_metrics.mae, 2)} sub="pickups / hour" />
              <Stat label="Zone RMSE" value={fmt(d.zone_metrics.rmse, 2)} />
              <Stat label="R²" value={fmt(d.zone_metrics.r2, 3)} />
              <Stat label="Interval coverage" value={d.intervals ? pct(d.intervals.empirical_test_coverage) : "—"} sub="nominal 80% (all zones)" />
            </div>
            <Plot height={320} data={[...band(d.series, "rgba(34,211,238,0.12)"),
              { x: d.series.map((p) => p.ts), y: d.series.map((p) => p.actual), type: "scatter", mode: "lines", name: "Actual", line: { color: "#E2E8F0", width: 1.3 } },
              { x: d.series.map((p) => p.ts), y: d.series.map((p) => p.predicted), type: "scatter", mode: "lines", name: "Predicted", line: { color: "#3B82F6", width: 1.6 } },
              ...band(d.forecast, "rgba(139,92,246,0.22)"),
              { x: d.forecast.map((p) => p.ts), y: d.forecast.map((p) => p.predicted), type: "scatter", mode: "lines+markers", name: "Forecast", line: { color: "#8B5CF6" } },
            ]} />
          </>)}</Async>
      </Card>
    </div>
  );
}

interface TT { eta_minutes: number; baseline_eta_minutes: number | null; distance_mi: number; distance_source: string; weather_used: Record<string, number | null> | null; model: string; model_version: string; test_mae_minutes: number | null; note: string }

function TripView({ o }: { o: Options }) {
  const [pu, setPu] = useState(161);
  const [dz, setDz] = useState(132);
  const [dep, setDep] = useState(() => new Date().toISOString().slice(0, 16));
  const [dist, setDist] = useState("");
  const [res, setRes] = useState<TT>();
  const [err, setErr] = useState<string>();
  const [busy, setBusy] = useState(false);
  if (!o.travel_time_available) return <Notice tone="warn">Travel-time model not trained yet. Run <code>python -m src.pipeline train --model travel_time</code>.</Notice>;
  const submit = async () => {
    setBusy(true); setErr(undefined);
    try {
      setRes(await api<TT>("/api/predict/travel-time", { method: "POST", body: JSON.stringify({ pu_zone: pu, do_zone: dz, departure: dep, distance_mi: dist ? Number(dist) : null }) }));
    } catch (e) {
      const nr = e instanceof ApiError ? e.notReady : null;
      setErr(nr ? nr.hint : (e as Error).message); setRes(undefined);
    } finally { setBusy(false); }
  };
  const zoneSel = (v: number, set: (n: number) => void) => (
    <select value={v} onChange={(e) => set(Number(e.target.value))} className="w-full">
      {o.zones.map((z) => <option key={z.zone} value={z.zone}>{z.borough} · {z.zone_name}</option>)}
    </select>);
  return (
    <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
      <Card title="Journey" subtitle="NYC taxi travel time between TLC zones">
        <div className="space-y-3">
          <Field label="Pickup zone">{zoneSel(pu, setPu)}</Field>
          <Field label="Drop-off zone">{zoneSel(dz, setDz)}</Field>
          <Field label="Departure (local NYC time)"><input type="datetime-local" value={dep} onChange={(e) => setDep(e.target.value)} /></Field>
          <Field label="Distance, miles (optional)"><input value={dist} onChange={(e) => setDist(e.target.value)} placeholder="historical OD median if empty" /></Field>
          <button onClick={submit} disabled={busy} className="w-full rounded-md bg-electric-500 py-2 text-sm font-medium text-white hover:bg-electric-600 disabled:opacity-50">{busy ? "Estimating…" : "Estimate travel time"}</button>
          {err && <Notice tone="warn">{err}</Notice>}
        </div>
      </Card>
      <Card title="Estimate">
        {res ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat label="Predicted duration" value={`${fmt(res.eta_minutes, 1)} min`} accent="text-signal-cyan" sub={res.test_mae_minutes != null ? `test MAE ± ${fmt(res.test_mae_minutes, 1)} min` : undefined} />
              <Stat label="Naive baseline" value={res.baseline_eta_minutes != null ? `${fmt(res.baseline_eta_minutes, 1)} min` : "—"} sub="distance ÷ city median speed at that hour" />
              <Stat label="Distance" value={`${fmt(res.distance_mi, 2)} mi`} sub={res.distance_source.replaceAll("_", " ")} />
              <Stat label="Model" value={<span className="text-base">{res.model}</span>} sub={res.model_version} />
            </div>
            {res.weather_used && <Notice>Weather at departure hour: {Object.entries(res.weather_used).map(([k, v]) => `${k.replace("_2m", "").replace("_10m", "")} ${v == null ? "n/a" : fmt(v, 1)}`).join(" · ")}</Notice>}
            <p className="text-xs text-slate-500">{res.note}</p>
          </div>
        ) : <p className="text-sm text-slate-400">Choose an origin, destination and departure time.</p>}
      </Card>
    </div>
  );
}
