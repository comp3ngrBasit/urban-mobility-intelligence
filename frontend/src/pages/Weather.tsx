import { useState } from "react";
import Plot from "../components/Plot";
import { Async, Card, Notice, PageHeader, Segmented } from "../components/ui";
import { useApi } from "../lib/api";
import { fmt, fmtInt, pct, pval } from "../lib/format";
import type { Anova, Corr, TwoGroup } from "../lib/types";

interface W {
  note: string;
  chicago_speed?: { precip_vs_speed_anomaly: Corr; temp_vs_speed_anomaly: Corr; wind_vs_speed_anomaly: Corr; rain_intensity_anova_peak: Anova;
    by_category: { weather_category: string; mean: number; count: number }[]; scatter: { precipitation: number; speed_anom: number; temperature_2m: number }[] };
  nyc_demand?: { precip_vs_demand_anomaly: Corr; temp_vs_demand_anomaly: Corr; rain_vs_dry: TwoGroup };
  chicago_crashes?: { precip_vs_crash_anomaly: Corr; rain_vs_dry_crash_rate: TwoGroup; snow_vs_none: TwoGroup | null; reported_weather_vs_serious: { chi2?: number; p?: number; cramers_v?: number };
    by_reported_weather: { weather_condition: string; crashes: number; serious_rate: number }[] };
}
interface ByCond { horizon_min: number; by_condition: Record<string, { condition: string; n: number; actual_mean_speed: number; predicted_mean_speed: number; model_mae: number }[]> }

function CorrRow({ label, c }: { label: string; c?: Corr }) {
  if (!c) return null;
  const sig = (c.pearson_p ?? 1) < 0.05;
  return (
    <tr><td>{label}</td><td className="num text-right">{fmtInt(c.n)}</td>
      <td className={`num text-right ${sig ? "text-white" : "text-slate-500"}`}>{fmt(c.pearson_r, 3)}</td>
      <td className="num text-right text-xs text-slate-400">{c.pearson_ci95 ? `[${fmt(c.pearson_ci95[0], 3)}, ${fmt(c.pearson_ci95[1], 3)}]` : "—"}</td>
      <td className="num text-right">{pval(c.pearson_p)}</td><td className="num text-right">{fmt(c.spearman_rho, 3)}</td><td className="num text-right">{pval(c.spearman_p)}</td></tr>
  );
}

export default function Weather() {
  const w = useApi<W>("/api/weather/impact");
  const opts = useApi<{ speed_horizons_min: number[] }>("/api/predict/options");
  const [h, setH] = useState(60);
  const [grp, setGrp] = useState<"rain_intensity" | "weather_category">("rain_intensity");
  const bc = useApi<ByCond>(opts.data?.speed_horizons_min.length ? `/api/weather/predictions-by-condition?horizon=${h}` : null);
  return (
    <>
      <PageHeader title="Weather Impact Analysis" description="Hourly Open-Meteo weather joined to Chicago speeds, NYC taxi demand and Chicago crashes. Outcomes are de-seasonalised (deviation from the hour × weekday mean) before correlating." />
      <Async state={w} h={600}>{(d) => (
        <div className="space-y-5">
          <Notice>{d.note}</Notice>
          <Card title="Correlation tests" subtitle="Pearson r with Fisher-z 95% CI, Spearman ρ; bold = p < 0.05">
            <table className="data"><thead><tr><th>Relationship</th><th className="text-right">n (hours)</th><th className="text-right">Pearson r</th><th className="text-right">95% CI</th><th className="text-right">p</th><th className="text-right">Spearman ρ</th><th className="text-right">p</th></tr></thead>
              <tbody>
                <CorrRow label="Precipitation → Chicago speed anomaly" c={d.chicago_speed?.precip_vs_speed_anomaly} />
                <CorrRow label="Temperature → Chicago speed anomaly" c={d.chicago_speed?.temp_vs_speed_anomaly} />
                <CorrRow label="Wind → Chicago speed anomaly" c={d.chicago_speed?.wind_vs_speed_anomaly} />
                <CorrRow label="Precipitation → NYC taxi demand anomaly" c={d.nyc_demand?.precip_vs_demand_anomaly} />
                <CorrRow label="Temperature → NYC taxi demand anomaly" c={d.nyc_demand?.temp_vs_demand_anomaly} />
                <CorrRow label="Precipitation → Chicago crash anomaly" c={d.chicago_crashes?.precip_vs_crash_anomaly} />
              </tbody></table>
          </Card>
          <div className="grid gap-5 xl:grid-cols-2">
            {d.chicago_speed && (
              <Card title="Rainfall and peak-hour speed" subtitle={d.chicago_speed.rain_intensity_anova_peak.anova_F != null
                ? `One-way ANOVA F = ${fmt(d.chicago_speed.rain_intensity_anova_peak.anova_F, 2)}, p ${pval(d.chicago_speed.rain_intensity_anova_peak.anova_p)}, η² = ${fmt(d.chicago_speed.rain_intensity_anova_peak.eta_squared, 4)} · Kruskal p ${pval(d.chicago_speed.rain_intensity_anova_peak.kruskal_p)}` : d.chicago_speed.rain_intensity_anova_peak.note}>
                {d.chicago_speed.rain_intensity_anova_peak.groups && (() => {
                  const order = ["None", "Light", "Moderate", "Heavy"].filter((k) => d.chicago_speed!.rain_intensity_anova_peak.groups![k]);
                  const g = d.chicago_speed!.rain_intensity_anova_peak.groups!;
                  return <Plot height={280} data={[{ x: order, y: order.map((k) => g[k].mean), type: "bar", marker: { color: ["#22D3EE", "#3B82F6", "#8B5CF6", "#F2545B"] },
                    error_y: { type: "data", symmetric: false, array: order.map((k) => (g[k].hi ?? 0) - (g[k].mean ?? 0)), arrayminus: order.map((k) => (g[k].mean ?? 0) - (g[k].lo ?? 0)), color: "#94A3B8" },
                    text: order.map((k) => `n=${g[k].n}`), textposition: "outside" }]} layout={{ yaxis: { title: { text: "mean speed (mph), weekday peak" } } }} />;
                })()}
              </Card>
            )}
            {d.chicago_speed && (
              <Card title="Precipitation vs speed anomaly" subtitle="Sample of hourly observations, coloured by temperature">
                <Plot height={280} data={[{ x: d.chicago_speed.scatter.map((p) => p.precipitation), y: d.chicago_speed.scatter.map((p) => p.speed_anom), type: "scattergl", mode: "markers",
                  marker: { size: 4, opacity: 0.6, color: d.chicago_speed.scatter.map((p) => p.temperature_2m), colorscale: [[0, "#3B82F6"], [1, "#F5A524"]], colorbar: { title: { text: "°C" }, thickness: 8 } } }]}
                  layout={{ xaxis: { title: { text: "precipitation (mm/h)" } }, yaxis: { title: { text: "speed − typical (mph)" } } }} />
              </Card>
            )}
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            {d.chicago_crashes && (
              <Card title="Crashes by police-reported weather" subtitle={d.chicago_crashes.reported_weather_vs_serious.chi2 != null ? `Weather × serious injury: χ² = ${fmt(d.chicago_crashes.reported_weather_vs_serious.chi2, 1)}, p ${pval(d.chicago_crashes.reported_weather_vs_serious.p)}, V = ${fmt(d.chicago_crashes.reported_weather_vs_serious.cramers_v, 3)}` : ""}>
                <table className="data"><thead><tr><th>Weather</th><th className="text-right">Crashes</th><th className="text-right">Serious rate</th></tr></thead>
                  <tbody>{d.chicago_crashes.by_reported_weather.slice(0, 9).map((r) => <tr key={r.weather_condition}><td>{r.weather_condition}</td><td className="num text-right">{fmtInt(r.crashes)}</td><td className="num text-right">{pct(r.serious_rate, 2)}</td></tr>)}</tbody></table>
                <div className="mt-3 text-xs text-slate-400">Hourly crash count, rain vs dry hours: Welch p {pval(d.chicago_crashes.rain_vs_dry_crash_rate.welch_p)}, Cohen's d {fmt(d.chicago_crashes.rain_vs_dry_crash_rate.cohens_d ?? null, 3)}</div>
              </Card>
            )}
            <Card title="Model predictions under observed weather" subtitle="Speed model on the held-out test period, grouped by the weather at issue time"
              right={<div className="flex gap-2">
                {opts.data?.speed_horizons_min.length ? <Segmented value={h} onChange={setH} options={opts.data.speed_horizons_min.map((m) => ({ value: m, label: m >= 60 ? `${m / 60}h` : `${m}m` }))} /> : null}
                <Segmented value={grp} onChange={setGrp} options={[{ value: "rain_intensity", label: "Rain" }, { value: "weather_category", label: "Category" }]} /></div>}>
              <Async state={bc} h={260}>{(b) => {
                const rows = b.by_condition[grp] ?? [];
                return rows.length ? (
                  <Plot height={260} data={[
                    { x: rows.map((r) => r.condition), y: rows.map((r) => r.actual_mean_speed), type: "bar", name: "Actual mean", marker: { color: "rgba(226,232,240,0.6)" } },
                    { x: rows.map((r) => r.condition), y: rows.map((r) => r.predicted_mean_speed), type: "bar", name: "Predicted mean", marker: { color: "#3B82F6" } },
                    { x: rows.map((r) => r.condition), y: rows.map((r) => r.model_mae), type: "scatter", mode: "markers+lines", name: "MAE", yaxis: "y2", line: { color: "#F5A524" } },
                  ]} layout={{ barmode: "group", yaxis: { title: { text: "mph" } }, yaxis2: { overlaying: "y", side: "right", showgrid: false, title: { text: "MAE" } } }} />
                ) : <Notice>No weather groups with ≥ 20 test rows.</Notice>;
              }}</Async>
            </Card>
          </div>
        </div>)}</Async>
    </>
  );
}
