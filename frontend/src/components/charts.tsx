import Plot from "./Plot";
import { DOW } from "../lib/format";
import type { CI, Heatmap } from "../lib/types";

/** Hourly mean with 95% CI ribbon, one trace per day type. */
export function HourlyCI({ rows, unit, height = 280 }: { rows: (CI & { daytype: string; hour: number })[]; unit: string; height?: number }) {
  const colors: Record<string, [string, string]> = { weekday: ["#3B82F6", "rgba(59,130,246,0.15)"], weekend: ["#8B5CF6", "rgba(139,92,246,0.15)"] };
  const traces = Object.keys(colors).flatMap((dt) => {
    const r = rows.filter((x) => x.daytype === dt).sort((a, b) => a.hour - b.hour);
    if (!r.length) return [];
    const [c, f] = colors[dt];
    return [
      { x: r.map((x) => x.hour), y: r.map((x) => x.hi), type: "scatter" as const, mode: "lines" as const, line: { width: 0 }, showlegend: false, hoverinfo: "skip" as const },
      { x: r.map((x) => x.hour), y: r.map((x) => x.lo), type: "scatter" as const, mode: "lines" as const, line: { width: 0 }, fill: "tonexty" as const, fillcolor: f, showlegend: false, hoverinfo: "skip" as const },
      { x: r.map((x) => x.hour), y: r.map((x) => x.mean), type: "scatter" as const, mode: "lines+markers" as const, name: dt, line: { color: c, width: 2 }, marker: { size: 4 } },
    ];
  });
  return <Plot height={height} data={traces} layout={{ xaxis: { title: { text: "hour of day" }, dtick: 3 }, yaxis: { title: { text: unit } } }} />;
}

export function DowHourHeatmap({ h, colorscale = "Blues", height = 260, zTitle }: { h: Heatmap; colorscale?: string | [number, string][]; height?: number; zTitle?: string }) {
  return (
    <Plot height={height} data={[{ z: h.z, x: h.hours, y: h.dows.map((d) => DOW[d]), type: "heatmap", colorscale,
      colorbar: { thickness: 10, tickfont: { color: "#94A3B8", size: 10 }, title: { text: zTitle ?? "", font: { size: 10 } } }, hovertemplate: "%{y} %{x}:00<br>%{z:.2f}<extra></extra>" }]}
      layout={{ xaxis: { dtick: 3, title: { text: "hour" } }, yaxis: { autorange: "reversed" } }} />
  );
}

export const NAVY_SCALE: [number, string][] = [[0, "#0F1729"], [0.35, "#1E3A8A"], [0.7, "#3B82F6"], [1, "#22D3EE"]];
export const HEAT_SCALE: [number, string][] = [[0, "#F2545B"], [0.5, "#F5A524"], [1, "#22D3EE"]];
