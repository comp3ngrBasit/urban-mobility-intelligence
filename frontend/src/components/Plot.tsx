import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";

const PlotlyComponent = createPlotlyComponent(Plotly);

const AXIS = {
  gridcolor: "rgba(148,163,184,0.08)",
  zerolinecolor: "rgba(148,163,184,0.15)",
  linecolor: "rgba(148,163,184,0.2)",
  tickfont: { color: "#94A3B8", size: 11 },
  title: { font: { color: "#94A3B8", size: 11 } },
};

export const SERIES = ["#3B82F6", "#22D3EE", "#8B5CF6", "#F5A524", "#2DD4A4", "#F2545B"];

export default function Plot({ data, layout, height = 300 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  const base: Partial<Layout> = {
    height,
    margin: { l: 52, r: 16, t: 12, b: 40 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", color: "#CBD5E1", size: 12 },
    colorway: SERIES,
    hoverlabel: { bgcolor: "#131D33", bordercolor: "#2A3858", font: { color: "#E2E8F0" } },
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11, color: "#94A3B8" } },
    ...layout,
    xaxis: { ...AXIS, ...(layout?.xaxis ?? {}) },
    yaxis: { ...AXIS, ...(layout?.yaxis ?? {}) },
  };
  return (
    <PlotlyComponent
      data={data}
      layout={base}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
