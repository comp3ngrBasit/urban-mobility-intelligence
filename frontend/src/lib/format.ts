export const fmtInt = (v?: number | null) => (v == null || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString("en-US"));
export const fmt = (v?: number | null, d = 2) =>
  v == null || Number.isNaN(v) ? "—" : v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
export const pct = (v?: number | null, d = 1) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
export const pval = (p?: number | null) => (p == null ? "—" : p < 0.001 ? "< 0.001" : p.toFixed(3));
export const fmtTs = (s?: string | null) =>
  s ? new Date(s).toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";
export const fmtDate = (s?: string | null) =>
  s ? new Date(s).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—";
export const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export const LEVEL_COLORS: Record<string, string> = {
  Low: "#2DD4A4",
  Moderate: "#F5A524",
  High: "#FB7A3C",
  Severe: "#F2545B",
};
export const SEVERITY_COLORS: Record<string, string> = {
  "No injury": "#5B9BFF",
  Minor: "#22D3EE",
  Serious: "#F5A524",
  Fatal: "#F2545B",
};

/** Speed ratio (observed / typical) -> colour on a red-amber-cyan ramp. */
export const ratioColor = (r?: number | null) => {
  if (r == null || Number.isNaN(r)) return "#475569";
  if (r < 0.4) return "#F2545B";
  if (r < 0.6) return "#FB7A3C";
  if (r < 0.8) return "#F5A524";
  return "#22D3EE";
};
