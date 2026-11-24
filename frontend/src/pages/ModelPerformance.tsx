import { useEffect, useState } from "react";
import Plot from "../components/Plot";
import { Async, Card, Notice, PageHeader, Stat } from "../components/ui";
import { useApi } from "../lib/api";
import { fmt, fmtDate, fmtInt, pct } from "../lib/format";
import type { ModelSummary } from "../lib/types";

interface Detail {
  task: string; version: string;
  metrics: { split: Record<string, unknown>; best_model: string; models: Record<string, { params: Record<string, unknown>; valid_score: number; cv_scores: number[]; cv_mean: number; cv_std: number; test: Record<string, unknown> }>; baselines: Record<string, Record<string, unknown>> };
  feature_config: { features: string[]; kind: string; labels?: string[]; target: string; horizon_min?: number; horizon_h?: number };
  metadata: Record<string, unknown>; importance: { feature: string; importance: number }[] | null;
  shap: { mean_abs_shap?: { feature: string; value: number }[]; beeswarm?: { feature: string; shap: number[]; feature_value: number[] }[]; error?: string } | null;
  errors: { hour?: { hour: number; mae: number; bias: number }[]; residual_hist?: { counts: number[]; edges: number[] } } | null;
  intervals: { nominal: number; empirical_test_coverage: number; mean_width?: number } | null;
}

const REG = ["mae", "rmse", "r2", "mape_pct"] as const;
const CLF = ["f1_macro", "accuracy", "roc_auc_ovr_macro"] as const;

export default function ModelPerformance() {
  const list = useApi<ModelSummary[]>("/api/models");
  const [task, setTask] = useState<string>();
  useEffect(() => { if (!task && list.data?.length) setTask(list.data.find((m) => m.kind !== "forecast_classical")?.task ?? list.data[0].task); }, [list.data, task]);
  const det = useApi<Detail>(task && list.data?.find((m) => m.task === task)?.kind !== "forecast_classical" ? `/api/models/${task}` : null);
  const preds = useApi<{ rows: Record<string, number | string>[] }>(task && det.data ? `/api/models/${task}/predictions?sample=2500` : null);

  return (
    <>
      <PageHeader title="Model Performance" description="All metrics are computed on a chronologically held-out test period never used for training or model selection (selection uses the validation period). Baselines are evaluated on the same rows." />
      <Async state={list} h={400}>{(ms) => (
        <div className="space-y-5">
          <Card title="Model registry" subtitle="Best candidate per task vs naive baselines (test period)">
            <div className="overflow-auto"><table className="data"><thead><tr><th>Task</th><th>Best model</th><th>Test period</th><th className="text-right">Primary metric</th><th className="text-right">Best baseline</th><th className="text-right">Lift</th><th>Version</th></tr></thead>
              <tbody>{ms.filter((m) => m.kind !== "forecast_classical").map((m) => {
                const key = m.kind === "clf" ? "f1_macro" : "mae";
                const v = m.best_test?.[key] ?? null;
                const bases = Object.entries(m.baselines ?? {}).map(([k, b]) => [k, b[key]] as const).filter(([, x]) => x != null);
                const best = bases.length ? bases.reduce((a, b) => (m.kind === "clf" ? (b[1]! > a[1]! ? b : a) : (b[1]! < a[1]! ? b : a))) : null;
                const lift = best && v != null && best[1] ? (m.kind === "clf" ? v / best[1] - 1 : 1 - v / best[1]) : null;
                return (
                  <tr key={m.task} onClick={() => setTask(m.task)} className={`cursor-pointer hover:bg-ink-800 ${task === m.task ? "bg-electric-500/10" : ""}`}>
                    <td className="font-medium">{m.task}</td><td>{m.best_model}</td><td className="text-xs text-slate-400">{m.test_period ? `${fmtDate(m.test_period[0])} – ${fmtDate(m.test_period[1])}` : ""}</td>
                    <td className="num text-right">{key.toUpperCase()} {fmt(v, 3)}</td><td className="num text-right text-slate-400">{best ? `${best[0]}: ${fmt(best[1], 3)}` : "—"}</td>
                    <td className={`num text-right ${lift != null && lift > 0 ? "text-signal-green" : "text-signal-red"}`}>{lift != null ? pct(lift) : "—"}</td>
                    <td className="text-xs text-slate-500">{m.version}</td></tr>);
              })}</tbody></table></div>
            {ms.some((m) => m.kind === "forecast_classical") && <ClassicalTable m={ms.find((x) => x.kind === "forecast_classical")!} />}
          </Card>
          {task && <Async state={det} h={500}>{(d) => <DetailView d={d} preds={preds.data?.rows} />}</Async>}
        </div>)}</Async>
    </>
  );
}

function ClassicalTable({ m }: { m: ModelSummary }) {
  const rows = Object.entries(m.aggregate ?? {});
  return (
    <div className="mt-5">
      <div className="label mb-2">Classical forecasting on the busiest zones (rolling origin)</div>
      <table className="data"><thead><tr><th>Model · horizon</th><th className="text-right">MAE</th><th className="text-right">RMSE</th><th className="text-right">R²</th></tr></thead>
        <tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td className="num text-right">{fmt(v.mae as number, 2)}</td><td className="num text-right">{fmt(v.rmse as number, 2)}</td><td className="num text-right">{fmt(v.r2 as number, 3)}</td></tr>)}</tbody></table>
    </div>
  );
}

function DetailView({ d, preds }: { d: Detail; preds?: Record<string, number | string>[] }) {
  const clf = d.feature_config.kind === "clf";
  const best = d.metrics.models[d.metrics.best_model];
  const keys = clf ? CLF : REG;
  const cm = clf ? (best.test.confusion_matrix as { labels: string[]; matrix: number[][] }) : null;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        {keys.map((k) => <Stat key={k} label={k.replace("_", " ")} value={fmt(best.test[k] as number | null, 3)} />)}
        <Stat label="Time-series CV" value={fmt(best.cv_mean, 3)} sub={`± ${fmt(best.cv_std, 3)} over ${best.cv_scores.length} folds (${clf ? "macro-F1" : "−RMSE"})`} />
        {d.intervals && <Stat label="80% interval coverage" value={pct(d.intervals.empirical_test_coverage)} sub={d.intervals.mean_width ? `mean width ${fmt(d.intervals.mean_width, 1)}` : undefined} />}
      </div>
      <Card title="Candidate comparison" subtitle={`Training rows ${fmtInt(d.metrics.split.train_rows as number)} · validation ${fmtInt(d.metrics.split.valid_rows as number)} · test ${fmtInt(d.metrics.split.test_rows as number)}`}>
        <table className="data"><thead><tr><th>Model</th>{keys.map((k) => <th key={k} className="text-right">{k}</th>)}<th className="text-right">CV mean ± sd</th><th>Tuned params</th></tr></thead>
          <tbody>
            {Object.entries(d.metrics.models).map(([n, m]) => <tr key={n} className={n === d.metrics.best_model ? "text-white" : "text-slate-300"}><td>{n}{n === d.metrics.best_model && <span className="ml-2 rounded bg-electric-500/20 px-1.5 text-[10px] text-electric-400">selected</span>}</td>
              {keys.map((k) => <td key={k} className="num text-right">{fmt(m.test[k] as number | null, 3)}</td>)}<td className="num text-right">{fmt(m.cv_mean, 3)} ± {fmt(m.cv_std, 3)}</td>
              <td className="max-w-[320px] truncate font-mono text-[10px] text-slate-500">{JSON.stringify(m.params)}</td></tr>)}
            {Object.entries(d.metrics.baselines).map(([n, m]) => <tr key={n} className="text-slate-500"><td>baseline · {n}</td>{keys.map((k) => <td key={k} className="num text-right">{fmt(m[k] as number | null, 3)}</td>)}<td /><td /></tr>)}
          </tbody></table>
      </Card>
      <div className="grid gap-5 xl:grid-cols-2">
        {clf && cm ? (
          <Card title="Confusion matrix" subtitle="Rows = actual, columns = predicted (test period)">
            <Plot height={340} data={[{ z: cm.matrix.map((r) => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / s); }), x: cm.labels, y: cm.labels, type: "heatmap",
              colorscale: [[0, "#0F1729"], [1, "#3B82F6"]], text: cm.matrix as unknown as string[][], texttemplate: "%{text}", hovertemplate: "actual %{y} → %{x}<br>%{z:.1%}<extra></extra>", showscale: false }]}
              layout={{ margin: { l: 90, r: 10, t: 10, b: 50 }, yaxis: { autorange: "reversed", title: { text: "actual", standoff: 12 } }, xaxis: { title: { text: "predicted" } } }} />
          </Card>
        ) : preds && (
          <Card title="Actual vs predicted" subtitle="Random sample of test rows">
            <Plot height={340} data={[{ x: preds.map((r) => r.actual as number), y: preds.map((r) => r.predicted as number), type: "scattergl", mode: "markers", marker: { size: 3, color: "#3B82F6", opacity: 0.4 }, name: "rows" },
              { x: [0, Math.max(...preds.map((r) => r.actual as number))], y: [0, Math.max(...preds.map((r) => r.actual as number))], type: "scatter", mode: "lines", line: { color: "#94A3B8", dash: "dash" }, name: "y = x" }]}
              layout={{ xaxis: { title: { text: "actual" } }, yaxis: { title: { text: "predicted" } } }} />
          </Card>
        )}
        <Card title="Feature importance" subtitle="Model-native (gain / impurity), normalised">
          {d.importance?.length ? <Plot height={340} data={[{ y: d.importance.slice(0, 15).map((f) => f.feature), x: d.importance.slice(0, 15).map((f) => f.importance), type: "bar", orientation: "h", marker: { color: "#22D3EE" } }]}
            layout={{ margin: { l: 170, r: 10, t: 10, b: 30 }, yaxis: { autorange: "reversed" } }} /> : <Notice>Not available for this estimator.</Notice>}
        </Card>
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Card title="SHAP explanations" subtitle={clf ? `TreeExplainer, class "${d.feature_config.labels?.[0]}" · each dot one test row coloured by feature value` : "TreeExplainer · each dot one test row coloured by feature value"}>
          {d.shap?.beeswarm ? (
            <Plot height={380} data={d.shap.beeswarm.slice(0, 10).reverse().map((b, i) => {
              const fv = b.feature_value.filter((v) => v != null && !Number.isNaN(v)); const lo = Math.min(...fv), hi = Math.max(...fv);
              return { x: b.shap, y: b.shap.map(() => i + (Math.random() - 0.5) * 0.5), type: "scattergl" as const, mode: "markers" as const, name: b.feature, showlegend: false,
                marker: { size: 3.5, color: b.feature_value.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5)), colorscale: [[0, "#3B82F6"], [1, "#F2545B"]] as [number, string][], cmin: 0, cmax: 1 } };
            })} layout={{ margin: { l: 170, r: 10, t: 10, b: 36 }, xaxis: { title: { text: "SHAP value" } },
              yaxis: { tickvals: d.shap.beeswarm.slice(0, 10).map((_, i) => i), ticktext: d.shap.beeswarm.slice(0, 10).reverse().map((b) => b.feature) } }} />
          ) : <Notice>{d.shap?.error ?? "SHAP not available."}</Notice>}
        </Card>
        {!clf && d.errors ? (
          <Card title="Error analysis" subtitle="Mean absolute error and bias by hour of day; residual distribution">
            {d.errors.hour && <Plot height={190} data={[{ x: [...d.errors.hour].sort((a, b) => a.hour - b.hour).map((r) => r.hour), y: [...d.errors.hour].sort((a, b) => a.hour - b.hour).map((r) => r.mae), type: "bar", name: "MAE", marker: { color: "#8B5CF6" } },
              { x: [...d.errors.hour].sort((a, b) => a.hour - b.hour).map((r) => r.hour), y: [...d.errors.hour].sort((a, b) => a.hour - b.hour).map((r) => r.bias), type: "scatter", mode: "lines", name: "bias", line: { color: "#F5A524" } }]} layout={{ xaxis: { dtick: 3 } }} />}
            {d.errors.residual_hist && <Plot height={180} data={[{ x: d.errors.residual_hist.edges.slice(0, -1), y: d.errors.residual_hist.counts, type: "bar", marker: { color: "#3B82F6" } }]} layout={{ xaxis: { title: { text: "predicted − actual" } } }} />}
          </Card>
        ) : (
          <Card title="Per-class performance">
            <table className="data"><thead><tr><th>Class</th><th className="text-right">Precision</th><th className="text-right">Recall</th><th className="text-right">F1</th><th className="text-right">Support</th></tr></thead>
              <tbody>{(d.feature_config.labels ?? []).map((l) => { const r = (best.test.per_class as Record<string, Record<string, number>>)[l]; return r ? <tr key={l}><td>{l}</td><td className="num text-right">{fmt(r.precision, 3)}</td><td className="num text-right">{fmt(r.recall, 3)}</td><td className="num text-right">{fmt(r["f1-score"], 3)}</td><td className="num text-right">{fmtInt(r.support)}</td></tr> : null; })}</tbody></table>
          </Card>
        )}
      </div>
      <Card title="Feature configuration"><div className="flex flex-wrap gap-1.5">{d.feature_config.features.map((f) => <span key={f} className="rounded bg-ink-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">{f}</span>)}</div></Card>
    </div>
  );
}
