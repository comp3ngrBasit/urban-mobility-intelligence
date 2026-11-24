export type Mode = "live" | "latest_available" | "historical" | "unavailable";
export interface Freshness { source: string; data_as_of: string | null; mode: Mode; note: string | null; last_ingestion: string | null; last_ingestion_status: string | null }
export interface CI { mean: number | null; lo: number | null; hi: number | null; n: number }
export interface Heatmap { z: number[][]; hours: number[]; dows: number[]; unit?: string }
export interface Corr { n: number; pearson_r: number | null; pearson_p?: number; pearson_ci95?: [number, number]; spearman_rho: number | null; spearman_p?: number }
export interface TwoGroup { welch_t?: number; welch_p?: number; mannwhitney_p?: number; cohens_d?: number | null; note?: string; [k: string]: unknown }
export interface Anova { anova_F?: number; anova_p?: number; eta_squared?: number; kruskal_p?: number; groups?: Record<string, CI>; note?: string }

export interface Overview {
  mobility: { total_trips: number; period: [string, string]; daily: { ts: string[]; pickups: number[] }; borough_peak_hours: { borough: string; peak_hour: number; pickups_at_peak: number }[]; scope_note: string } | null;
  nyc_live: { freshness: Freshness; links: number; median_speed_mph: number; slow_links: number } | null;
  chicago_traffic: { mean_speed: CI; period: [string, string]; congestion_distribution: Record<string, number>; segments: number; scope_note: string } | null;
  accidents: { total_crashes: number; crashes_per_day: CI; severity_counts: Record<string, number>; fatalities: number; injuries_total: number; period: [string, string]; monthly: { month: string; crashes: number; serious: number; fatal: number }[] } | null;
  predicted_congestion: { target_time: string; distribution: Record<string, number> } | null;
}

export interface Segment { segment_id: number; street?: string; direction?: string; from_street?: string; to_street?: string; start_latitude: number; start_longitude: number; end_latitude: number; end_longitude: number }

export interface ModelSummary {
  task: string; kind: "reg" | "clf" | "forecast_classical"; best_model?: string; version: string; created_at?: string;
  test_period?: [string, string]; best_test?: Record<string, number | null>;
  baselines?: Record<string, Record<string, number | null>>;
  candidates?: Record<string, Record<string, number | null>>;
  aggregate?: Record<string, Record<string, number | null>>;
}
