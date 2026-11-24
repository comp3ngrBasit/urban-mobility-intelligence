import { useCallback, useEffect, useRef, useState } from "react";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
  get notReady(): { what: string; hint: string } | null {
    const d = this.detail as { error?: string; what?: string; hint?: string } | undefined;
    return this.status === 404 && d?.error === "not_ready" ? { what: d.what ?? "", hint: d.hint ?? "" } : null;
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      detail = (await res.json()).detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export interface ApiState<T> {
  data: T | undefined;
  error: ApiError | Error | undefined;
  loading: boolean;
  reload: () => void;
}

/** Fetch `path` whenever it changes. Pass null to skip. Optional polling for live views. */
export function useApi<T>(path: string | null, pollMs?: number): ApiState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError | Error>();
  const [loading, setLoading] = useState(Boolean(path));
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!path) return;
    const id = ++seq.current;
    setLoading(true);
    api<T>(path)
      .then((d) => {
        if (id === seq.current) {
          setData(d);
          setError(undefined);
        }
      })
      .catch((e) => id === seq.current && setError(e))
      .finally(() => id === seq.current && setLoading(false));
  }, [path, tick]);

  useEffect(() => {
    if (!pollMs || !path) return;
    const t = setInterval(reload, pollMs);
    return () => clearInterval(t);
  }, [pollMs, path, reload]);

  return { data, error, loading, reload };
}

export const qs = (p: Record<string, string | number | boolean | null | undefined | string[]>) => {
  const u = new URLSearchParams();
  Object.entries(p).forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") return;
    if (Array.isArray(v)) v.forEach((x) => u.append(k, x));
    else u.append(k, String(v));
  });
  const s = u.toString();
  return s ? `?${s}` : "";
};
