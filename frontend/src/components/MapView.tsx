import { useEffect, useRef } from "react";
import maplibregl, { type LayerSpecification, type Map as MlMap, type GeoJSONSourceSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

// Free CARTO dark vector basemap (no API key).
const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const FALLBACK_STYLE = { version: 8 as const, sources: {}, layers: [{ id: "bg", type: "background" as const, paint: { "background-color": "#0B1220" } }] };

export interface MapLayer {
  id: string;
  data: GeoJSON.FeatureCollection;
  /** MapLibre layer specs without `source` (loosely typed; validated by MapLibre at runtime). */
  layers: ({ id: string; type: string } & Record<string, unknown>)[];
  popup?: (p: Record<string, unknown>) => string;
  onClick?: (p: Record<string, unknown>) => void;
}

export const CENTERS = {
  nyc: { center: [-73.96, 40.74] as [number, number], zoom: 10.6 },
  chicago: { center: [-87.68, 41.86] as [number, number], zoom: 10.2 },
};

export default function MapView({ layers, center, zoom, height = 520 }: {
  layers: readonly MapLayer[]; center: [number, number]; zoom: number; height?: number;
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap>();
  const ready = useRef(false);
  const latest = useRef(layers);
  latest.current = layers;

  useEffect(() => {
    if (!el.current || typeof window === "undefined" || !("WebGLRenderingContext" in window)) return;
    const m = new maplibregl.Map({ container: el.current, style: STYLE, center, zoom, attributionControl: { compact: true } });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    const onReady = () => {
      ready.current = true;
      sync(m, latest.current);
    };
    m.on("load", onReady);
    // Basemap unreachable (offline / blocked): fall back to a plain dark canvas so the
    // analytical layers still render.
    let fellBack = false;
    m.on("error", () => {
      if (ready.current || fellBack) return;
      fellBack = true;
      m.setStyle(FALLBACK_STYLE);
      m.once("style.load", onReady);
    });
    map.current = m;
    return () => m.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    map.current?.flyTo({ center, zoom, duration: 800 });
  }, [center[0], center[1], zoom]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (map.current && ready.current) sync(map.current, layers);
  }, [layers]);

  return <div ref={el} className="w-full overflow-hidden rounded-lg border border-ink-700" style={{ height }} />;
}

const handlers = new WeakMap<MlMap, Map<string, (e: maplibregl.MapLayerMouseEvent) => void>>();

function sync(m: MlMap, layers: readonly MapLayer[]) {
  const wanted = new Set(layers.map((l) => l.id));
  const style = m.getStyle();
  // remove stale user layers/sources
  for (const lyr of style.layers ?? []) {
    const src = (lyr as { source?: string }).source;
    if (src?.startsWith("u-") && !wanted.has(src.slice(2))) m.removeLayer(lyr.id);
  }
  for (const id of Object.keys(style.sources)) if (id.startsWith("u-") && !wanted.has(id.slice(2))) m.removeSource(id);

  let hmap = handlers.get(m);
  if (!hmap) handlers.set(m, (hmap = new Map()));

  for (const l of layers) {
    const sid = `u-${l.id}`;
    const src = m.getSource(sid) as maplibregl.GeoJSONSource | undefined;
    if (src) src.setData(l.data);
    else m.addSource(sid, { type: "geojson", data: l.data } as GeoJSONSourceSpecification);
    for (const spec of l.layers) {
      if (m.getLayer(spec.id)) m.removeLayer(spec.id);
      m.addLayer({ ...spec, source: sid } as unknown as LayerSpecification);
      const prev = hmap.get(spec.id);
      if (prev) m.off("click", spec.id, prev);
      if (l.popup || l.onClick) {
        const h = (e: maplibregl.MapLayerMouseEvent) => {
          const p = (e.features?.[0]?.properties ?? {}) as Record<string, unknown>;
          if (l.popup) new maplibregl.Popup({ closeButton: false, maxWidth: "300px" }).setLngLat(e.lngLat).setHTML(l.popup(p)).addTo(m);
          l.onClick?.(p);
        };
        m.on("click", spec.id, h);
        m.on("mouseenter", spec.id, () => (m.getCanvas().style.cursor = "pointer"));
        m.on("mouseleave", spec.id, () => (m.getCanvas().style.cursor = ""));
        hmap.set(spec.id, h);
      }
    }
  }
}

export const fc = (features: GeoJSON.Feature[]): GeoJSON.FeatureCollection => ({ type: "FeatureCollection", features });
export const point = (lon: number, lat: number, properties: Record<string, unknown>): GeoJSON.Feature => ({
  type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties,
});
export const line = (coords: [number, number][], properties: Record<string, unknown>): GeoJSON.Feature => ({
  type: "Feature", geometry: { type: "LineString", coordinates: coords }, properties,
});
export const kv = (rows: [string, unknown][]) =>
  rows.map(([k, v]) => `<div style="display:flex;justify-content:space-between;gap:12px"><span style="color:#94A3B8">${k}</span><b>${v ?? "—"}</b></div>`).join("");
