import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import LiveTraffic from "./pages/LiveTraffic";
import Prediction from "./pages/Prediction";
import Mobility from "./pages/Mobility";
import Accidents from "./pages/Accidents";
import Weather from "./pages/Weather";
import GeoExplorer from "./pages/GeoExplorer";
import ModelPerformance from "./pages/ModelPerformance";
import Pipeline from "./pages/Pipeline";

export const NAV = [
  { to: "/", label: "Executive Overview", el: <Overview /> },
  { to: "/live", label: "Live Traffic", el: <LiveTraffic /> },
  { to: "/prediction", label: "Traffic Prediction", el: <Prediction /> },
  { to: "/mobility", label: "Mobility Analytics", el: <Mobility /> },
  { to: "/accidents", label: "Accident Intelligence", el: <Accidents /> },
  { to: "/weather", label: "Weather Impact", el: <Weather /> },
  { to: "/geo", label: "Geospatial Explorer", el: <GeoExplorer /> },
  { to: "/models", label: "Model Performance", el: <ModelPerformance /> },
  { to: "/pipeline", label: "Data & Pipeline", el: <Pipeline /> },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-5 py-5">
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#3B82F6" />
            <stop offset="1" stopColor="#8B5CF6" />
          </linearGradient>
        </defs>
        <rect width="26" height="26" rx="7" fill="url(#lg)" opacity=".18" />
        <path d="M5 18 C 9 18, 9 8, 13 8 S 17 18, 21 18" stroke="#22D3EE" strokeWidth="2" fill="none" strokeLinecap="round" />
        <circle cx="13" cy="8" r="2" fill="#3B82F6" />
      </svg>
      <div className="leading-tight">
        <div className="text-[13px] font-semibold text-white">Urban Mobility</div>
        <div className="text-[11px] text-slate-500">Traffic Intelligence</div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="flex min-h-screen bg-ink-900 bg-[radial-gradient(ellipse_at_top_right,rgba(139,92,246,0.07),transparent_45%)]">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-ink-800 bg-ink-950/60 md:flex">
        <Logo />
        <nav className="flex flex-1 flex-col gap-0.5 px-3">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-[13px] transition-colors ${
                  isActive ? "bg-electric-500/10 font-medium text-white shadow-[inset_2px_0_0_#3B82F6]" : "text-slate-400 hover:bg-ink-800 hover:text-slate-200"}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-ink-800 px-5 py-4 text-[11px] leading-relaxed text-slate-500">
          NYC TLC · NYC DOT · Chicago Data Portal · OpenStreetMap · Open-Meteo
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-5 py-6 md:px-8">
        <select className="mb-4 w-full md:hidden" onChange={(e) => (window.location.href = e.target.value)}
          defaultValue={window.location.pathname}>
          {NAV.map((n) => <option key={n.to} value={n.to}>{n.label}</option>)}
        </select>
        <Routes>
          {NAV.map((n) => <Route key={n.to} path={n.to} element={n.el} />)}
        </Routes>
      </main>
    </div>
  );
}
