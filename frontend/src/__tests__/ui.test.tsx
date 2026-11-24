import { render, screen } from "@testing-library/react";
import { Async, EmptyState, FreshnessBadge } from "../components/ui";
import { ApiError, qs } from "../lib/api";
import { fmt, pct, ratioColor } from "../lib/format";

describe("FreshnessBadge", () => {
  it("labels live, latest-available and historical data distinctly", () => {
    const { rerender } = render(<FreshnessBadge mode="live" />);
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    rerender(<FreshnessBadge mode="latest_available" asOf="2026-11-24T10:00:00" />);
    expect(screen.getByText("LATEST AVAILABLE")).toBeInTheDocument();
    rerender(<FreshnessBadge mode="historical" />);
    expect(screen.getByText("HISTORICAL")).toBeInTheDocument();
  });
});

describe("Async", () => {
  const base = { loading: false, reload: () => {} };
  it("renders an empty state with the pipeline hint when the API reports not_ready", () => {
    const err = new ApiError(404, { error: "not_ready", what: "mobility", hint: "python -m src.pipeline analyze" });
    render(<Async state={{ ...base, data: undefined, error: err }}>{() => <div>never</div>}</Async>);
    expect(screen.getByText(/No data yet for mobility/)).toBeInTheDocument();
    expect(screen.getByText("python -m src.pipeline analyze")).toBeInTheDocument();
    expect(screen.queryByText("never")).toBeNull();
  });
  it("shows an error with retry for other failures", () => {
    render(<Async state={{ ...base, data: undefined, error: new ApiError(500, "boom") }}>{() => null}</Async>);
    expect(screen.getByText(/Request failed/)).toBeInTheDocument();
  });
  it("renders children with data", () => {
    render(<Async state={{ ...base, data: { n: 3 }, error: undefined }}>{(d) => <span>n={d.n}</span>}</Async>);
    expect(screen.getByText("n=3")).toBeInTheDocument();
  });
});

describe("helpers", () => {
  it("formats numbers and handles missing values", () => {
    expect(fmt(1.2345, 2)).toBe("1.23");
    expect(fmt(null)).toBe("—");
    expect(pct(0.256)).toBe("25.6%");
  });
  it("builds query strings and skips empty values", () => {
    expect(qs({ a: 1, b: "", c: null, d: ["x", "y"] })).toBe("?a=1&d=x&d=y");
  });
  it("maps speed ratios to the congestion ramp", () => {
    expect(ratioColor(0.3)).toBe("#F2545B");
    expect(ratioColor(0.9)).toBe("#22D3EE");
  });
  it("EmptyState shows command", () => {
    render(<EmptyState what="x" hint="cmd" />);
    expect(screen.getByText("cmd")).toBeInTheDocument();
  });
});
