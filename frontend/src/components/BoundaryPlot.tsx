import { useMemo } from "react";
import { Plot } from "./Plot";
import type { BoundaryResponse, DatasetResponse, SampleResult } from "../types";

export const CLASS_COLORS = ["#3987e5", "#d95926"]; // validated categorical slots 1-2 (dark)
const SURFACE_SCALE: [number, string][] = [
  [0, "#1c4f8f"], [0.35, "#22364f"], [0.5, "#383835"], [0.65, "#57301f"], [1, "#9a3d17"],
];

interface Props {
  boundary: BoundaryResponse | null;
  dataset: DatasetResponse | null;
  selected?: number | null;
  epsilon?: number;
  norm?: "linf" | "l2";
  samples?: SampleResult[];
  onSelect?: (testIndex: number) => void;
  height?: number;
  showTrain?: boolean;
  title: string;
}

export function BoundaryPlot({ boundary, dataset, selected, epsilon, norm = "linf", samples, onSelect,
  height = 440, showTrain = true, title }: Props) {
  const { data, layout } = useMemo(() => {
    const traces: unknown[] = [];
    if (boundary) {
      traces.push({
        type: "heatmap", x: boundary.x, y: boundary.y, z: boundary.p1, zmin: 0, zmax: 1,
        colorscale: SURFACE_SCALE, showscale: true, zsmooth: "best", name: "P(class 1)",
        colorbar: { thickness: 10, len: 0.8, title: { text: "P(class 1)", side: "right" }, tickfont: { size: 10 } },
        hovertemplate: "x1=%{x:.2f}<br>x2=%{y:.2f}<br>P(class 1)=%{z:.3f}<extra>model</extra>",
      });
      traces.push({
        type: "contour", x: boundary.x, y: boundary.y, z: boundary.p1, showscale: false,
        contours: { coloring: "none", start: 0.5, end: 0.5, size: 1 },
        line: { color: "#e6edf3", width: 2 }, name: "decision boundary", hoverinfo: "skip", showlegend: true,
      });
      const def = boundary.target_system.defense;
      if (def && def.kind === "confidence_rejection") {
        traces.push({
          type: "contour", x: boundary.x, y: boundary.y, z: boundary.p1, showscale: false,
          contours: { coloring: "none", start: 1 - def.threshold, end: def.threshold, size: 2 * def.threshold - 1 },
          line: { color: "#fab219", width: 1.5, dash: "dash" }, name: `abstain band (τ=${def.threshold})`,
          hoverinfo: "skip", showlegend: true,
        });
      }
    }
    if (dataset) {
      if (showTrain) {
        for (const c of [0, 1]) {
          const pts = dataset.train.X.filter((_, i) => dataset.train.y[i] === c);
          traces.push({
            type: "scatter", mode: "markers", x: pts.map((p) => p[0]), y: pts.map((p) => p[1]),
            name: `train · class ${c}`, marker: { size: 5, color: CLASS_COLORS[c], opacity: 0.35 },
            hoverinfo: "skip",
          });
        }
      }
      for (const c of [0, 1]) {
        const idx = dataset.test.y.map((y, i) => (y === c ? i : -1)).filter((i) => i >= 0);
        traces.push({
          type: "scatter", mode: "markers", name: `test · class ${c}`,
          x: idx.map((i) => dataset.test.X[i][0]), y: idx.map((i) => dataset.test.X[i][1]),
          customdata: idx,
          marker: { size: 9, color: CLASS_COLORS[c], line: { color: "#0f141b", width: 2 } },
          hovertemplate: "test #%{customdata}<br>x=(%{x:.3f}, %{y:.3f})<extra>class " + c + "</extra>",
        });
      }
    }
    const shapes: unknown[] = [];
    if (samples) {
      const lx: (number | null)[] = [], ly: (number | null)[] = [];
      const ok = samples.filter((s) => s.x_adversarial && s.attack_success);
      // Abstention is neither an evasion nor a correct prediction: plotted separately.
      const abstained = samples.filter((s) => s.x_adversarial && !s.attack_success && s.pred_adversarial === -1);
      const fail = samples.filter((s) => s.x_adversarial && !s.attack_success && s.pred_adversarial !== -1);
      for (const s of samples) {
        if (!s.x_adversarial) continue;
        lx.push(s.x_original[0], s.x_adversarial[0], null);
        ly.push(s.x_original[1], s.x_adversarial[1], null);
      }
      traces.push({ type: "scatter", mode: "lines", x: lx, y: ly, name: "perturbation δ",
        line: { color: "#c5d1dd", width: 1.5 }, hoverinfo: "skip" });
      const advTrace = (list: SampleResult[], name: string, symbol: string, color: string) => ({
        type: "scatter", mode: "markers", name,
        x: list.map((s) => s.x_adversarial![0]), y: list.map((s) => s.x_adversarial![1]),
        customdata: list.map((s) => [s.test_index, s.pred_adversarial, s.confidence_adversarial, s.perturbation_linf]),
        marker: { symbol, size: 11, color, line: { color: "#0f141b", width: 1.5 } },
        hovertemplate: "adversarial for test #%{customdata[0]}<br>pred=%{customdata[1]} (−1 = abstain)" +
          "<br>conf=%{customdata[2]:.3f}<br>‖δ‖∞=%{customdata[3]:.4f}<extra>" + name + "</extra>",
      });
      traces.push(advTrace(ok, "adversarial · evaded", "x", "#d03b3b"));
      traces.push(advTrace(fail, "adversarial · still correct", "circle-open", "#c5d1dd"));
      if (abstained.length) traces.push(advTrace(abstained, "adversarial · abstained", "diamond-open", "#e0a23a"));
    }
    if (dataset && selected != null && dataset.test.X[selected]) {
      const [sx, sy] = dataset.test.X[selected];
      traces.push({ type: "scatter", mode: "markers", x: [sx], y: [sy], name: `selected #${selected}`,
        marker: { size: 18, color: "rgba(0,0,0,0)", line: { color: "#3ad1e0", width: 2.5 } },
        hoverinfo: "skip" });
      if (epsilon) {
        shapes.push({
          type: norm === "linf" ? "rect" : "circle", xref: "x", yref: "y",
          x0: sx - epsilon, x1: sx + epsilon, y0: sy - epsilon, y1: sy + epsilon,
          line: { color: "#3ad1e0", width: 1.5, dash: "dot" }, fillcolor: "rgba(58,209,224,0.06)",
        });
      }
    }
    const dom = dataset?.dataset.domain;
    return {
      data: traces,
      layout: {
        shapes,
        xaxis: { title: { text: "x1" }, range: dom ? dom[0] : undefined, constrain: "domain" },
        yaxis: { title: { text: "x2" }, range: dom ? dom[1] : undefined, scaleanchor: "x", scaleratio: 1 },
        uirevision: title,
      },
    };
  }, [boundary, dataset, selected, epsilon, norm, samples, showTrain, title]);

  return (
    <Plot data={data} layout={layout} height={height} ariaLabel={title}
      onPointClick={(p) => {
        if (onSelect && typeof p.customdata === "number") onSelect(p.customdata);
      }} />
  );
}
