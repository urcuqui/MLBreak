import { useMemo } from "react";
import { Plot } from "./Plot";
import { CLASS_COLORS, SURFACE_SCALE } from "./BoundaryPlot";
import type { BoundaryResponse, TestRow, TrainRow } from "../types";

// Status colours are reserved for status and always paired with a distinct symbol + legend label.
const STATUS = { good: "#0ca30c", warning: "#fab219", critical: "#d03b3b" };

export type LabelMode = "true" | "observed" | "sanitized";
export type Highlight = "none" | "poisoned" | "detection";

interface Props {
  boundary: BoundaryResponse | null;
  compare?: BoundaryResponse | null; // drawn as a dashed line (the CLEAN boundary)
  rows: TrainRow[];
  labelMode: LabelMode;
  highlight: Highlight;
  testRows?: TestRow[];
  brokenKey?: "pred_poisoned" | "pred_defended";
  domain?: number[][];
  height?: number;
  title: string;
}

const labelOf = (r: TrainRow, mode: LabelMode) =>
  mode === "true" ? r.y_true : mode === "observed" ? r.y_observed : r.y_sanitized;

export function PoisonPlot({ boundary, compare, rows, labelMode, highlight, testRows, brokenKey, domain,
  height = 400, title }: Props) {
  const { data, layout } = useMemo(() => {
    const traces: unknown[] = [];
    if (boundary) {
      traces.push({
        type: "heatmap", x: boundary.x, y: boundary.y, z: boundary.p1, zmin: 0, zmax: 1, colorscale: SURFACE_SCALE,
        zsmooth: "best", showscale: true, name: "P(class 1)",
        colorbar: { thickness: 10, len: 0.8, title: { text: "P(class 1)", side: "right" }, tickfont: { size: 10 } },
        hovertemplate: "x1=%{x:.2f}<br>x2=%{y:.2f}<br>P(class 1)=%{z:.3f}<extra>model</extra>",
      });
      traces.push({
        type: "contour", x: boundary.x, y: boundary.y, z: boundary.p1, showscale: false, hoverinfo: "skip",
        contours: { coloring: "none", start: 0.5, end: 0.5, size: 1 }, line: { color: "#e6edf3", width: 2 },
        name: "decision boundary", showlegend: true,
      });
    }
    if (compare) {
      traces.push({
        type: "contour", x: compare.x, y: compare.y, z: compare.p1, showscale: false, hoverinfo: "skip",
        contours: { coloring: "none", start: 0.5, end: 0.5, size: 1 },
        line: { color: "#3ad1e0", width: 2, dash: "dash" }, name: "clean boundary (reference)", showlegend: true,
      });
    }
    const shown = rows.filter((r) => labelOf(r, labelMode) != null);
    for (const c of [0, 1]) {
      const pts = shown.filter((r) => labelOf(r, labelMode) === c);
      traces.push({
        type: "scatter", mode: "markers", name: `train · labelled ${c}`,
        x: pts.map((r) => r.x[0]), y: pts.map((r) => r.x[1]),
        customdata: pts.map((r) => [r.train_index, r.y_true, r.y_observed]),
        marker: { size: 6, color: CLASS_COLORS[c], opacity: 0.8, line: { color: "#0f141b", width: 1 } },
        hovertemplate: "train #%{customdata[0]}<br>true label %{customdata[1]} · given %{customdata[2]}<extra></extra>",
      });
    }
    const mark = (list: TrainRow[], name: string, symbol: string, color: string) => traces.push({
      type: "scatter", mode: "markers", name, x: list.map((r) => r.x[0]), y: list.map((r) => r.x[1]),
      customdata: list.map((r) => [r.train_index, r.y_true, r.y_observed, r.knn_agreement ?? -1]),
      // "-open" symbols are stroked with marker.color, so it must be the visible colour.
      marker: { symbol, size: 11, color, line: { color, width: 2 } },
      hovertemplate: "train #%{customdata[0]}<br>true %{customdata[1]} → given %{customdata[2]}" +
        "<br>kNN agreement %{customdata[3]:.2f}<extra>" + name + "</extra>",
    });
    if (highlight === "poisoned") {
      mark(rows.filter((r) => r.poisoned), "poisoned (label flipped)", "diamond-open", "#e6edf3");
    } else if (highlight === "detection") {
      mark(rows.filter((r) => r.detection === "true_positive"), "poisoned · flagged (caught)", "diamond-open", STATUS.good);
      mark(rows.filter((r) => r.detection === "false_negative"), "poisoned · missed", "x-open", STATUS.critical);
      mark(rows.filter((r) => r.detection === "false_positive"), "clean · flagged (false positive)", "triangle-up-open",
        STATUS.warning);
    }
    if (testRows && brokenKey) {
      const broken = testRows.filter((t) => t.pred_clean === t.y_true && t[brokenKey] !== t.y_true);
      traces.push({
        type: "scatter", mode: "markers", name: "test · broken by poisoning",
        x: broken.map((t) => t.x[0]), y: broken.map((t) => t.x[1]), customdata: broken.map((t) => [t.test_index, t.y_true]),
        marker: { symbol: "x", size: 9, color: STATUS.critical, line: { width: 0 } },
        hovertemplate: "test #%{customdata[0]} (true %{customdata[1]})<br>clean model right, this model wrong" +
          "<extra></extra>",
      });
    }
    return {
      data: traces,
      layout: {
        xaxis: { title: { text: "x1" }, range: domain?.[0], constrain: "domain" },
        yaxis: { title: { text: "x2" }, range: domain?.[1], scaleanchor: "x", scaleratio: 1 },
        uirevision: title,
      },
    };
  }, [boundary, compare, rows, labelMode, highlight, testRows, brokenKey, domain, title]);
  return <Plot data={data} layout={layout} height={height} ariaLabel={title} />;
}
