import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

interface Props {
  data: unknown[];
  layout: Record<string, unknown>;
  height?: number;
  onPointClick?: (pt: { curveNumber: number; pointIndex: number; customdata?: unknown }) => void;
  ariaLabel: string;
}

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "#0f141b",
  font: { family: "Inter, system-ui, sans-serif", color: "#c5d1dd", size: 12 },
  margin: { l: 44, r: 12, t: 10, b: 10 },
  xaxis: { gridcolor: "#1c2530", zerolinecolor: "#26313d", linecolor: "#26313d" },
  yaxis: { gridcolor: "#1c2530", zerolinecolor: "#26313d", linecolor: "#26313d" },
  hoverlabel: { bgcolor: "#161d26", bordercolor: "#2c3a48", font: { color: "#e6edf3" } },
  legend: { orientation: "h", y: -0.2, yanchor: "top", x: 0, font: { size: 11 } },
};

export function Plot({ data, layout, height = 420, onPointClick, ariaLabel }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onPointClick);
  clickRef.current = onPointClick;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const merged = {
      ...BASE_LAYOUT, ...layout, height,
      xaxis: { ...BASE_LAYOUT.xaxis, ...(layout.xaxis as object) },
      yaxis: { ...BASE_LAYOUT.yaxis, ...(layout.yaxis as object) },
    };
    Plotly.react(el, data, merged, { displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"] }).then(() => {
      const gd = el as unknown as { removeAllListeners?: (e: string) => void; on?: (e: string, f: (ev: { points: { curveNumber: number; pointIndex: number; customdata?: unknown }[] }) => void) => void };
      gd.removeAllListeners?.("plotly_click");
      gd.on?.("plotly_click", (ev) => { const p = ev.points?.[0]; if (p && clickRef.current) clickRef.current(p); });
    });
  }, [data, layout, height]);

  useEffect(() => () => { if (ref.current) Plotly.purge(ref.current); }, []);

  return <div ref={ref} role="img" aria-label={ariaLabel} style={{ width: "100%", minHeight: height }} />;
}
