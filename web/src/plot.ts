import { INK } from "./colors";

export const CONFIG = { displayModeBar: false, responsive: false, staticPlot: false };

const axis = (title: string, range?: [number, number]) => ({
  title: { text: title, font: { size: 12, color: INK.ink2 } },
  range, gridcolor: INK.grid, zerolinecolor: INK.axis, linecolor: INK.axis,
  tickfont: { size: 11, color: INK.muted }, fixedrange: true,
});

export function planeLayout(xTitle: string, yTitle: string, range: [number, number], size = 520): any {
  return {
    width: size, height: size, margin: { l: 46, r: 12, t: 8, b: 40 },
    paper_bgcolor: INK.surface, plot_bgcolor: INK.surface,
    xaxis: axis(xTitle, range), yaxis: { ...axis(yTitle, range), scaleanchor: "x" },
    showlegend: false, dragmode: false, hovermode: false,
  };
}

export function stripLayout(xTitle: string, yTitle: string, range: [number, number] | undefined,
                            width: number, height: number): any {
  return {
    width, height, margin: { l: 46, r: 12, t: 8, b: 36 },
    paper_bgcolor: INK.surface, plot_bgcolor: INK.surface,
    xaxis: axis(xTitle, range), yaxis: axis(yTitle),
    showlegend: false, dragmode: false, hovermode: false,
  };
}

/** Mouse event -> data coordinates of a Plotly plot (null when outside). */
export function toData(gd: HTMLElement, ev: MouseEvent): [number, number] | null {
  const fl = (gd as any)._fullLayout;
  if (!fl) return null;
  const xa = fl.xaxis, ya = fl.yaxis;
  const rect = gd.getBoundingClientRect();
  const px = ev.clientX - rect.left - xa._offset;
  const py = ev.clientY - rect.top - ya._offset;
  if (px < 0 || py < 0 || px > xa._length || py > ya._length) return null;
  return [xa.p2d(px), ya.p2d(py)];
}

/** Axis-aligned ellipses as one polyline trace (nulls separate shapes). */
export function ellipses(centers: number[][], radii: number[][], scale = 1, n = 28) {
  const x: (number | null)[] = [], y: (number | null)[] = [];
  centers.forEach(([cx, cy], i) => {
    const [sx, sy] = radii[i];
    for (let k = 0; k <= n; k++) {
      const t = (2 * Math.PI * k) / n;
      x.push(cx + scale * sx * Math.cos(t));
      y.push(cy + scale * sy * Math.sin(t));
    }
    x.push(null); y.push(null);
  });
  return { x, y };
}
