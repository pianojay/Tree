import Plotly from "./draw";
import { get, latest } from "./api";
import { DIV, INK, SERIES, treeMarker } from "./colors";
import { CONFIG, ellipses, planeLayout, stripLayout, toData } from "./plot";
import { BaseState, View, planeRow } from "./view";

type State = BaseState & {
  axis: number[]; d_field: number[][]; z: number[][]; fake: number[][]; grad: number[][]; grid_lines: number[][][];
};

const SIZE = 400;
const ZWIN: [number, number] = [-3.5, 3.5];

const TEMPLATE = `
<div class="controls">
  <label>Dataset <select data-k="dataset"></select></label>
  <button class="primary" data-k="run">Train</button>
  <button data-k="reset">Reset</button>
  <label>D steps per G step <select data-k="dSteps"><option>1</option><option>2</option><option>3</option><option>5</option></select></label>
  <label><input type="checkbox" data-k="freezeD"> freeze D</label>
  <label><input type="checkbox" data-k="freezeG"> freeze G</label>
  <label><input type="checkbox" data-k="showField" checked> <i class="div-swatch"></i> discriminator field</label>
  <label><input type="checkbox" data-k="showGrad" checked> generator's signal</label>
  <label><input type="checkbox" data-k="showGrid"> latent grid, pushed by G</label>
  <span class="stat" data-k="stat"></span>
</div>
<div class="plane-row" data-k="row"></div>
<div class="info-row">
  <div class="card">
    <h2>The game</h2>
    <p class="sub">Discriminator and generator losses (non-saturating). Neither curve is a measure of quality.</p>
    <div data-k="loss"></div>
  </div>
  <div class="card">
    <h2>The referee's verdicts</h2>
    <p class="sub">Mean D(x) on real Trees and on generated ones. At the ideal equilibrium both sit at ½.</p>
    <div data-k="verdict"></div>
  </div>
  <div class="card">
    <h2>Reading the planes</h2>
    <ul class="legend-list">
      <li><span class="key"><i class="div-swatch"></i></span>D(x) = P(real): red = "fake", grey = undecided, blue = "real"</li>
      <li><span class="key"><i class="tree-dot"></i></span>generated Trees G(z), in their own pixel colors; the same color marks their z on the left. If the left plane turns one color, the generator has collapsed.</li>
      <li><span class="key" style="color:${INK.ink}">•</span>real training Trees</li>
      <li><span class="key" style="color:${INK.ink}">↗</span>the generator's only signal: the direction of ∇ₓ log D(x) at its own samples</li>
      <li><span class="key" style="color:${SERIES[1]}">●</span>hover the latent plane: z and G(z)</li>
    </ul>
    <div class="readout" data-k="readout"></div>
  </div>
</div>`;

export class GANView extends View<State> {
  kind = "gan";
  planes: Record<string, HTMLElement> = {};
  hz?: { z: number[]; x: number[]; p: number };

  constructor(root: HTMLElement) {
    super(root, TEMPLATE);
    this.wireCommon();
    this.planes = planeRow(this.el.row, [
      { key: "z", title: "Latent z ~ N(0, I)", sub: "colored by what G makes of each z" },
      { key: "x", title: "Data space x", sub: "real Trees, generated Trees, and the referee" },
    ], [`<span>generator G</span><span class="glyph">→</span><span class="back">no inverse,<br>no density</span>`]);
    const live = () => this.control("config", this.liveConfig());
    this.el.dSteps.onchange = live;
    this.el.freezeD.onchange = live;
    this.el.freezeG.onchange = live;
    for (const k of ["showField", "showGrad", "showGrid"]) this.el[k].onchange = () => this.renderPlanes();
    this.planes.z.addEventListener("mousemove", (ev) => { const z = toData(this.planes.z, ev); if (z) this.generateAt(z); });
    this.planes.x.addEventListener("mousemove", (ev) => { const x = toData(this.planes.x, ev); if (x) this.judgeAt(x); });
    this.planes.z.addEventListener("mouseleave", () => { this.hz = undefined; this.renderPlanes(); });
  }

  liveConfig() {
    return { d_steps: Number((this.el.dSteps as HTMLSelectElement).value),
      freeze_d: (this.el.freezeD as HTMLInputElement).checked, freeze_g: (this.el.freezeG as HTMLInputElement).checked };
  }
  resetConfig() { return this.liveConfig(); }
  onReset() { this.hz = undefined; }

  statusText(s: State) {
    const last = s.history[s.history.length - 1];
    return `step ${s.step}` + (last ? ` · D(real) ${last[3].toFixed(2)} · D(fake) ${last[4].toFixed(2)}` : "");
  }

  checked(k: string) { return (this.el[k] as HTMLInputElement).checked; }

  generateAt = latest(async (z: number[]) => {
    const r = await get<{ x: number[]; p_real: number }>(`/api/gan/generate?z1=${z[0].toFixed(3)}&z2=${z[1].toFixed(3)}`);
    this.hz = { z, x: r.x, p: r.p_real };
    this.el.readout.textContent = `G(${z.map((v) => v.toFixed(2)).join(", ")}) = (${r.x.map((v) => v.toFixed(2)).join(", ")}) · D = ${r.p_real.toFixed(2)}`;
    this.renderPlanes();
  });

  judgeAt = latest(async (x: number[]) => {
    const r = await get<{ p_real: number }>(`/api/gan/discriminate?x1=${x[0].toFixed(3)}&x2=${x[1].toFixed(3)}`);
    this.el.readout.textContent = `D(${x.map((v) => v.toFixed(2)).join(", ")}) = P(real) = ${r.p_real.toFixed(2)}`;
  });

  render() { this.renderPlanes(); this.renderLoss(); }

  renderPlanes() {
    const s = this.state; if (!s) return;
    const xs = (pts: number[][]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) });

    // latent plane: each z wears the colors of the Tree G makes from it
    const prior = ellipses([[0, 0], [0, 0]], [[1, 1], [2, 2]], 1, 64);
    const zt: any[] = [
      { type: "scatter", mode: "lines", ...prior, line: { color: INK.ink2, width: 1.5, dash: "dash" } },
      { type: "scatter", mode: "markers", ...xs(s.z), marker: treeMarker(s.fake, 6) },
    ];
    if (this.hz)
      zt.push({ type: "scatter", mode: "markers", x: [this.hz.z[0]], y: [this.hz.z[1]],
        marker: { size: 13, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    Plotly.react(this.planes.z, zt, planeLayout("z₁", "z₂", ZWIN, SIZE), CONFIG);

    // data plane
    const t: any[] = [];
    if (this.checked("showField"))
      t.push({ type: "heatmap", x: s.axis, y: s.axis, z: s.d_field, zmin: 0, zmax: 1, colorscale: DIV,
        showscale: false, zsmooth: "best", opacity: 0.85 });
    if (this.checked("showGrid")) {
      const x: (number | null)[] = [], y: (number | null)[] = [];
      s.grid_lines.forEach((line) => { line.forEach((p) => { x.push(p[0]); y.push(p[1]); }); x.push(null); y.push(null); });
      t.push({ type: "scatter", mode: "lines", x, y, line: { color: INK.ink2, width: 0.8 }, opacity: 0.5 });
    }
    t.push({ type: "scatter", mode: "markers", ...xs(s.data), marker: { size: 3, color: INK.ink, opacity: 0.45 } });
    t.push({ type: "scatter", mode: "markers", ...xs(s.fake), marker: treeMarker(s.fake, 6) });
    if (this.checked("showGrad")) {
      const x: (number | null)[] = [], y: (number | null)[] = [];
      s.grad.forEach((g, i) => {
        const n = Math.hypot(g[0], g[1]) || 1, p = s.fake[i];
        x.push(p[0], p[0] + (0.35 * g[0]) / n, null);
        y.push(p[1], p[1] + (0.35 * g[1]) / n, null);
      });
      t.push({ type: "scatter", mode: "lines", x, y, line: { color: INK.ink, width: 1.2 } });
    }
    if (this.hz)
      t.push({ type: "scatter", mode: "markers", x: [this.hz.x[0]], y: [this.hz.x[1]],
        marker: { size: 13, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    Plotly.react(this.planes.x, t, planeLayout("x₁ (pixel 1)", "x₂ (pixel 2)", s.window, SIZE), CONFIG);
  }

  renderLoss() {
    const s = this.state; if (!s) return;
    const x = s.history.map((h) => h[0]);
    const legend = (layout: any) => {
      layout.showlegend = true;
      layout.legend = { orientation: "h", y: 1.15, font: { size: 11, color: INK.ink2 } };
      layout.margin.t = 24;
      return layout;
    };
    Plotly.react(this.el.loss, [
      { type: "scatter", mode: "lines", name: "D loss", x, y: s.history.map((h) => h[1]), line: { color: SERIES[0], width: 2 } },
      { type: "scatter", mode: "lines", name: "G loss", x, y: s.history.map((h) => h[2]), line: { color: SERIES[1], width: 2 } },
    ], legend(stripLayout("step", "loss", undefined, 400, 220)), CONFIG);
    const vl = legend(stripLayout("step", "mean D(x)", undefined, 400, 220));
    vl.yaxis.range = [0, 1];
    Plotly.react(this.el.verdict, [
      { type: "scatter", mode: "lines", name: "D(real)", x, y: s.history.map((h) => h[3]), line: { color: SERIES[0], width: 2 } },
      { type: "scatter", mode: "lines", name: "D(fake)", x, y: s.history.map((h) => h[4]), line: { color: SERIES[1], width: 2 } },
    ], vl, CONFIG);
  }
}
