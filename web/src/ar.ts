import Plotly from "plotly.js-dist-min";
import { get, latest } from "./api";
import { INK, SERIES, SEQ } from "./colors";
import { CONFIG, planeLayout, stripLayout, toData } from "./plot";
import { BaseState, View } from "./view";

type State = BaseState & { axis: number[]; density: number[][]; fine: number[]; marginal1: number[]; samples: number[][] };
type Cond = { x1: number; fine: number[]; p: number[]; p1: number };

const K_CHOICES = [1, 2, 3, 4, 6, 8, 12];

const TEMPLATE = `
<div class="controls">
  <label>Dataset <select data-k="dataset"></select></label>
  <label>Mixture components K <select data-k="k">${K_CHOICES.map((k) => `<option>${k}</option>`).join("")}</select></label>
  <button class="primary" data-k="run">Train</button>
  <button data-k="reset">Reset</button>
  <label><input type="checkbox" data-k="showDensity" checked> <i class="dens-swatch"></i> model density</label>
  <label><input type="checkbox" data-k="showData" checked> training data</label>
  <label><input type="checkbox" data-k="showSamples"> model samples</label>
  <button data-k="sampleOne">Sample one Tree, pixel by pixel</button>
  <span class="stat" data-k="stat"></span>
</div>
<div class="grid">
  <div class="col">
    <div class="card">
      <h2>Data plane: p(x₁, x₂) = p(x₁) · p(x₂ | x₁)</h2>
      <p class="sub"><i class="dens-swatch"></i> blue: the model's density p<sub>θ</sub>(x), √ scale (darker = more probable). Drag across the plane to move x₁; the orange curve is p(x₂ | x₁) riding on that line.</p>
      <div data-k="plane"></div>
    </div>
    <div class="card">
      <h2>First pixel: p(x₁)</h2>
      <div data-k="marg"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>The crux map: x₁ ↦ p(x₂ | x₁)</h2>
      <p class="sub" data-k="condTitle"></p>
      <div data-k="cond"></div>
    </div>
    <div class="card">
      <h2>Likelihood at the cursor (chain rule)</h2>
      <p class="sub">−log p(x) = −log p(x₁) − log p(x₂ | x₁). Hover over the data plane.</p>
      <div class="ll-bar" data-k="llbar"></div>
      <div class="ll-legend" data-k="lllegend"></div>
    </div>
    <div class="card">
      <h2>Training loss (negative log-likelihood)</h2>
      <div data-k="loss"></div>
    </div>
  </div>
</div>`;

export class ARView extends View<State> {
  kind = "ar";
  cond?: Cond;
  a = -1.6;                       // current x1 line
  dragging = false;
  drops: number[][] = [];         // Trees sampled step by step
  anim: { x1: number; x2: number; t0: number } | null = null;

  constructor(root: HTMLElement) {
    super(root, TEMPLATE);
    this.wireCommon();
    this.el.k.onchange = () => this.reset();
    for (const k of ["showDensity", "showData", "showSamples"]) this.el[k].onchange = () => this.renderPlane();
    this.el.sampleOne.onclick = () => this.sampleOne();

    // Plotly covers the page while a button is held, so drag moves are read on window.
    const plane = this.el.plane;
    plane.addEventListener("mousedown", (ev) => { this.dragging = true; this.moveX1(ev); });
    window.addEventListener("mouseup", () => (this.dragging = false));
    window.addEventListener("mousemove", (ev) => {
      if (this.dragging) this.moveX1(ev);
      else if (plane.contains(ev.target as Node)) this.hoverPoint(ev);
    });
  }

  resetConfig() { return { k: Number((this.el.k as HTMLSelectElement).value) }; }
  onReset() { this.drops = []; }

  statusText(s: State) {
    (this.el.k as HTMLSelectElement).value = String(s.cfg.k);
    const last = s.history[s.history.length - 1];
    return `step ${s.step}` + (last ? ` · NLL ${last[1].toFixed(3)}` : "");
  }

  async render() {
    await this.fetchCond(this.a);
    this.renderPlane(); this.renderMarg(); this.renderCond(); this.renderLoss();
  }

  // --- interaction -----------------------------------------------------------

  moveX1(ev: MouseEvent) {
    const p = toData(this.el.plane, ev);
    if (!p) return;
    this.a = p[0];
    this.renderPlane(); this.renderMarg();
    this.fetchCond(this.a);
  }

  fetchCond = latest(async (x1: number) => {
    this.cond = await get<Cond>(`/api/ar/conditional?x1=${x1.toFixed(3)}`);
    this.renderPlane(); this.renderMarg(); this.renderCond();
  });

  hoverPoint(ev: MouseEvent) {
    const p = toData(this.el.plane, ev);
    if (p) this.queryPoint(p);
  }

  queryPoint = latest(async (p: number[]) => {
    const r = await get<{ lp1: number; lp2: number }>(`/api/ar/point?x1=${p[0].toFixed(3)}&x2=${p[1].toFixed(3)}`);
    this.renderLL(p, r.lp1, r.lp2);
  });

  async sampleOne() {
    const s = await get<{ x1: number; x2: number }>("/api/ar/sample_one");
    this.a = s.x1;
    await this.fetchCond(s.x1);
    this.anim = { ...s, t0: performance.now() };
    const tick = () => {
      if (!this.anim) return;
      this.renderPlane(); this.renderMarg();
      if (performance.now() - this.anim.t0 < 1400) requestAnimationFrame(tick);
      else { this.drops.push([this.anim.x1, this.anim.x2]); this.anim = null; this.renderPlane(); this.renderMarg(); }
    };
    requestAnimationFrame(tick);
  }

  /** Animation phase: 0..1 = first marble falls into p(x1); 1..2 = second falls along the line. */
  phase() {
    return this.anim ? Math.min(2, (performance.now() - this.anim.t0) / 600) : 2;
  }

  // --- rendering -------------------------------------------------------------

  checked(k: string) { return (this.el[k] as HTMLInputElement).checked; }

  renderPlane() {
    const s = this.state; if (!s) return;
    const [lo, hi] = s.window;
    const traces: any[] = [];
    if (this.checked("showDensity"))
      traces.push({ type: "heatmap", x: s.axis, y: s.axis, z: s.density.map((row) => row.map(Math.sqrt)),
        colorscale: SEQ, showscale: false, zsmooth: "best" });
    if (this.checked("showData"))
      traces.push({ type: "scattergl", mode: "markers", x: s.data.map((p) => p[0]), y: s.data.map((p) => p[1]),
        marker: { size: 3, color: INK.ink, opacity: 0.35 } });
    if (this.checked("showSamples"))
      traces.push({ type: "scattergl", mode: "markers", x: s.samples.map((p) => p[0]), y: s.samples.map((p) => p[1]),
        marker: { size: 4, color: SERIES[1], opacity: 0.7 } });

    // the conditional "on a string": p(x2 | x1=a) drawn sideways from the x1 line
    if (this.cond && this.phase() >= 0.6) {
      const max = Math.max(...this.cond.p, 1e-9), w = 0.9 / max;
      const xs = [this.a, ...this.cond.p.map((p) => this.a + w * p), this.a];
      const ys = [this.cond.fine[0], ...this.cond.fine, this.cond.fine[this.cond.fine.length - 1]];
      traces.push({ type: "scatter", mode: "lines", x: xs, y: ys, fill: "toself",
        fillcolor: "rgba(235,104,52,0.18)", line: { color: SERIES[1], width: 2 } });
    }
    if (this.drops.length)
      traces.push({ type: "scatter", mode: "markers", x: this.drops.map((d) => d[0]), y: this.drops.map((d) => d[1]),
        marker: { size: 10, color: "rgba(0,0,0,0)", line: { color: INK.ink, width: 2 } } });
    if (this.anim && this.phase() > 1) {
      const f = this.phase() - 1, y = hi + (this.anim.x2 - hi) * f * f;
      traces.push({ type: "scatter", mode: "markers", x: [this.anim.x1], y: [y],
        marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    }
    const layout = planeLayout("x₁ (pixel 1)", "x₂ (pixel 2)", [lo, hi]);
    layout.shapes = [{ type: "line", x0: this.a, x1: this.a, y0: lo, y1: hi,
      line: { color: SERIES[1], width: 1.5, dash: "dot" } }];
    Plotly.react(this.el.plane, traces, layout, CONFIG);
  }

  renderMarg() {
    const s = this.state; if (!s) return;
    const traces: any[] = [{ type: "scatter", mode: "lines", x: s.fine, y: s.marginal1, fill: "tozeroy",
      fillcolor: "rgba(42,120,214,0.15)", line: { color: SERIES[0], width: 2 } }];
    const top = Math.max(...s.marginal1) * 1.15;
    if (this.anim && this.cond) {
      const f = Math.min(1, this.phase()), y = top + (this.cond.p1 - top) * f * f;
      traces.push({ type: "scatter", mode: "markers", x: [this.anim.x1], y: [y],
        marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    }
    const layout = stripLayout("x₁", "density", s.window, 520, 150);
    layout.yaxis.range = [0, top];
    layout.shapes = [{ type: "line", x0: this.a, x1: this.a, y0: 0, y1: top,
      line: { color: SERIES[1], width: 1.5, dash: "dot" } }];
    Plotly.react(this.el.marg, traces, layout, CONFIG);
  }

  renderCond() {
    const c = this.cond, s = this.state; if (!c || !s) return;
    this.el.condTitle.textContent = `p(x₂ | x₁ = ${this.a.toFixed(2)}), with x₂ on the vertical axis to match the plane`;
    const layout = stripLayout("density", "x₂", undefined, 380, 300);
    layout.yaxis.range = s.window;
    Plotly.react(this.el.cond, [{ type: "scatter", mode: "lines", x: c.p, y: c.fine, fill: "tozerox",
      fillcolor: "rgba(235,104,52,0.18)", line: { color: SERIES[1], width: 2 } }], layout, CONFIG);
  }

  renderLL(p: number[], lp1: number, lp2: number) {
    const n1 = -lp1, n2 = -lp2, tot = Math.max(Math.abs(n1) + Math.abs(n2), 1e-9);
    this.el.llbar.innerHTML =
      `<span style="width:${(100 * Math.abs(n1)) / tot}%;background:${SERIES[0]}"></span>` +
      `<span style="width:${(100 * Math.abs(n2)) / tot}%;background:${SERIES[1]}"></span>`;
    this.el.lllegend.innerHTML =
      `<span>x = (${p[0].toFixed(2)}, ${p[1].toFixed(2)})</span>` +
      `<span><i class="swatch" style="background:${SERIES[0]}"></i>−log p(x₁) = ${n1.toFixed(2)}</span>` +
      `<span><i class="swatch" style="background:${SERIES[1]}"></i>−log p(x₂|x₁) = ${n2.toFixed(2)}</span>` +
      `<span><b>total ${(n1 + n2).toFixed(2)} nats</b></span>`;
  }

  renderLoss() {
    const s = this.state; if (!s) return;
    const layout = stripLayout("step", "NLL (nats)", undefined, 380, 200);
    Plotly.react(this.el.loss, [{ type: "scatter", mode: "lines", x: s.history.map((h) => h[0]),
      y: s.history.map((h) => h[1]), line: { color: SERIES[0], width: 2 } }], layout, CONFIG);
  }
}
