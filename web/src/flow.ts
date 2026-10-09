import Plotly from "plotly.js-dist-min";
import { get, latest } from "./api";
import { INK, SERIES, SEQ, treeMarker } from "./colors";
import { CONFIG, planeLayout, stripLayout, toData } from "./plot";
import { BaseState, View, planeRow } from "./view";

type Stage = { density: number[][]; data: number[][]; grid: number[][][] };
type State = BaseState & { axis: number[]; stages: Stage[]; orders: string[]; samples: number[][] };

const SIZE = 290;
const SUB = "₀₁₂₃₄₅₆₇₈₉";
const sub = (k: number) => String(k).split("").map((d) => SUB[Number(d)]).join("");

const TEMPLATE = `
<div class="controls">
  <label>Dataset <select data-k="dataset"></select></label>
  <label>Layers L <select data-k="layers">${[1, 2, 3, 4, 5, 6].map((k) => `<option>${k}</option>`).join("")}</select></label>
  <label><input type="checkbox" data-k="permute" checked> permute between layers</label>
  <button class="primary" data-k="run">Train</button>
  <button data-k="reset">Reset</button>
  <label><input type="checkbox" data-k="showDensity" checked> <i class="dens-swatch"></i> density</label>
  <label><input type="checkbox" data-k="showData" checked> data, mapped back</label>
  <label><input type="checkbox" data-k="showGrid" checked> base grid, pushed forward</label>
  <span class="stat" data-k="stat"></span>
</div>
<div class="plane-row" data-k="row"></div>
<div class="info-row">
  <div class="card">
    <h2>Training loss (negative log-likelihood)</h2>
    <p class="sub">−log p(x) = −log N(z) − Σₖ log |det ∂f<sub>k</sub>⁻¹|: exact, layer by layer.</p>
    <div data-k="loss"></div>
  </div>
  <div class="card">
    <h2>Reading the chain</h2>
    <ul class="legend-list">
      <li><span class="key"><i class="dens-swatch"></i></span>density of the distribution at each stage: N(0, I) pushed through f₁ … f<sub>k</sub> (√ scale, each plane scaled to its own maximum)</li>
      <li><span class="key"><i class="tree-dot"></i></span>training Trees mapped back through f⁻¹, wearing their own pixel colors; at stage 0 they should look like N(0, I)</li>
      <li><span class="key" style="color:${INK.muted}">▦</span>a square grid on the base, pushed forward: the rubber sheet each layer stretches and squeezes</li>
      <li><span class="key" style="color:${SERIES[1]}">●</span>hover any plane: the same point traced through every stage</li>
      <li>Each layer: one pixel gets an elementwise affine map; the other gets an affine map whose shift and scale depend on the first. With "permute" off every layer has the same order, and the chain can never do better than one layer.</li>
    </ul>
    <div class="readout" data-k="readout"></div>
  </div>
</div>`;

export class FlowView extends View<State> {
  kind = "flow";
  planes: HTMLElement[] = [];
  built = "";
  trace?: number[][];

  constructor(root: HTMLElement) {
    super(root, TEMPLATE);
    this.wireCommon();
    this.el.layers.onchange = () => this.reset();
    this.el.permute.onchange = () => this.reset();
    for (const k of ["showDensity", "showData", "showGrid"]) this.el[k].onchange = () => this.renderPlanes();
  }

  resetConfig() {
    return { layers: Number((this.el.layers as HTMLSelectElement).value), permute: (this.el.permute as HTMLInputElement).checked };
  }
  onReset() { this.trace = undefined; }

  statusText(s: State) {
    (this.el.layers as HTMLSelectElement).value = String(s.cfg.layers);
    (this.el.permute as HTMLInputElement).checked = !!s.cfg.permute;
    const last = s.history[s.history.length - 1];
    return `step ${s.step}` + (last ? ` · NLL ${last[1].toFixed(3)}` : "");
  }

  checked(k: string) { return (this.el[k] as HTMLInputElement).checked; }

  layoutPlanes(s: State) {
    const key = s.orders.join("|");
    if (key === this.built) return;
    this.built = key;
    const L = s.orders.length;
    const specs = Array.from({ length: L + 1 }, (_, k) => ({
      key: String(k),
      title: k === 0 ? "z = y₀ ~ N(0, I)" : k === L ? `x = y${sub(k)} (data space)` : `y${sub(k)}`,
      sub: k === 0 ? "base" : `after f${sub(k)}`,
    }));
    const arrows = s.orders.map((o, i) =>
      `<span>f${sub(i + 1)}</span><span class="glyph">→</span><span class="back">← f${sub(i + 1)}⁻¹</span><span class="back">${o}</span>`);
    const planes = planeRow(this.el.row, specs, arrows);
    this.planes = specs.map((p) => planes[p.key]);
    this.planes.forEach((plane, k) =>
      plane.addEventListener("mousemove", (ev) => { const p = toData(plane, ev); if (p) this.traceAt([k, p[0], p[1]]); }));
  }

  traceAt = latest(async ([k, x1, x2]: number[]) => {
    const r = await get<{ points: number[][] }>(`/api/flow/trace?stage=${k}&x1=${x1.toFixed(3)}&x2=${x2.toFixed(3)}`);
    this.trace = r.points;
    this.el.readout.textContent = r.points.map((p, i) => `y${sub(i)} = (${p[0].toFixed(2)}, ${p[1].toFixed(2)})`).join("  →  ");
    this.renderPlanes();
  });

  render() {
    if (!this.state) return;
    this.layoutPlanes(this.state);
    this.renderPlanes();
    this.renderLoss();
  }

  renderPlanes() {
    const s = this.state; if (!s || this.planes.length !== s.stages.length) return;
    const original = s.stages[s.stages.length - 1].data;   // color every stage's points by the Tree they came from
    s.stages.forEach((st, k) => {
      const t: any[] = [];
      if (this.checked("showDensity")) {
        const z = st.density.map((row) => row.map(Math.sqrt));
        t.push({ type: "heatmap", x: s.axis, y: s.axis, z, colorscale: SEQ, showscale: false, zsmooth: "best" });
      }
      if (this.checked("showGrid")) {
        const x: (number | null)[] = [], y: (number | null)[] = [];
        st.grid.forEach((line) => { line.forEach((p) => { x.push(p[0]); y.push(p[1]); }); x.push(null); y.push(null); });
        t.push({ type: "scatter", mode: "lines", x, y, line: { color: INK.muted, width: 0.8 }, opacity: 0.6 });
      }
      if (this.checked("showData"))
        t.push({ type: "scatter", mode: "markers", x: st.data.map((p) => p[0]), y: st.data.map((p) => p[1]),
          marker: treeMarker(original, 5) });
      if (this.trace)
        t.push({ type: "scatter", mode: "markers", x: [this.trace[k][0]], y: [this.trace[k][1]],
          marker: { size: 13, color: SERIES[1], line: { color: "#fff", width: 2 } } });
      const L = s.stages.length - 1;
      const lab = k === 0 ? ["z₁", "z₂"] : k === L ? ["x₁", "x₂"] : [`y${sub(k)},₁`, `y${sub(k)},₂`];
      Plotly.react(this.planes[k], t, planeLayout(lab[0], lab[1], s.window, SIZE), CONFIG);
    });
  }

  renderLoss() {
    const s = this.state; if (!s) return;
    Plotly.react(this.el.loss, [{ type: "scatter", mode: "lines", x: s.history.map((h) => h[0]),
      y: s.history.map((h) => h[1]), line: { color: SERIES[0], width: 2 } }],
      stripLayout("step", "NLL (nats)", undefined, 500, 220), CONFIG);
  }
}
