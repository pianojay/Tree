import Plotly from "plotly.js-dist-min";
import { get, latest } from "./api";
import { INK, SERIES, treeMarker } from "./colors";
import { CONFIG, ellipses, planeLayout, stripLayout, toData } from "./plot";
import { BaseState, View, planeRow } from "./view";

type State = BaseState & {
  sigma_x: number[]; samples: number[][]; latent_mu: number[][]; latent_sigma: number[][];
  kl_points: number[]; grid_lines: number[][][];
};
type Enc = { mu: number[]; sigma: number[]; kl: number; recon: number[]; sigma_x: number[];
  eps: number[][]; z: number[][]; xhat: number[][] };
type Dec = { mean: number[]; sigma: number[] };

const SIZE = 340;
const ZWIN: [number, number] = [-3.5, 3.5];
const N_ELLIPSES = 80;

const TEMPLATE = `
<div class="controls">
  <label>Dataset <select data-k="dataset"></select></label>
  <button class="primary" data-k="run">Train</button>
  <button data-k="reset">Reset</button>
  <label>Layout <select data-k="mode">
    <option value="three">three planes: x → z → x̂</option>
    <option value="two">two planes: data and model share x</option>
  </select></label>
  <label>β <input type="range" data-k="beta" min="0" max="2" step="0.05" value="1"><b data-k="betaVal">1.00</b></label>
  <label><input type="checkbox" data-k="showGrid" checked> decoder grid</label>
  <label><input type="checkbox" data-k="showEll" checked> posterior ellipses</label>
  <label><input type="checkbox" data-k="showSamples" checked> model samples</label>
  <span class="stat" data-k="stat"></span>
</div>
<div class="plane-row" data-k="row"></div>
<div class="info-row">
  <div class="card">
    <h2>Tug of war: reconstruction vs. KL</h2>
    <p class="sub">Loss = reconstruction + β · KL(q(z|x) ‖ p(z)). Move β while training; β = 0 is a plain autoencoder.</p>
    <div data-k="loss"></div>
  </div>
  <div class="card">
    <h2>Reading the planes</h2>
    <ul class="legend-list">
      <li><span class="key"><i class="tree-dot"></i></span>a training Tree in its own pixel colors (fill = x₁, ring = x₂); the same Tree keeps its colors in z</li>
      <li><span class="key" style="color:${INK.ink2}">- -</span>prior p(z) at 1σ and 2σ; grey ellipses: q(z | x) at 1σ</li>
      <li><span class="key" style="color:${INK.muted}">▦</span>decoder grid: the latent grid pushed through the decoder mean</li>
      <li><span class="key" style="color:${INK.ink}">•</span>model samples: z ~ p(z), then x ~ p(x | z)</li>
      <li><span class="key" style="color:${SERIES[1]}">•</span>hover: ε ~ N(0, I) (grey) → z = μ + σ·ε → decoded x̂</li>
    </ul>
    <div class="readout" data-k="readout"></div>
  </div>
</div>`;

export class VAEView extends View<State> {
  kind = "vae";
  enc?: Enc & { x: number[] };
  dec?: Dec & { z: number[] };
  planes: Record<string, HTMLElement> = {};
  mode = "";

  constructor(root: HTMLElement) {
    super(root, TEMPLATE);
    this.wireCommon();
    const beta = this.el.beta as HTMLInputElement;
    beta.oninput = () => {
      this.el.betaVal.textContent = Number(beta.value).toFixed(2);
      this.control("config", { beta: Number(beta.value) });
    };
    this.el.mode.onchange = () => this.render();
    for (const k of ["showGrid", "showEll", "showSamples"]) this.el[k].onchange = () => this.render();
  }

  resetConfig() { return { beta: Number((this.el.beta as HTMLInputElement).value) }; }
  onReset() { this.enc = this.dec = undefined; }

  statusText(s: State) {
    const beta = this.el.beta as HTMLInputElement;
    if (document.activeElement !== beta) {
      beta.value = String(s.cfg.beta);
      this.el.betaVal.textContent = Number(s.cfg.beta).toFixed(2);
    }
    const last = s.history[s.history.length - 1];
    const meanKL = s.kl_points.reduce((a, b) => a + b, 0) / s.kl_points.length;
    return `step ${s.step}` + (last ? ` · rec ${last[2].toFixed(2)} · KL ${last[3].toFixed(2)}` : "") +
      ` · mean KL/point ${meanKL.toFixed(2)} nats`;
  }

  checked(k: string) { return (this.el[k] as HTMLInputElement).checked; }

  /** (Re)build the plane row when the layout changes. */
  layoutPlanes() {
    const mode = (this.el.mode as HTMLSelectElement).value;
    if (mode === this.mode) return;
    this.mode = mode;
    const enc = `<span>encoder</span><span>q(z | x)</span><span class="glyph">→</span>`;
    const dec = `<span>decoder</span><span>p(x | z)</span><span class="glyph">→</span>`;
    this.planes = mode === "three"
      ? planeRow(this.el.row, [
          { key: "x", title: "Data x ~ p<sub>data</sub>", sub: "training Trees" },
          { key: "z", title: "Latent z", sub: "prior and posteriors" },
          { key: "xhat", title: "Model x̂ ~ p<sub>θ</sub>", sub: "what the decoder generates" },
        ], [enc, dec])
      : planeRow(this.el.row, [
          { key: "x", title: "Data space x", sub: "p<sub>data</sub> and p<sub>θ</sub> on one plane" },
          { key: "z", title: "Latent z", sub: "prior and posteriors" },
        ], [`<span>encoder q(z | x)</span><span class="glyph">→</span><span class="glyph back">←</span><span class="back">decoder p(x | z)</span>`]);
    this.planes.x.addEventListener("mousemove", (ev) => { const p = toData(this.planes.x, ev); if (p) this.encodeAt(p); });
    this.planes.z.addEventListener("mousemove", (ev) => { const z = toData(this.planes.z, ev); if (z) this.decodeAt(z); });
    for (const key of Object.keys(this.planes))
      this.planes[key].addEventListener("mouseleave", () => { this.enc = this.dec = undefined; this.renderPlanes(); });
  }

  encodeAt = latest(async (p: number[]) => {
    const e = await get<Enc>(`/api/vae/encode?x1=${p[0].toFixed(3)}&x2=${p[1].toFixed(3)}`);
    this.enc = { ...e, x: p }; this.dec = undefined;
    const f = (v: number[]) => v.map((u) => u.toFixed(2)).join(", ");
    this.el.readout.textContent = `encoder: x = (${f(p)}) ↦ q(z|x) = N(μ = (${f(e.mu)}), σ = (${f(e.sigma)})) · KL to prior ${e.kl.toFixed(2)} nats`;
    this.renderPlanes();
  });

  decodeAt = latest(async (z: number[]) => {
    const d = await get<Dec>(`/api/vae/decode?z1=${z[0].toFixed(3)}&z2=${z[1].toFixed(3)}`);
    this.dec = { ...d, z }; this.enc = undefined;
    const f = (v: number[]) => v.map((u) => u.toFixed(2)).join(", ");
    this.el.readout.textContent = `decoder: z = (${f(z)}) ↦ p(x|z) = N(mean = (${f(d.mean)}), σₓ = (${f(d.sigma)}))`;
    this.renderPlanes();
  });

  render() { this.layoutPlanes(); this.renderPlanes(); this.renderLoss(); }

  renderPlanes() {
    const s = this.state; if (!s) return;
    const three = this.mode === "three";
    const dim = this.enc || this.dec ? 0.25 : 1;
    const xs = (pts: number[][]) => ({ x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) });

    // traces for the model side of data space (decoder grid, samples, hover results)
    const modelTraces = (): any[] => {
      const t: any[] = [];
      if (this.checked("showGrid")) {
        const x: (number | null)[] = [], y: (number | null)[] = [];
        s.grid_lines.forEach((line) => { line.forEach((p) => { x.push(p[0]); y.push(p[1]); }); x.push(null); y.push(null); });
        t.push({ type: "scatter", mode: "lines", x, y, line: { color: INK.muted, width: 1 }, opacity: dim < 1 ? 0.25 : 0.55 });
      }
      if (this.checked("showSamples"))
        t.push({ type: "scatter", mode: "markers", ...xs(s.samples), marker: { size: 3.5, color: INK.ink, opacity: dim < 1 ? 0.2 : 0.55 } });
      if (this.enc) {
        t.push({ type: "scatter", mode: "markers", ...xs(this.enc.xhat), marker: { size: 5, color: SERIES[1] } });
        t.push({ type: "scatter", mode: "markers", x: [this.enc.recon[0]], y: [this.enc.recon[1]],
          marker: { size: 11, symbol: "x-thin", line: { color: INK.ink, width: 2 } } });
      }
      if (this.dec) {
        t.push({ type: "scatter", mode: "lines", ...ellipses([this.dec.mean], [this.dec.sigma]), fill: "toself",
          fillcolor: "rgba(235,104,52,0.15)", line: { color: SERIES[1], width: 2 } });
        t.push({ type: "scatter", mode: "markers", x: [this.dec.mean[0]], y: [this.dec.mean[1]],
          marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
      }
      return t;
    };

    // data plane
    const dataTraces: any[] = three ? [] : modelTraces();
    dataTraces.unshift({ type: "scatter", mode: "markers", ...xs(s.data), marker: treeMarker(s.data, 6, dim) });
    if (this.enc)
      dataTraces.push({ type: "scatter", mode: "markers", x: [this.enc.x[0]], y: [this.enc.x[1]],
        marker: { size: 14, color: "rgba(0,0,0,0)", line: { color: INK.ink, width: 2 } } });
    Plotly.react(this.planes.x, dataTraces, planeLayout("x₁ (pixel 1)", "x₂ (pixel 2)", s.window, SIZE), CONFIG);

    // latent plane
    const prior = ellipses([[0, 0], [0, 0]], [[1, 1], [2, 2]], 1, 64);
    const zt: any[] = [{ type: "scatter", mode: "lines", ...prior, line: { color: INK.ink2, width: 1.5, dash: "dash" } }];
    if (this.checked("showEll")) {
      const k = Math.min(N_ELLIPSES, s.latent_mu.length);
      zt.push({ type: "scatter", mode: "lines", ...ellipses(s.latent_mu.slice(0, k), s.latent_sigma.slice(0, k)),
        line: { color: INK.muted, width: 1 }, opacity: dim < 1 ? 0.2 : 0.6 });
    }
    zt.push({ type: "scatter", mode: "markers", ...xs(s.latent_mu), marker: treeMarker(s.data, 6, dim) });
    if (this.enc) {
      const lx: (number | null)[] = [], ly: (number | null)[] = [];
      this.enc.eps.forEach((e, i) => { lx.push(e[0], this.enc!.z[i][0], null); ly.push(e[1], this.enc!.z[i][1], null); });
      zt.push({ type: "scatter", mode: "lines", x: lx, y: ly, line: { color: INK.muted, width: 0.6 }, opacity: 0.5 });
      zt.push({ type: "scatter", mode: "markers", ...xs(this.enc.eps), marker: { size: 4, color: INK.muted } });
      zt.push({ type: "scatter", mode: "markers", ...xs(this.enc.z), marker: { size: 5, color: SERIES[1] } });
      zt.push({ type: "scatter", mode: "lines", ...ellipses([this.enc.mu], [this.enc.sigma]), line: { color: SERIES[1], width: 2.5 } });
    }
    if (this.dec)
      zt.push({ type: "scatter", mode: "markers", x: [this.dec.z[0]], y: [this.dec.z[1]],
        marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    Plotly.react(this.planes.z, zt, planeLayout("z₁", "z₂", ZWIN, SIZE), CONFIG);

    // model plane (three-plane layout only)
    if (three)
      Plotly.react(this.planes.xhat, modelTraces(), planeLayout("x̂₁", "x̂₂", s.window, SIZE), CONFIG);
  }

  renderLoss() {
    const s = this.state; if (!s) return;
    const x = s.history.map((h) => h[0]);
    const layout = stripLayout("step", "nats", undefined, 500, 220);
    layout.showlegend = true;
    layout.legend = { orientation: "h", y: 1.15, font: { size: 11, color: INK.ink2 } };
    layout.margin.t = 24;
    Plotly.react(this.el.loss, [
      { type: "scatter", mode: "lines", name: "reconstruction", x, y: s.history.map((h) => h[2]), line: { color: SERIES[0], width: 2 } },
      { type: "scatter", mode: "lines", name: "KL", x, y: s.history.map((h) => h[3]), line: { color: SERIES[1], width: 2 } },
    ], layout, CONFIG);
  }
}
