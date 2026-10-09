import Plotly from "plotly.js-dist-min";
import { get, latest, post } from "./api";
import { INK, SERIES, pixelColor } from "./colors";
import { CONFIG, ellipses, planeLayout, stripLayout, toData } from "./plot";
import { drawGallery } from "./sprites";

type State = {
  step: number; running: boolean; dataset: string; history: number[][];
  window: [number, number]; data: number[][]; beta: number; sigma_x: number[];
  samples: number[][]; latent_mu: number[][]; latent_sigma: number[][];
  kl_points: number[]; grid_lines: number[][][];
};
type Enc = { mu: number[]; sigma: number[]; kl: number; recon: number[] };
type Dec = { mean: number[]; sigma: number[] };

const ZWIN: [number, number] = [-3.5, 3.5];
const N_ELLIPSES = 80;

// fixed standard-normal draws, for showing z = mu + sigma * eps
const EPS: number[][] = Array.from({ length: 40 }, () => {
  const u = Math.random() || 1e-9, v = Math.random();
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
});

const TEMPLATE = `
<div class="controls">
  <label>Dataset <select data-k="dataset"></select></label>
  <button class="primary" data-k="run">Train</button>
  <button data-k="reset">Reset</button>
  <label>β <input type="range" data-k="beta" min="-2" max="1.3" step="0.01" value="0"><b data-k="betaVal">1.00</b></label>
  <label><input type="checkbox" data-k="showGrid" checked> decoder grid</label>
  <label><input type="checkbox" data-k="showEll" checked> posterior ellipses</label>
  <label><input type="checkbox" data-k="showSamples"> decoded samples</label>
  <span class="stat" data-k="stat"></span>
</div>
<div class="grid">
  <div class="col">
    <div class="card">
      <h2>Data plane x</h2>
      <p class="sub">Points wear their own pixel colors (fill = x₁, ring = x₂). Grey mesh: the latent grid pushed through the decoder. Hover a point: its encoding appears on the right.</p>
      <div data-k="plane"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>Latent plane z</h2>
      <p class="sub">Dashed: prior p(z) at 1σ and 2σ. Points: posterior means of the same Trees; ellipses: q(z | x) at 1σ. Hover anywhere: the decoder projects that z back to the data plane.</p>
      <div data-k="latent"></div>
      <div class="readout" data-k="readout"></div>
    </div>
    <div class="card">
      <h2>Tug of war: reconstruction vs. KL</h2>
      <p class="sub">Loss = reconstruction + β · KL(q(z|x) ‖ p(z)). Move β while training.</p>
      <div data-k="loss"></div>
    </div>
    <div class="card">
      <h2>Decoded Trees (z ~ p(z))</h2>
      <canvas class="gallery" data-k="gallery"></canvas>
    </div>
  </div>
</div>`;

export class VAEView {
  el: Record<string, HTMLElement> = {};
  state?: State;
  enc?: Enc & { x: number[] };
  dec?: Dec & { z: number[] };
  busy = false;
  timer?: number;

  constructor(private root: HTMLElement) {
    root.innerHTML = TEMPLATE;
    root.querySelectorAll<HTMLElement>("[data-k]").forEach((n) => (this.el[n.dataset.k!] = n));
    this.wire();
  }

  async activate() {
    const names = await get<string[]>("/api/datasets");
    const sel = this.el.dataset as HTMLSelectElement;
    if (!sel.options.length) sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
    await this.poll(true);
    this.timer = window.setInterval(() => this.poll(), 300);
  }

  deactivate() { window.clearInterval(this.timer); }

  async poll(force = false) {
    const s = await get<State>("/api/vae/state");
    const changed = force || !this.state || s.step !== this.state.step || s.dataset !== this.state.dataset;
    this.state = s;
    (this.el.dataset as HTMLSelectElement).value = s.dataset;
    this.el.run.textContent = s.running ? "Pause" : "Train";
    const last = s.history[s.history.length - 1];
    const meanKL = s.kl_points.reduce((a, b) => a + b, 0) / s.kl_points.length;
    this.el.stat.textContent = `step ${s.step}` +
      (last ? ` · rec ${last[2].toFixed(2)} · KL ${last[3].toFixed(2)}` : "") + ` · mean KL/point ${meanKL.toFixed(2)} nats`;
    if (changed) this.renderAll();
  }

  wire() {
    this.el.run.onclick = () => post("/api/vae/control", { action: this.state?.running ? "pause" : "start" }).then(() => this.poll());
    this.el.reset.onclick = () => this.reset((this.el.dataset as HTMLSelectElement).value);
    (this.el.dataset as HTMLSelectElement).onchange = (e) => this.reset((e.target as HTMLSelectElement).value);
    const beta = this.el.beta as HTMLInputElement;
    beta.oninput = () => {
      const b = 10 ** Number(beta.value);
      this.el.betaVal.textContent = b.toFixed(2);
      post("/api/vae/control", { action: "beta", beta: b });
    };
    for (const k of ["showGrid", "showEll", "showSamples"]) this.el[k].onchange = () => this.renderAll();

    this.el.plane.addEventListener("mousemove", (ev) => this.hoverData(ev));
    this.el.latent.addEventListener("mousemove", (ev) => this.hoverLatent(ev));
    this.el.plane.addEventListener("mouseleave", () => { this.enc = undefined; this.renderAll(); });
    this.el.latent.addEventListener("mouseleave", () => { this.dec = undefined; this.renderAll(); });
  }

  async reset(dataset: string) {
    const b = 10 ** Number((this.el.beta as HTMLInputElement).value);
    await post("/api/vae/control", { action: "reset", dataset, beta: b });
    this.enc = this.dec = undefined;
    await this.poll(true);
  }

  hoverData(ev: MouseEvent) {
    const p = toData(this.el.plane, ev);
    if (p) this.encodeAt(p);
  }

  hoverLatent(ev: MouseEvent) {
    const z = toData(this.el.latent, ev);
    if (z) this.decodeAt(z);
  }

  encodeAt = latest(async (p: number[]) => {
    const e = await get<Enc>(`/api/vae/encode?x1=${p[0].toFixed(3)}&x2=${p[1].toFixed(3)}`);
    this.enc = { ...e, x: p }; this.dec = undefined;
    this.el.readout.textContent =
      `encoder: x = (${p[0].toFixed(2)}, ${p[1].toFixed(2)}) ↦ q(z|x) = N(μ = (${e.mu.map((v) => v.toFixed(2)).join(", ")}), σ = (${e.sigma.map((v) => v.toFixed(2)).join(", ")})) · KL to prior ${e.kl.toFixed(2)} nats`;
    this.renderPlane(); this.renderLatent();
  });

  decodeAt = latest(async (z: number[]) => {
    const d = await get<Dec>(`/api/vae/decode?z1=${z[0].toFixed(3)}&z2=${z[1].toFixed(3)}`);
    this.dec = { ...d, z }; this.enc = undefined;
    this.el.readout.textContent =
      `decoder: z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)}) ↦ p(x|z) = N(mean = (${d.mean.map((v) => v.toFixed(2)).join(", ")}), σₓ = (${d.sigma.map((v) => v.toFixed(2)).join(", ")}))`;
    this.renderPlane(); this.renderLatent();
  });

  renderAll() { this.renderPlane(); this.renderLatent(); this.renderLoss(); this.renderGallery(); }

  treeMarkers(pts: number[][], size = 7) {
    return { size, color: pts.map((p) => pixelColor(p[0])), line: { color: pts.map((p) => pixelColor(p[1])), width: 2 } };
  }

  renderPlane() {
    const s = this.state; if (!s) return;
    const traces: any[] = [];
    if ((this.el.showGrid as HTMLInputElement).checked) {
      const x: (number | null)[] = [], y: (number | null)[] = [];
      s.grid_lines.forEach((line) => { line.forEach((p) => { x.push(p[0]); y.push(p[1]); }); x.push(null); y.push(null); });
      traces.push({ type: "scatter", mode: "lines", x, y, line: { color: INK.muted, width: 1 }, opacity: 0.55 });
    }
    traces.push({ type: "scatter", mode: "markers", x: s.data.map((p) => p[0]), y: s.data.map((p) => p[1]),
      marker: this.treeMarkers(s.data, 6), opacity: this.enc || this.dec ? 0.25 : 1 });
    if ((this.el.showSamples as HTMLInputElement).checked)
      traces.push({ type: "scatter", mode: "markers", x: s.samples.map((p) => p[0]), y: s.samples.map((p) => p[1]),
        marker: { size: 4, color: INK.ink, opacity: 0.6 } });
    if (this.enc) {
      traces.push({ type: "scatter", mode: "markers", x: [this.enc.x[0]], y: [this.enc.x[1]],
        marker: { size: 14, color: "rgba(0,0,0,0)", line: { color: INK.ink, width: 2 } } });
      traces.push({ type: "scatter", mode: "markers", x: [this.enc.recon[0]], y: [this.enc.recon[1]],
        marker: { size: 11, symbol: "x-thin", line: { color: INK.ink, width: 2 } } });
    }
    if (this.dec) {
      const e = ellipses([this.dec.mean], [this.dec.sigma]);
      traces.push({ type: "scatter", mode: "lines", ...e, fill: "toself", fillcolor: "rgba(235,104,52,0.15)",
        line: { color: SERIES[1], width: 2 } });
      traces.push({ type: "scatter", mode: "markers", x: [this.dec.mean[0]], y: [this.dec.mean[1]],
        marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    }
    Plotly.react(this.el.plane, traces, planeLayout("x₁ (pixel 1)", "x₂ (pixel 2)", s.window), CONFIG);
  }

  renderLatent() {
    const s = this.state; if (!s) return;
    const prior = ellipses([[0, 0], [0, 0]], [[1, 1], [2, 2]], 1, 64);
    const traces: any[] = [{ type: "scatter", mode: "lines", ...prior, line: { color: INK.ink2, width: 1.5, dash: "dash" } }];
    if ((this.el.showEll as HTMLInputElement).checked) {
      const k = Math.min(N_ELLIPSES, s.latent_mu.length);
      const e = ellipses(s.latent_mu.slice(0, k), s.latent_sigma.slice(0, k));
      traces.push({ type: "scatter", mode: "lines", ...e, line: { color: INK.muted, width: 1 }, opacity: this.enc || this.dec ? 0.2 : 0.6 });
    }
    traces.push({ type: "scatter", mode: "markers", x: s.latent_mu.map((p) => p[0]), y: s.latent_mu.map((p) => p[1]),
      marker: this.treeMarkers(s.data, 6), opacity: this.enc || this.dec ? 0.25 : 1 });
    if (this.enc) {
      // reparameterization: fixed eps ~ N(0, I) (grey) mapped to z = mu + sigma * eps (orange)
      const z = EPS.map(([a, b]) => [this.enc!.mu[0] + this.enc!.sigma[0] * a, this.enc!.mu[1] + this.enc!.sigma[1] * b]);
      const lx: (number | null)[] = [], ly: (number | null)[] = [];
      EPS.forEach((e, i) => { lx.push(e[0], z[i][0], null); ly.push(e[1], z[i][1], null); });
      traces.push({ type: "scatter", mode: "lines", x: lx, y: ly, line: { color: INK.muted, width: 0.6 }, opacity: 0.5 });
      traces.push({ type: "scatter", mode: "markers", x: EPS.map((e) => e[0]), y: EPS.map((e) => e[1]),
        marker: { size: 4, color: INK.muted } });
      traces.push({ type: "scatter", mode: "markers", x: z.map((p) => p[0]), y: z.map((p) => p[1]),
        marker: { size: 5, color: SERIES[1] } });
      const e = ellipses([this.enc.mu], [this.enc.sigma]);
      traces.push({ type: "scatter", mode: "lines", ...e, line: { color: SERIES[1], width: 2.5 } });
    }
    if (this.dec)
      traces.push({ type: "scatter", mode: "markers", x: [this.dec.z[0]], y: [this.dec.z[1]],
        marker: { size: 12, color: SERIES[1], line: { color: "#fff", width: 2 } } });
    Plotly.react(this.el.latent, traces, planeLayout("z₁", "z₂", ZWIN, 380), CONFIG);
  }

  renderLoss() {
    const s = this.state; if (!s) return;
    const x = s.history.map((h) => h[0]);
    const layout = stripLayout("step", "nats", undefined, 380, 220);
    layout.showlegend = true;
    layout.legend = { orientation: "h", y: 1.12, font: { size: 11, color: INK.ink2 } };
    layout.margin.t = 24;
    Plotly.react(this.el.loss, [
      { type: "scatter", mode: "lines", name: "reconstruction", x, y: s.history.map((h) => h[2]), line: { color: SERIES[0], width: 2 } },
      { type: "scatter", mode: "lines", name: "KL", x, y: s.history.map((h) => h[3]), line: { color: SERIES[1], width: 2 } },
    ], layout, CONFIG);
  }

  renderGallery() {
    const s = this.state; if (!s) return;
    drawGallery(this.el.gallery as HTMLCanvasElement, s.samples.slice(0, 72), 24);
  }
}
