import { get, post } from "./api";

export type BaseState = {
  kind: string; step: number; running: boolean; dataset: string; history: number[][];
  window: [number, number]; data: number[][]; cfg: Record<string, any>;
  generation?: number; error?: string;
  heldout_nll?: number;
};

export type PlaneSpec = { key: string; title: string; sub?: string };

/**
 * A row of equal-size planes joined by arrows (the maps between them).
 * Returns the plot element of each plane by key.
 */
export function planeRow(row: HTMLElement, planes: PlaneSpec[], arrows: string[]): Record<string, HTMLElement> {
  // each arrow travels with the plane it leads into, so a wrapped row starts with its arrow
  row.innerHTML = planes.map((p, i) => `
    <div class="plane-unit">
      ${i > 0 && arrows[i - 1] ? `<div class="arrow">${arrows[i - 1]}</div>` : ""}
      <div class="card plane-card">
        <h2>${p.title}</h2><p class="sub">${p.sub ?? ""}</p>
        <div data-plane="${p.key}"></div>
      </div>
    </div>`).join("");
  const out: Record<string, HTMLElement> = {};
  row.querySelectorAll<HTMLElement>("[data-plane]").forEach((n) => (out[n.dataset.plane!] = n));
  return out;
}

/** Shared lifecycle: dataset menu, train/pause/reset, polling the server's snapshot. */
export abstract class View<S extends BaseState> {
  el: Record<string, HTMLElement> = {};
  state?: S;
  timer?: number;
  private polling = false;
  private request = 0;
  private active = false;
  abstract kind: string;

  constructor(protected root: HTMLElement, template: string) {
    root.innerHTML = template;
    this.collect(root);
  }

  collect(node: HTMLElement) {
    node.querySelectorAll<HTMLElement>("[data-k]").forEach((n) => (this.el[n.dataset.k!] = n));
  }

  /** Config sent with a reset (rebuilds the model). */
  resetConfig(): Record<string, any> { return {}; }
  abstract render(): void;
  abstract statusText(s: S): string;

  wireCommon() {
    this.el.run.onclick = () => {void this.control(this.state?.running ? "pause" : "start").catch(e=>this.showError(e));};
    this.el.reset.onclick = () => {void this.reset().catch(e=>this.showError(e));};
    this.el.dataset.onchange = () => {void this.reset().catch(e=>this.showError(e));};
  }

  async control(action: string, config?: Record<string, any>) {
    await post(`/api/${this.kind}/control`, { action, config });
    await this.poll();
  }

  async reset() {
    const dataset = (this.el.dataset as HTMLSelectElement).value;
    await post(`/api/${this.kind}/control`, { action: "reset", dataset, config: this.resetConfig() });
    this.onReset();
    await this.poll(true);
  }

  onReset() {}

  async activate() {
    this.active = true;
    this.el.stat.textContent='Loading model…';
    const sel = this.el.dataset as HTMLSelectElement;
    if (!sel.options.length) {
      const names = await get<string[]>("/api/datasets");
      if(!this.active)return;
      sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
    }
    await this.poll(true);
    if(!this.active)return;
    this.timer = window.setInterval(() => { if (!document.hidden) void this.poll().catch(e=>this.showError(e)); }, 500);
  }

  deactivate() { this.active=false;this.request++;window.clearInterval(this.timer);void post(`/api/${this.kind}/control`,{action:'pause'}).catch(e=>this.showError(e)); }
  showError(e:unknown){window.clearInterval(this.timer);this.el.stat.textContent=`Unable to run this model: ${String(e)}`;this.el.stat.setAttribute('role','alert');}

  async poll(force = false) {
    if(!this.active)return;
    if(this.polling&&!force)return;
    const request=++this.request;
    this.polling=true;
    try {
    const s = await get<S>(`/api/${this.kind}/state`);
    if(request!==this.request||!this.active)return;
    if(s.error)throw new Error(s.error);
    if (!s.cfg) {   // the page was rebuilt but the server still runs older code
      this.el.stat.textContent = "⚠ The server is running older code than this page. Restart it (uvicorn server:app).";
      return;
    }
    const changed = force || !this.state || s.generation !== this.state.generation || s.step !== this.state.step || s.dataset !== this.state.dataset;
    this.state = s;
    this.el.stat.removeAttribute('role');
    (this.el.dataset as HTMLSelectElement).value = s.dataset;
    this.el.run.textContent = s.running ? "Pause" : "Train";
    this.el.stat.textContent = this.statusText(s)+(s.heldout_nll!==undefined?` · held-out NLL ${s.heldout_nll.toFixed(3)}`:'');
    if (changed) await this.render();
    }catch(e){this.showError(e);throw e;}finally{this.polling=false;}
  }
}
