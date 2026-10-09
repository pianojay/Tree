import { get, post } from "./api";

export type BaseState = {
  kind: string; step: number; running: boolean; dataset: string; history: number[][];
  window: [number, number]; data: number[][]; cfg: Record<string, any>;
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
      ${i > 0 ? `<div class="arrow">${arrows[i - 1]}</div>` : ""}
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
    this.el.run.onclick = () => this.control(this.state?.running ? "pause" : "start");
    this.el.reset.onclick = () => this.reset();
    this.el.dataset.onchange = () => this.reset();
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
    const sel = this.el.dataset as HTMLSelectElement;
    if (!sel.options.length) {
      const names = await get<string[]>("/api/datasets");
      sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
    }
    await this.poll(true);
    this.timer = window.setInterval(() => { if (!document.hidden) this.poll(); }, 400);
  }

  deactivate() { window.clearInterval(this.timer); }

  async poll(force = false) {
    const s = await get<S>(`/api/${this.kind}/state`);
    const changed = force || !this.state || s.step !== this.state.step || s.dataset !== this.state.dataset;
    this.state = s;
    (this.el.dataset as HTMLSelectElement).value = s.dataset;
    this.el.run.textContent = s.running ? "Pause" : "Train";
    this.el.stat.textContent = this.statusText(s);
    if (changed) this.render();
  }
}
