import "./style.css";
import { ARView } from "./ar";
import { FlowView } from "./flow";
import { GANView } from "./gan";
import { VAEView } from "./vae";

const views = {
  ar: new ARView(document.getElementById("view-ar")!),
  vae: new VAEView(document.getElementById("view-vae")!),
  flow: new FlowView(document.getElementById("view-flow")!),
  gan: new GANView(document.getElementById("view-gan")!),
};
type Key = keyof typeof views;
let current: Key | null = null;

function show(key: Key) {
  if (current === key) return;
  if (current) views[current].deactivate();
  current = key;
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${key}`));
  document.querySelectorAll<HTMLElement>(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === key));
  void views[key].activate().catch(e=>views[key].showError(e));
  if (location.hash !== `#${key}`) history.replaceState(null, "", `#${key}`);
}

const fromHash = () => {
  const k = location.hash.slice(1);
  return (k in views ? k : "ar") as Key;
};
document.querySelectorAll<HTMLElement>(".tab").forEach((t) => (t.onclick = () => show(t.dataset.view as Key)));
window.addEventListener("hashchange", () => show(fromHash()));
show(fromHash());
