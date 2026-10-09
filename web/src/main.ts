import "./style.css";
import { ARView } from "./ar";
import { VAEView } from "./vae";

const views = {
  ar: new ARView(document.getElementById("view-ar")!),
  vae: new VAEView(document.getElementById("view-vae")!),
};
type Key = keyof typeof views;
let current: Key | null = null;

function show(key: Key) {
  if (current) views[current].deactivate();
  current = key;
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${key}`));
  document.querySelectorAll<HTMLElement>(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === key));
  views[key].activate();
  history.replaceState(null, "", `#${key}`);
}

document.querySelectorAll<HTMLElement>(".tab").forEach((t) => (t.onclick = () => show(t.dataset.view as Key)));
show((location.hash.slice(1) as Key) in views ? (location.hash.slice(1) as Key) : "ar");
