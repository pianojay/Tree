import { pixelColor, INK } from "./colors";

/** Draw Trees as actual 2-pixel images: pixel 1 on top, pixel 2 below. */
export function drawGallery(canvas: HTMLCanvasElement, points: number[][], cols = 24, cell = 13) {
  const gap = 4;
  const rows = Math.ceil(points.length / cols);
  const w = cols * (cell + gap);
  const h = rows * (2 * cell + gap);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.fillStyle = INK.surface;
  ctx.fillRect(0, 0, w, h);
  points.forEach(([a, b], i) => {
    const x = (i % cols) * (cell + gap);
    const y = Math.floor(i / cols) * (2 * cell + gap);
    ctx.fillStyle = pixelColor(a);
    ctx.fillRect(x, y, cell, cell);
    ctx.fillStyle = pixelColor(b);
    ctx.fillRect(x, y + cell, cell, cell);
  });
}
