// Display only: a pixel value in R maps to a hue on an open arc (0..300 deg),
// squashed by a logistic so every real value gets a color. Models never see this.
export const hue = (v: number) => 300 / (1 + Math.exp(-v));

/** Pixel 1 (crown): full color. */
export const pixelColor = (v: number) => `hsl(${hue(v).toFixed(0)}, 78%, 50%)`;
/** Pixel 2 (trunk): desaturated and darker, so orange reads as brown. */
export const pixel2Color = (v: number) => `hsl(${hue(v).toFixed(0)}, 45%, 32%)`;

/** A Tree drawn as a marker: fill = pixel 1, ring = pixel 2 (colors of `colorBy`). */
export function treeMarker(colorBy: number[][], size = 7, opacity = 1) {
  return {
    size, opacity,
    color: colorBy.map((p) => pixelColor(p[0])),
    line: { color: colorBy.map((p) => pixel2Color(p[1])), width: 2 },
  };
}

export const INK = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
};

export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"];

// Sequential blue, light -> dark: model density.
export const SEQ: [number, string][] = [
  [0, "#fcfcfb"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
  [0.6, "#3987e5"], [0.75, "#256abf"], [0.88, "#184f95"], [1, "#0d366b"],
];

// Diverging red <-> blue with a neutral gray midpoint: discriminator P(real).
export const DIV: [number, string][] = [
  [0, "#c0393a"], [0.25, "#ea9a98"], [0.5, "#f0efec"], [0.75, "#9ec5f4"], [1, "#256abf"],
];
