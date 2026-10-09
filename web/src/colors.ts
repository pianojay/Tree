// Display only: a pixel value in R maps to a hue on an open arc (0..300 deg),
// squashed by a logistic so every real value gets a color. Models never see this.
export const hue = (v: number) => 300 / (1 + Math.exp(-v));
export const pixelColor = (v: number) => `hsl(${hue(v).toFixed(0)}, 78%, 50%)`;

export const INK = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
};

export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"];

// Sequential blue, light -> dark (lightest recedes into the surface).
export const SEQ: [number, string][] = [
  [0, "#fcfcfb"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
  [0.6, "#3987e5"], [0.75, "#256abf"], [0.88, "#184f95"], [1, "#0d366b"],
];
