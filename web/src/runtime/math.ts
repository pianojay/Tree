/** One set of model equations, used by the production TF.js runtime and comparisons. */
export type Family = 'ar' | 'vae' | 'flow' | 'gan';
export type Config = { k?: number; layers?: number; permute?: boolean; beta?: number; d_steps?: number; freeze_d?: boolean; freeze_g?: boolean };
export type Params<T> = Record<string, T>;
export interface Ops<T> {
  add(a: T, b: T | number): T; sub(a: T, b: T | number): T; mul(a: T, b: T | number): T; div(a: T, b: T | number): T;
  exp(a: T): T; log(a: T): T; tanh(a: T): T; leaky(a: T): T; softplus(a: T): T;
  clip(a: T, lo: number, hi: number): T; square(a: T): T; neg(a: T): T;
  col(a: T, i: number, width?: number): T; matmul(a: T, b: T): T; concat(a: T[], axis: number): T;
  sum(a: T, axis: number, keep?: boolean): T; mean(a: T): T; logsumexp(a: T, axis: number, keep?: boolean): T;
  logsoftmax(a: T): T; zerosLikeColumn(a: T): T;
}
export const LOG2PI = Math.log(2 * Math.PI);

export function kernels<T>(o: Ops<T>) {
  const mlp = (p: Params<T>, prefix: string, x: T, leaky = false) => {
    for (let i = 0; i < 3; i++) {
      x = o.add(o.matmul(x, p[`${prefix}.W${i}`]), p[`${prefix}.b${i}`]);
      if (i < 2) x = leaky ? o.leaky(x) : o.tanh(x);
    }
    return x;
  };
  const mixture = (x: T, logits: T, mu: T, logs: T) => {
    const logn = o.sub(o.sub(o.mul(o.square(o.div(o.sub(x, mu), o.exp(logs))), -0.5), logs), LOG2PI / 2);
    return o.logsumexp(o.add(o.logsoftmax(logits), logn), -1, true);
  };
  const arParts = (p: Params<T>, x: T, k: number) => {
    const x1 = o.col(x, 0), out = mlp(p, 'cond', x1);
    return [mixture(x1, p.p1_logits, p.p1_mu, o.clip(p.p1_log_sigma, -5, 2)),
      mixture(o.col(x, 1), o.col(out, 0, k), o.col(out, k, k), o.clip(o.col(out, 2 * k, k), -5, 2))];
  };
  const encode = (p: Params<T>, x: T) => {
    const h = mlp(p, 'enc', x);
    return [o.col(h, 0, 2), o.clip(o.col(h, 2, 2), -12, 6)];
  };
  const klPoints = (mu: T, lv: T) => o.mul(o.sum(o.sub(o.sub(o.add(o.square(mu), o.exp(lv)), 1), lv), -1, true), 0.5);
  const decoderLogs = (p: Params<T>) => o.clip(p.log_sigma_x, Math.log(0.02), 2);
  const vaeLoss = (p: Params<T>, x: T, eps: T, beta: number) => {
    const [mu, lv] = encode(p, x);
    const z = o.add(mu, o.mul(o.exp(o.mul(lv, 0.5)), eps));
    const decoded = mlp(p, 'dec', z), logs = decoderLogs(p);
    const rec = o.mean(o.sum(o.add(o.add(o.mul(o.square(o.div(o.sub(x, decoded), o.exp(logs))), 0.5), logs), LOG2PI / 2), -1));
    const kl = o.mean(klPoints(mu, lv));
    return [o.add(rec, o.mul(kl, beta)), rec, kl];
  };
  const layer = (p: Params<T>, x: T, n: number, permute: boolean, inverse: boolean): [T, T] => {
    const i = permute ? n % 2 : 0, j = 1 - i, a = p[`l${n}.a`], b = p[`l${n}.b`];
    const xi = inverse ? o.col(x, i) : o.add(o.mul(o.col(x, i), o.exp(b)), a);
    const ts = mlp(p, `l${n}.net`, xi), shift = o.col(ts, 0), scale = o.mul(o.tanh(o.div(o.col(ts, 1), 2)), 2);
    const ui = inverse ? o.mul(o.sub(xi, a), o.exp(o.neg(b))) : xi;
    const uj = inverse ? o.mul(o.sub(o.col(x, j), shift), o.exp(o.neg(scale))) : o.add(shift, o.mul(o.exp(scale), o.col(x, j)));
    return [o.concat(i === 0 ? [ui, uj] : [uj, ui], 1), inverse ? o.neg(o.add(scale, b)) : o.add(scale, b)];
  };
  const pull = (p: Params<T>, x: T, stage: number, permute: boolean): [T, T] => {
    let ld = o.zerosLikeColumn(x);
    for (let n = stage - 1; n >= 0; n--) { const [z, d] = layer(p, x, n, permute, true); x = z; ld = o.add(ld, d); }
    return [x, ld];
  };
  const push = (p: Params<T>, x: T, start: number, end: number, permute: boolean) => {
    for (let n = start; n < end; n++) x = layer(p, x, n, permute, false)[0];
    return x;
  };
  const flowLogp = (p: Params<T>, x: T, stage: number, permute: boolean) => {
    const [z, ld] = pull(p, x, stage, permute);
    return o.add(o.sub(o.mul(o.sum(o.square(z), -1, true), -0.5), LOG2PI), ld);
  };
  const dLoss = (p: Params<T>, x: T, z: T) => o.add(o.mean(o.softplus(o.neg(mlp(p, 'D', x, true)))), o.mean(o.softplus(mlp(p, 'D', mlp(p, 'G', z, true), true))));
  const gLoss = (p: Params<T>, z: T) => o.mean(o.softplus(o.neg(mlp(p, 'D', mlp(p, 'G', z, true), true))));
  const loss = (kind: Family, p: Params<T>, x: T, noise: Params<T>, cfg: Config, group = 'all'): T => {
    if (kind === 'ar') { const [a, b] = arParts(p, x, cfg.k ?? 8); return o.neg(o.mean(o.add(a, b))); }
    if (kind === 'vae') return vaeLoss(p, x, noise.eps, cfg.beta ?? 1)[0];
    if (kind === 'flow') return o.neg(o.mean(flowLogp(p, x, cfg.layers ?? 3, cfg.permute ?? true)));
    return group === 'D' ? dLoss(p, x, noise.z_d) : gLoss(p, noise.z_g);
  };
  return { mlp, mixture, arParts, encode, klPoints, decoderLogs, vaeLoss, layer, pull, push, flowLogp, dLoss, gLoss, loss };
}
