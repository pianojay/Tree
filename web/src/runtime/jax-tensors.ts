/** Borrowed-input tensor facade for the shared session code.
 * Every wrapper owns one jax reference; scopes release all unreturned wrappers.
 * Counts describe this facade's owned arrays, not backend/JIT allocator memory.
 */
import {numpy as np, nn, grad as jaxGrad, jit, init, defaultDevice} from '@jax-js/jax';
import type {Ops} from './math';
let scopes: Set<Tensor>[] = [];
const live = new Set<Tensor>();
export class Tensor {
  private disposed = false;
  constructor(public value: np.Array) { live.add(this); scopes.at(-1)?.add(this); }
  get shape() { return this.value.shape; }
  async data() { return this.value.ref.data(); }
  dataSync() { return this.value.ref.dataSync(); }
  async array(): Promise<any> {
    const data = await this.data(); let i = 0;
    const nest = (axis: number): any => axis === this.shape.length ? data[i++] : Array.from({length: this.shape[axis]}, () => nest(axis + 1));
    return nest(0);
  }
  dispose() { if (!this.disposed) { this.disposed = true; this.value.dispose(); live.delete(this); } }
}
export type Tensor2D = Tensor;
const wrap = (a: np.Array) => new Tensor(a);
const ref = (a: Tensor | number) => typeof a === 'number' ? a : a.value.ref;
function leaves(value: any, out = new Set<Tensor>()): Set<Tensor> {
  if (value instanceof Tensor) out.add(value);
  else if (Array.isArray(value)) value.forEach(v => leaves(v, out));
  else if (value && Object.getPrototypeOf(value) === Object.prototype) Object.values(value).forEach(v => leaves(v, out));
  return out;
}
export function tidy<T>(fn: () => T): T {
  const scope = new Set<Tensor>(); scopes.push(scope); let result: T;
  try { result = fn(); } catch (error) { scope.forEach(t => t.dispose()); throw error; }
  finally { scopes.pop(); }
  const retained = leaves(result); scope.forEach(t => { if (!retained.has(t)) t.dispose(); else scopes.at(-1)?.add(t); });
  return result;
}
export function dispose(value: any) { leaves(value).forEach(t => t.dispose()); }
export function tensor(data: any, shape?: number[], dtype?: string) {
  const a = np.array(data, {dtype: dtype === 'int32' ? np.int32 : np.float32});
  return wrap(shape ? a.reshape(shape) : a);
}
export const tensor2d = (data: any, shape?: number[]) => tensor(data, shape);
export const tensor1d = (data: any, dtype?: string) => tensor(data, undefined, dtype);
export const add = (a: Tensor, b: Tensor | number) => wrap(a.value.ref.add(ref(b)));
export const sub = (a: Tensor, b: Tensor | number) => wrap(a.value.ref.sub(ref(b)));
export const mul = (a: Tensor, b: Tensor | number) => wrap(a.value.ref.mul(ref(b)));
export const div = (a: Tensor, b: Tensor | number) => wrap(a.value.ref.div(ref(b)));
export const exp = (a: Tensor) => wrap(np.exp(a.value.ref));
export const neg = (a: Tensor) => wrap(a.value.ref.neg());
export const square = (a: Tensor) => wrap(np.square(a.value.ref));
export const softplus = (a: Tensor) => wrap(nn.softplus(a.value.ref));
export const sigmoid = (a: Tensor) => wrap(nn.sigmoid(a.value.ref));
export const clipByValue = (a: Tensor, lo: number, hi: number) => wrap(np.clip(a.value.ref, lo, hi));
export const reshape = (a: Tensor, shape: number[]) => wrap(a.value.ref.reshape(shape));
export const sum = (a: Tensor, axis?: number, keep = false) => wrap(a.value.ref.sum(axis, {keepdims: keep}));
export const stack = (a: Tensor[]) => wrap(np.stack(a.map(t => t.value.ref)));
export const slice = (a: Tensor, begin: number[], size: number[]) => wrap(a.value.ref.slice(...begin.map((v, i): [number, number] => [v, size[i] < 0 ? a.shape[i] : v + size[i]])));
const gatherArray = jit((a:np.Array, indices:np.Array) => np.take(a, indices, 0));
export const gather = (a: Tensor, indices: Tensor) => wrap(gatherArray(a.value.ref, indices.value.ref));
export const clone = (a: Tensor) => wrap(a.value.ref);
export const zerosLike = (a: Tensor) => wrap(np.zerosLike(a.value.ref));
export function grad(fn: (x: Tensor) => Tensor) {
  const f = jaxGrad((raw: np.Array) => tidy(() => { const x = wrap(raw); return fn(x).value.ref; }));
  return (x: Tensor) => wrap(f(x.value.ref));
}
export const ops: Ops<Tensor> = {
  add, sub, mul, div, exp, neg, square, softplus,
  log: a => wrap(np.log(a.value.ref)), tanh: a => wrap(np.tanh(a.value.ref)), leaky: a => wrap(nn.leakyRelu(a.value.ref, .2)),
  clip: clipByValue, col: (a, i, w = 1) => slice(a, [0, i], [-1, w]), matmul: (a, b) => wrap(np.matmul(a.value.ref, b.value.ref)),
  concat: (a, axis) => wrap(np.concatenate(a.map(t => t.value.ref), axis)), sum, mean: a => wrap(a.value.ref.mean()),
  logsumexp: (a, axis, keep = false) => wrap(nn.logsumexp(a.value.ref, axis, {keepdims: keep})),
  logsoftmax: a => wrap(nn.logSoftmax(a.value.ref, -1)), zerosLikeColumn: a => wrap(np.zeros([a.shape[0], 1])),
};
let backend = 'wasm';
export async function setup(device = 'wasm') {
  const available = await init(device as any); if (!available.includes(device as any)) throw new Error(`jax-js ${device} unavailable`);
  defaultDevice(device as any); backend = device; return backend;
}
export const getBackend = () => backend;
export const memory = () => ({numTensors: live.size, numBytes: [...live].reduce((n, t) => n + t.shape.reduce((a, b) => a * b, 1) * 4, 0)});
export const version = '0.1.26';
