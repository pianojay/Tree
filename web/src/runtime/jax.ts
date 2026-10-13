import {numpy as np, nn, tree, valueAndGrad, jit} from '@jax-js/jax';
import {kernels, type Config, type Family, type Ops, type Params} from './math';
import * as tensors from './jax-tensors';
import {initialize} from './initialization';
import {unpack, type PackedParams} from './arrays';
export {tensors, unpack};
export {Random} from './initialization';
export type {ArraySpec, PackedParams} from './arrays';
const eagerMath = kernels(tensors.ops);
const inferenceCache = new Map<string, any>();
// Compile kernel calls with every array (including weights) as an explicit input.
// Scalar configuration stays static. This cache never captures a model's weights.
function compiledKernel(name: keyof typeof eagerMath, args: any[]) {
  const inputs: tensors.Tensor[] = [];
  const encode = (a:any):any => a instanceof tensors.Tensor ? {array:inputs.push(a)-1} : Array.isArray(a) ? a.map(encode) : a && typeof a==='object' ? Object.fromEntries(Object.entries(a).map(([k,v])=>[k,encode(v)])) : a;
  const template=args.map(encode), key=JSON.stringify([name,template,inputs.map(a=>a.shape)]);
  let fn=inferenceCache.get(key);
  if(!fn) {
    fn=jit((raw:np.Array[])=>tensors.tidy(()=>{
      const wrapped=raw.map(a=>new tensors.Tensor(a));
      const decode=(a:any):any=>a&&typeof a==='object'&&'array' in a?wrapped[a.array]:Array.isArray(a)?a.map(decode):a&&typeof a==='object'?Object.fromEntries(Object.entries(a).map(([k,v])=>[k,decode(v)])):a;
      const result=(eagerMath[name] as any)(...template.map(decode));
      const retain=(a:any):any=>a instanceof tensors.Tensor?a.value.ref:Array.isArray(a)?a.map(retain):a;
      return retain(result);
    }));
    inferenceCache.set(key,fn);
    if(inferenceCache.size>64) {const oldest=inferenceCache.keys().next().value!;inferenceCache.get(oldest).dispose();inferenceCache.delete(oldest);}
  }
  const result=fn(inputs.map(a=>a.value.ref));
  const wrap=(a:any):any=>Array.isArray(a)?a.map(wrap):new tensors.Tensor(a);
  return wrap(result);
}
export const math=Object.fromEntries(Object.keys(eagerMath).map(name=>[name,(...args:any[])=>compiledKernel(name as keyof typeof eagerMath,args)])) as typeof eagerMath;
export const inferenceGraphCount=()=>inferenceCache.size;
export function disposeInferenceGraphs(){for(const fn of inferenceCache.values())fn.dispose();inferenceCache.clear();}
type A = np.Array;
type P = Params<A>;
// Scope used during tracing: inputs are borrowed, outputs explicitly retain refs.
function scope<T>(fn: (ops: Ops<A>) => T): T {
  const owned: A[] = [], keep = (a: A) => { owned.push(a); return a; };
  const ref = (a: A | number) => typeof a === 'number' ? a : a.ref;
  const ops: Ops<A> = {
    add: (a,b) => keep(a.ref.add(ref(b))), sub: (a,b) => keep(a.ref.sub(ref(b))), mul: (a,b) => keep(a.ref.mul(ref(b))), div: (a,b) => keep(a.ref.div(ref(b))),
    exp: a => keep(np.exp(a.ref)), log: a => keep(np.log(a.ref)), tanh: a => keep(np.tanh(a.ref)), leaky: a => keep(nn.leakyRelu(a.ref,.2)), softplus: a => keep(nn.softplus(a.ref)),
    clip: (a,l,h) => keep(np.clip(a.ref,l,h)), square: a => keep(np.square(a.ref)), neg: a => keep(a.ref.neg()),
    col: (a,i,w=1) => keep(a.ref.slice([],[i,i+w])), matmul: (a,b) => keep(np.matmul(a.ref,b.ref)), concat: (a,axis) => keep(np.concatenate(a.map(x=>x.ref),axis)),
    sum: (a,axis,k=false) => keep(a.ref.sum(axis,{keepdims:k})), mean: a => keep(a.ref.mean()), logsumexp: (a,axis,k=false) => keep(nn.logsumexp(a.ref,axis,{keepdims:k})),
    logsoftmax: a => keep(nn.logSoftmax(a.ref,-1)), zerosLikeColumn: a => keep(np.zeros([a.shape[0],1])),
  };
  try { return fn(ops); } finally { owned.forEach(a => a.dispose()); }
}
export class Model {
  p: Params<tensors.Tensor> = {};
  private states: {m: P; v: P; t: number}[];
  private compiled = new Map<string, any>();
  readonly groups: string[];
  constructor(public kind: Family, public cfg: Config, source?: PackedParams) {
    const initial = source ? Object.fromEntries(Object.entries(source).map(([k,a])=>[k,{shape:a.shape,data:unpack(a)}])) : initialize(kind,cfg);
    for (const [k,a] of Object.entries(initial)) this.p[k] = tensors.tensor(a.data,a.shape);
    this.groups = kind === 'gan' ? ['D','G'] : ['all'];
    this.states = this.groups.map(group => ({t:0, m:Object.fromEntries(this.keys(group).map(k=>[k,np.zerosLike(this.p[k].value.ref)])), v:Object.fromEntries(this.keys(group).map(k=>[k,np.zerosLike(this.p[k].value.ref)]))}));
  }
  keys(group = 'all') { return Object.keys(this.p).filter(k => group === 'all' || k.startsWith(group + '.')); }
  private raw() { return Object.fromEntries(Object.entries(this.p).map(([k,a])=>[k,a.value])); }
  private loss(group: string, cfg: Config) {
    return valueAndGrad((p:P,other:P,x:A,nz:P) => {
      try { return scope(o => kernels(o).loss(this.kind,{...p,...other},x,nz,cfg,group).ref); }
      finally { tree.dispose([p,other,x,nz]); }
    });
  }
  async check(x: tensors.Tensor, nz: Params<tensors.Tensor>) {
    const raw = this.raw(), noise = Object.fromEntries(Object.entries(nz).map(([k,a])=>[k,a.value])); const out:any = {};
    for (const group of this.groups) {
      const selected = Object.fromEntries(this.keys(group).map(k=>[k,raw[k]])), other = Object.fromEntries(Object.entries(raw).filter(([k])=>!this.keys(group).includes(k)));
      const [value,g] = this.loss(group,this.cfg)(tree.ref(selected),tree.ref(other),x.value.ref,tree.ref(noise)) as [A,P];
      try { out[group] = {loss:(await value.ref.data())[0],grads:Object.fromEntries(await Promise.all(Object.entries(g).map(async([k,a])=>[k,await a.ref.data()])))}; }
      finally { tree.dispose([value,g]); }
    }
    return out;
  }
  private update(group: string) {
    // Only mathematical configuration affects tracing. Control flags select calls outside JIT.
    const cfg: Config = {k:this.cfg.k,layers:this.cfg.layers,permute:this.cfg.permute,beta:this.cfg.beta};
    const key = JSON.stringify([group,cfg]); if (this.compiled.has(key)) return this.compiled.get(key);
    // A configuration change discards earlier graph constants instead of accumulating variants.
    for (const [oldKey,fn] of this.compiled) if (oldKey.startsWith(`["${group}"`)) { fn.dispose(); this.compiled.delete(oldKey); }
    const gradients = this.loss(group,cfg), lr=this.kind==='flow'?.002:this.kind==='gan'?.001:.003,b1=this.kind==='gan'?.5:.9;
    const step = (p:P,m:P,v:P,other:P,x:A,nz:P,c1:A,c2:A) => {
      const [loss,g] = gradients(tree.ref(p),tree.ref(other),x.ref,tree.ref(nz)) as [A,P];
      try {
        return scope(o => {
          const next:P={},mn:P={},vn:P={};
          for (const k in p) {
            const mm=o.add(o.mul(m[k],b1),o.mul(g[k],1-b1)), vv=o.add(o.mul(v[k],.999),o.mul(o.square(g[k]),.001));
            const root=np.sqrt(o.mul(vv,c2).ref);
            try { next[k]=o.sub(p[k],o.mul(o.div(o.mul(mm,c1),o.add(root,1e-8)),lr)).ref; mn[k]=mm.ref; vn[k]=vv.ref; }
            finally { root.dispose(); }
          }
          return [next,mn,vn,loss.ref];
        });
      } finally { tree.dispose([p,m,v,other,x,nz,c1,c2,loss,g]); }
    };
    const fn=jit(step); this.compiled.set(key,fn); return fn;
  }
  async train(x:tensors.Tensor,nz:Params<tensors.Tensor>,read=true,diagnostics=true,moreD?:()=>{x:tensors.Tensor;z:tensors.Tensor}):Promise<(number|null)[]> {
    const values:(number|null)[]=[];
    let parts:number[]=[];
    if(this.kind==='vae'&&read&&diagnostics) { const a=tensors.tidy(()=>math.vaeLoss(this.p,x,nz.eps,this.cfg.beta??1)); try {parts=await Promise.all(a.slice(1).map(async t=>(await t.data())[0]));} finally {tensors.dispose(a);} }
    for(let i=0;i<this.groups.length;i++) {
      const group=this.groups[i]; let value:number|null=null;
      if (group==='D'&&this.cfg.freeze_d || group==='G'&&this.cfg.freeze_g) { values.push(null); continue; }
      for(let d=0;d<(group==='D'?(this.cfg.d_steps??1):1);d++) {
        const extra=d>0?moreD?.():undefined, input=extra?.x??x, noise={...nz,...(extra?{z_d:extra.z}:{})};
        try {
          const raw=this.raw(), keys=this.keys(group), p=Object.fromEntries(keys.map(k=>[k,raw[k]])), other=Object.fromEntries(Object.entries(raw).filter(([k])=>!keys.includes(k)));
          const s=this.states[i],t=s.t+1,b1=this.kind==='gan'?.5:.9;
          const [next,m,v,loss]=this.update(group)(tree.ref(p),s.m,s.v,tree.ref(other),input.value.ref,Object.fromEntries(Object.entries(noise).map(([k,a])=>[k,a.value.ref])),np.array(1/(1-b1**t)),np.array(1/(1-.999**t))) as [P,P,P,A];
          for(const k of keys) {this.p[k].dispose();this.p[k]=new tensors.Tensor(next[k]);} this.states[i]={m,v,t};
          try {if(read)value=(await loss.ref.data())[0];}finally{loss.dispose();}
        } finally {if(extra)tensors.dispose(extra);}
      }
      values.push(value);
    }
    if(!read)return [];
    if(this.kind==='gan'&&diagnostics) {
      const fields=tensors.tidy(()=>[
        tensors.ops.mean(tensors.sigmoid(math.mlp(this.p,'D',x,true))),
        tensors.ops.mean(tensors.sigmoid(math.mlp(this.p,'D',math.mlp(this.p,'G',nz.z_g,true),true))),
      ]);
      try {return [...values,...await Promise.all(fields.map(async t=>(await t.data())[0]))];}finally{tensors.dispose(fields);}
    }
    return this.kind==='vae'?[...values,...parts]:values;
  }
  async values() {return Object.fromEntries(await Promise.all(Object.entries(this.p).map(async([k,a])=>[k,await a.data()])));}
  memory() {const arrays=this.states.flatMap(s=>[...Object.values(s.m),...Object.values(s.v)]);return {optimizerArrays:arrays.length,optimizerBytes:arrays.reduce((n,a)=>n+a.shape.reduce((x,y)=>x*y,1)*4,0),compiledGraphs:this.compiled.size};}
  dispose() {tensors.dispose(this.p);for(const s of this.states)tree.dispose([s.m,s.v]);for(const fn of this.compiled.values())fn.dispose();this.compiled.clear();}
}
