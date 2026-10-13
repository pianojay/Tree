import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/register_all_gradients.js';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops.js';
import '@tensorflow/tfjs-backend-cpu';
import { kernels, type Config, type Family, type Ops, type Params } from './math';
import {unpack,type ArraySpec,type PackedParams} from './arrays';
export {unpack,type ArraySpec,type PackedParams} from './arrays';

export const ops: Ops<tf.Tensor> = {
  add: (a,b) => tf.add(a,b), sub: (a,b) => tf.sub(a,b), mul: (a,b) => tf.mul(a,b), div: (a,b) => tf.div(a,b),
  exp: tf.exp, log: tf.log, tanh: tf.tanh, leaky: x => tf.leakyRelu(x,0.2), softplus: tf.softplus,
  clip: tf.clipByValue, square: tf.square, neg: tf.neg,
  col: (x,i,w=1) => tf.slice(x,[0,i],[-1,w]), matmul: (a,b) => tf.matMul(a,b), concat: (xs,axis) => tf.concat(xs,axis),
  sum: (x,axis,keep=false) => tf.sum(x,axis,keep), mean: x => tf.mean(x),
  logsumexp: (x,axis,keep=false) => tf.logSumExp(x,axis,keep), logsoftmax: x => tf.logSoftmax(x),
  zerosLikeColumn: x => tf.zeros([x.shape[0],1]),
};
export const math = kernels(ops);
export const tensor = (a: ArraySpec) => tf.tensor(unpack(a),a.shape);

import {initialize} from './initialization';
export {Random, initialize} from './initialization';

export class Model {
  p: Params<tf.Variable>={}; private optimizers: tf.AdamOptimizer[];
  constructor(public kind: Family, public cfg: Config, source?: PackedParams) {
    const init=source?Object.fromEntries(Object.entries(source).map(([k,a])=>[k,{shape:a.shape,data:unpack(a)}])):initialize(kind,cfg);
    for(const [k,a] of Object.entries(init)) this.p[k]=tf.tidy(()=>tf.variable(tf.tensor(a.data,a.shape)));
    const lr=kind==='flow'?0.002:kind==='gan'?0.001:0.003,b1=kind==='gan'?0.5:0.9;
    this.optimizers=Array.from({length:kind==='gan'?2:1},()=>tf.train.adam(lr,b1,0.999,1e-8));
  }
  keys(group='all'){return Object.keys(this.p).filter(k=>group==='all'||k.startsWith(group+'.'));}
  gradients(x:tf.Tensor,nz:Params<tf.Tensor>,group='all') {
    return tf.variableGrads(()=>math.loss(this.kind,this.p,x,nz,this.cfg,group) as tf.Scalar,this.keys(group).map(k=>this.p[k]));
  }
  async train(x:tf.Tensor,nz:Params<tf.Tensor>,read=true,diagnostics=true,moreD?:()=>{x:tf.Tensor;z:tf.Tensor}):Promise<(number|null)[]> {
    let rec:tf.Tensor|undefined,kl:tf.Tensor|undefined;
    if(this.kind!=='gan') {
      const value=this.optimizers[0].minimize(()=>{
        if(this.kind==='vae'){const parts=math.vaeLoss(this.p,x,nz.eps,this.cfg.beta??1);if(read&&diagnostics){rec=tf.keep(parts[1]);kl=tf.keep(parts[2]);}return parts[0] as tf.Scalar;}
        return math.loss(this.kind,this.p,x,nz,this.cfg) as tf.Scalar;
      },read,Object.values(this.p));
      if(!value)return [];
      const vals=await Promise.all([value,...(rec&&kl?[rec,kl]:[])].map(t=>t.data()));
      tf.dispose([value,...(rec&&kl?[rec,kl]:[])]);return vals.map(a=>a[0]);
    }
    let d:tf.Scalar|null=null,g:tf.Scalar|null=null;
    if(!this.cfg.freeze_d) for(let i=0;i<(this.cfg.d_steps??1);i++) {d?.dispose();const extra=i>0?moreD?.():undefined;try{d=this.optimizers[0].minimize(()=>math.dLoss(this.p,extra?.x??x,extra?.z??nz.z_d) as tf.Scalar,read,this.keys('D').map(k=>this.p[k]));}finally{if(extra)tf.dispose([extra.x,extra.z]);}}
    if(!this.cfg.freeze_g)g=this.optimizers[1].minimize(()=>math.gLoss(this.p,nz.z_g) as tf.Scalar,read,this.keys('G').map(k=>this.p[k]));
    if(!read)return [];
    if(!diagnostics){const out=await Promise.all([d,g].map(async t=>t?(await t.data())[0]:null));tf.dispose([...(d?[d]:[]),...(g?[g]:[])]);return out;}
    const pr=tf.tidy(()=>tf.mean(tf.sigmoid(math.mlp(this.p,'D',x,true)))),pf=tf.tidy(()=>tf.mean(tf.sigmoid(math.mlp(this.p,'D',math.mlp(this.p,'G',nz.z_g,true),true))));
    const vals=await Promise.all([d,g,pr,pf].map(async t=>t?(await t.data())[0]:null));tf.dispose([...(d?[d]:[]),...(g?[g]:[]),pr,pf]);return vals;
  }
  async values(){return Object.fromEntries(await Promise.all(Object.entries(this.p).map(async([k,v])=>[k,await v.data()])));}
  dispose(){Object.values(this.p).forEach(v=>v.dispose());this.optimizers.forEach(o=>o.dispose());}
}

export async function setup(backend='wasm') {
  if(backend==='wasm') {
    const w=await import('@tensorflow/tfjs-backend-wasm');
    w.setWasmPaths(new URL(`${import.meta.env.BASE_URL}wasm/`,self.location.origin).href);
    // Single-thread WASM also works on ordinary static hosting without isolation headers.
    w.setThreadsCount(1);
  }
  if(!await tf.setBackend(backend))throw new Error(`TensorFlow.js ${backend} could not initialize`);
  await tf.ready();return tf.getBackend();
}
export {tf};
