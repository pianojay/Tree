/** Comparison adapter only. Both runtimes evaluate runtime/math.ts. */
import { numpy as np, nn, tree, valueAndGrad, jit, init, defaultDevice } from '@jax-js/jax';
import {kernels,type Ops,type Family,type Config,type Params} from '../runtime/math';
import {unpack,type PackedParams,type ArraySpec} from '../runtime/arrays';
type A=np.Array;
type P=Params<A>;
function scope<T>(fn:(o:Ops<A>)=>T):T {
  const owned:A[]=[];const keep=(a:A)=>{owned.push(a);return a;};
  const ref=(a:A|number)=>typeof a==='number'?a:a.ref;
  const o:Ops<A>={
    add:(a,b)=>keep(a.ref.add(ref(b))),sub:(a,b)=>keep(a.ref.sub(ref(b))),mul:(a,b)=>keep(a.ref.mul(ref(b))),div:(a,b)=>keep(a.ref.div(ref(b))),
    exp:a=>keep(np.exp(a.ref)),log:a=>keep(np.log(a.ref)),tanh:a=>keep(np.tanh(a.ref)),leaky:a=>keep(nn.leakyRelu(a.ref,0.2)),softplus:a=>keep(nn.softplus(a.ref)),
    clip:(a,l,h)=>keep(np.clip(a.ref,l,h)),square:a=>keep(np.square(a.ref)),neg:a=>keep(a.ref.neg()),
    col:(a,i,w=1)=>keep(a.ref.slice([],[i,i+w])),matmul:(a,b)=>keep(np.matmul(a.ref,b.ref)),concat:(a,axis)=>keep(np.concatenate(a.map(x=>x.ref),axis)),
    sum:(a,axis,k=false)=>keep(a.ref.sum(axis,{keepdims:k})),mean:a=>keep(a.ref.mean()),
    logsumexp:(a,axis,k=false)=>keep(nn.logsumexp(a.ref,axis,{keepdims:k})),logsoftmax:a=>keep(nn.logSoftmax(a.ref,-1)),zerosLikeColumn:a=>keep(np.zeros([a.shape[0],1])),
  };
  try{return fn(o);}finally{for(const a of owned)a.dispose();}
}
export async function setupJax(backend:string){const devices=await init(backend as any);if(!devices.includes(backend as any))throw new Error(`jax-js ${backend} unavailable`);defaultDevice(backend as any);}
export const arrays=(p:PackedParams):P=>Object.fromEntries(Object.entries(p).map(([k,a])=>[k,np.array(unpack(a)).reshape(a.shape)]));
export const array=(a:ArraySpec)=>np.array(unpack(a)).reshape(a.shape);

export class JaxModel {
  private p:P[];private m:P[];private v:P[];private t=0;private groups:string[];
  private gradients:ReturnType<typeof valueAndGrad>[];
  private update:any;
  constructor(kind:Family,cfg:Config,source:PackedParams,useJit=true){
    this.groups=kind==='gan'?['D','G']:['all'];const all=arrays(source);
    this.p=this.groups.map(g=>Object.fromEntries(Object.entries(all).filter(([k])=>g==='all'||k.startsWith(g+'.'))));
    this.m=this.p.map(p=>tree.map((a:A)=>np.zerosLike(a.ref),p));this.v=this.p.map(p=>tree.map((a:A)=>np.zerosLike(a.ref),p));
    this.gradients=this.groups.map(group=>valueAndGrad((p:P,other:P,x:A,nz:P)=>{
      try{return scope(o=>kernels(o).loss(kind,{...p,...other},x,nz,cfg,group).ref);}finally{tree.dispose([p,other,x,nz]);}
    }));
    const lr=kind==='flow'?0.002:kind==='gan'?0.001:0.003,b1=kind==='gan'?0.5:0.9,b2=0.999;
    const step=(ps:P[],ms:P[],vs:P[],x:A,nz:P,c1:A,c2:A)=>{
      const pnew=[...ps],mnew=[...ms],vnew=[...vs];let total:A|undefined;
      for(let i=0;i<this.groups.length;i++){
        const other=Object.assign({},...pnew.filter((_,j)=>i!==j).map(p=>tree.ref(p)));
        const [loss,g]=this.gradients[i](tree.ref(pnew[i]),other,x.ref,tree.ref(nz)) as [A,P];
        const state=scope(o=>{
          const p:P={},m:P={},v:P={};for(const k of Object.keys(pnew[i])){
            const mn=o.add(o.mul(mnew[i][k],b1),o.mul(g[k],1-b1));const vn=o.add(o.mul(vnew[i][k],b2),o.mul(o.square(g[k]),1-b2));
            const denominator=np.sqrt(o.mul(vn,c2).ref);const delta=o.mul(o.div(o.mul(mn,c1),o.add(denominator,1e-8)),lr);denominator.dispose();
            p[k]=o.sub(pnew[i][k],delta).ref;m[k]=mn.ref;v[k]=vn.ref;
          }return [p,m,v];
        });
        tree.dispose([pnew[i],mnew[i],vnew[i],g]);[pnew[i],mnew[i],vnew[i]]=state;
        total=total?total.add(loss):loss;
      }
      tree.dispose([x,nz,c1,c2]);return [pnew,mnew,vnew,total!];
    };
    this.update=useJit?jit(step):step;
  }
  async check(x:A,nz:P){const out:any={};for(let i=0;i<this.groups.length;i++){
    const other=Object.assign({},...this.p.filter((_,j)=>i!==j).map(p=>tree.ref(p)));
    const [value,g]=this.gradients[i](tree.ref(this.p[i]),other,x.ref,tree.ref(nz)) as [A,P];
    out[this.groups[i]]={loss:(await value.data())[0],grads:Object.fromEntries(await Promise.all(Object.entries(g).map(async([k,a])=>[k,await a.ref.data()])))};tree.dispose(g);
  }return out;}
  async train(x:A,nz:P){this.t++;const b1=this.groups.length===2?0.5:0.9,c1=np.array(1/(1-b1**this.t)),c2=np.array(1/(1-0.999**this.t));let value:A;[this.p,this.m,this.v,value]=this.update(this.p,this.m,this.v,x.ref,tree.ref(nz),c1,c2);return (await value.data())[0];}
  async values(){return Object.fromEntries(await Promise.all(this.p.flatMap(p=>Object.entries(p).map(async([k,a])=>[k,await a.ref.data()]))));}
  dispose(){tree.dispose([this.p,this.m,this.v]);if(this.update.dispose)this.update.dispose();}
}
export {np,nn,tree,valueAndGrad};
