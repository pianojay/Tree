import { Model, Random, math, tf, unpack, type ArraySpec, type PackedParams } from './tf';
import { type Config, type Family, LOG2PI } from './math';

export type Baseline={params:Record<Family,PackedParams>;datasets:Record<string,ArraySpec>;heldout:Record<string,ArraySpec>};
const WINDOW=[-3,3];
export const axis=(lo:number,hi:number,n:number)=>Array.from({length:n},(_,i)=>lo+(hi-lo)*i/(n-1));
const AXIS=axis(-3,3,64),SMALL=axis(-3,3,48),FINE=axis(-3,3,161);
const defaults:Record<Family,Config>={ar:{k:8},vae:{beta:1},flow:{layers:3,permute:true},gan:{d_steps:1,freeze_d:false,freeze_g:false}};
const pts=(a:number[])=>tf.tensor2d(a.flatMap(y=>a.flatMap(x=>[x,y])),[a.length*a.length,2]);
const normals=(r:Random,n:number)=>tf.tensor2d(r.normals(n*2),[n,2]);
export const lines=(nPoints=61)=>{const a=axis(-3,3,nPoints),out:number[][][]=[];for(const t of axis(-2.5,2.5,11)){out.push(a.map(x=>[t,x]),a.map(x=>[x,t]));}return out;};
async function read(fn:()=>Record<string,tf.Tensor>){const r=tf.tidy(fn);const out:Record<string,any>={};try{for(const [k,t] of Object.entries(r))out[k]=await t.array();return out;}finally{tf.dispose(Object.values(r));}}

export class Session {
  model!:Model;cfg:Config;dataset='multimodal';data!:tf.Tensor2D;dataShow:number[][]=[];
  step=0;generation=0;running=false;history:(number|null)[][]=[];snapshot:any;
  private trainRng=new Random(0);private queryRng=new Random(7);private zShow:number[][]=[];
  private tail:Promise<unknown>=Promise.resolve();private timer:ReturnType<typeof setTimeout>|undefined;
  private snapshotAt=0;private snapshotCost=0;private failure:string|undefined;
  constructor(public kind:Family,private baseline:Baseline){this.cfg={...defaults[kind]};}
  serialize<T>(fn:()=>Promise<T>):Promise<T>{const job=this.tail.then(fn);this.tail=job.catch(()=>{});return job;}
  async init(){await this.reset(this.dataset);}
  private initial(){const c=this.cfg;return this.kind==='ar'&&c.k!==8||this.kind==='flow'&&(c.layers!==3||c.permute!==true)?undefined:this.baseline.params[this.kind];}
  async reset(dataset:string,cfg?:Config){
    this.stop();this.model?.dispose();this.data?.dispose();
    if(!this.baseline.datasets[dataset])throw new Error('Unknown dataset');
    this.cfg={...this.cfg,...cfg};this.dataset=dataset;this.generation++;this.step=0;this.history=[];this.failure=undefined;
    this.trainRng=new Random(0);this.queryRng=new Random(7);
    const a=this.baseline.datasets[dataset],values=unpack(a);this.data=tf.tensor2d(values,[2000,2]);
    this.dataShow=Array.from({length:500},(_,i)=>[values[2*i],values[2*i+1]]);
    const z=new Random(1);this.zShow=Array.from({length:500},()=>[z.normal(),z.normal()]);
    this.model=new Model(this.kind,this.cfg,this.initial());await this.refresh();
  }
  stop(){this.running=false;if(this.timer)clearTimeout(this.timer);this.timer=undefined;}
  start(){if(this.running)return;this.running=true;this.schedule();}
  private schedule(){this.timer=setTimeout(()=>{void this.serialize(async()=>{
    if(!this.running)return;
    try{
      for(let n=0;n<2&&this.running&&this.step<20000;n++){
        const ids=Int32Array.from({length:256},()=>Math.floor(this.trainRng.uniform()*2000));
        const x=tf.tidy(()=>tf.gather(this.data,tf.tensor1d(ids,'int32')));
        const nz=tf.tidy(()=>({eps:normals(this.trainRng,256),z_d:normals(this.trainRng,256),z_g:normals(this.trainRng,256)}));
        const log=(this.step+1)%5===0;
        try{const values=await this.model.train(x,nz,log,true,()=>tf.tidy(()=>({x:tf.gather(this.data,tf.tensor1d(Int32Array.from({length:256},()=>Math.floor(this.trainRng.uniform()*2000)),'int32')),z:normals(this.trainRng,256)})));this.step++;if(log)this.history.push([this.step,...values]);}
        finally{tf.dispose([x,...Object.values(nz)]);}
      }
      if(this.step>=20000)this.stop();
      if(performance.now()-this.snapshotAt>Math.max(500,2*this.snapshotCost)||!this.running)await this.refresh();
    }catch(e){this.failure=String(e);this.stop();}
    if(this.running)this.schedule();
  });},0);}
  async pause(){this.stop();await this.refresh();}
  async configure(cfg:Config){this.cfg={...this.cfg,...cfg};this.model.cfg=this.cfg;await this.refresh();}
  state(){return {...this.snapshot,running:this.running,error:this.failure};}
  async refresh(){const t=performance.now();const extra=await this.extra();let heldout:number|undefined;
    if(this.kind==='ar'||this.kind==='flow'){
      const a=this.baseline.heldout?.[this.dataset];if(a){const v=tf.tidy(()=>math.loss(this.kind,this.model.p,tf.tensor2d(unpack(a),[512,2]),{},this.cfg));heldout=(await v.data())[0];v.dispose();}
    }
    this.snapshot={kind:this.kind,dataset:this.dataset,generation:this.generation,step:this.step,history:this.history.slice(-3000),window:WINDOW,cfg:{...this.cfg},data:this.dataShow,heldout_nll:heldout,...extra};this.snapshotAt=performance.now();this.snapshotCost=this.snapshotAt-t;}
  private async sample(n:number):Promise<number[][]>{
    const m=this.model,p=m.p,r=this.queryRng;
    if(this.kind==='ar'){
      const [l,mu,s]=await Promise.all([p.p1_logits,p.p1_mu,p.p1_log_sigma].map(t=>t.data()));
      const pick=(l:ArrayLike<number>,mu:ArrayLike<number>,s:ArrayLike<number>)=>{const max=Math.max(...Array.from(l)),weights=Array.from(l,x=>Math.exp(x-max));let u=r.uniform()*weights.reduce((a,b)=>a+b,0),k=weights.length-1;for(let i=0;i<weights.length;i++)if((u-=weights[i])<=0){k=i;break;}return mu[k]+Math.exp(Math.max(-5,Math.min(2,s[k])))*r.normal();};
      const x1=Array.from({length:n},()=>pick(l,mu,s));const k=this.cfg.k??8;
      const out=await read(()=>({o:math.mlp(p,'cond',tf.tensor2d(x1,[n,1]))}));
      return x1.map((a,i)=>[a,pick(out.o[i].slice(0,k),out.o[i].slice(k,2*k),out.o[i].slice(2*k))]);
    }
    const result=await read(()=>{const z=normals(r,n);if(this.kind==='flow')return {x:math.push(p,z,0,this.cfg.layers??3,this.cfg.permute??true)};if(this.kind==='gan')return {x:math.mlp(p,'G',z,true)};return {x:tf.add(math.mlp(p,'dec',z),tf.mul(tf.exp(math.decoderLogs(p)),normals(r,n)))};});return result.x;
  }
  private async extra(){
    const p=this.model.p,l=lines(this.kind==='flow'?41:61),nPoints=l[0].length,n=l.length*nPoints;
    if(this.kind==='ar'){
      const out=await read(()=>{const grid=pts(AXIS),[a,b]=math.arParts(p,grid,this.cfg.k??8);return {density:tf.reshape(tf.exp(tf.add(a,b)),[64,64]),marginal1:tf.reshape(tf.exp(math.mixture(tf.tensor2d(FINE,[161,1]),p.p1_logits,p.p1_mu,tf.clipByValue(p.p1_log_sigma,-5,2))),[161])};});
      return {...out,axis:AXIS,fine:FINE,samples:await this.sample(500)};
    }
    if(this.kind==='vae'){
      const out=await read(()=>{const [mu,lv]=math.encode(p,tf.tensor2d(this.dataShow)),grid=math.mlp(p,'dec',tf.tensor2d(l.flat(),[n,2]));return {sigma_x:tf.exp(math.decoderLogs(p)),latent_mu:mu,latent_sigma:tf.exp(tf.mul(lv,0.5)),kl_points:tf.reshape(math.klPoints(mu,lv),[500]),grid_lines:tf.reshape(grid,[l.length,nPoints,2])};});
      return {...out,samples:await this.sample(500)};
    }
    if(this.kind==='flow'){
      const stage=await read(()=>{const out:Record<string,tf.Tensor>={},perm=this.cfg.permute??true;for(let k=0;k<=(this.cfg.layers??3);k++){
        const data=tf.tensor2d(this.dataShow.slice(0,300)),total=this.cfg.layers??3;
        let x:tf.Tensor=data;for(let i=total-1;i>=k;i--)x=math.layer(p,x,i,perm,true)[0];
        out[`d${k}`]=tf.reshape(tf.exp(math.flowLogp(p,pts(SMALL),k,perm)),[48,48]);out[`x${k}`]=x;out[`g${k}`]=tf.reshape(math.push(p,tf.tensor2d(l.flat(),[n,2]),0,k,perm),[l.length,nPoints,2]);
      }return out;});
      return {axis:SMALL,stages:Array.from({length:(this.cfg.layers??3)+1},(_,k)=>({density:stage[`d${k}`],data:stage[`x${k}`],grid:stage[`g${k}`]})),orders:Array.from({length:this.cfg.layers??3},(_,k)=>this.cfg.permute&&k%2?'x₂→x₁':'x₁→x₂'),samples:await this.sample(500)};
    }
    const out=await read(()=>{const z=tf.tensor2d(this.zShow),fake=math.mlp(p,'G',z,true);return {fake,grad:tf.grad(x=>tf.neg(tf.sum(tf.softplus(tf.neg(math.mlp(p,'D',x,true))))))(tf.slice(fake,[0,0],[120,2])),d_field:tf.reshape(tf.sigmoid(math.mlp(p,'D',pts(AXIS),true)),[64,64]),grid_lines:tf.reshape(math.mlp(p,'G',tf.tensor2d(l.flat(),[n,2]),true),[l.length,nPoints,2])};});
    return {...out,z:this.zShow,axis:AXIS};
  }
  async query(what:string,u:URL):Promise<any>{
    const q=(k:string)=>Number(u.searchParams.get(k)),p=this.model.p,x1=q('x1'),x2=q('x2'),z1=q('z1'),z2=q('z2');
    if(what==='sample_one'){const [x]=await this.sample(1);return {x1:x[0],x2:x[1]};}
    if(this.kind==='ar'){
      if(what==='point'){const out=await read(()=>{const [a,b]=math.arParts(p,tf.tensor2d([[x1,x2]]),this.cfg.k??8);return {lp1:tf.reshape(a,[]),lp2:tf.reshape(b,[])};});return out;}
      if(what==='conditional')return {...await read(()=>{const k=this.cfg.k??8,o=math.mlp(p,'cond',tf.tensor2d([[x1]]));return {p:tf.reshape(tf.exp(math.mixture(tf.tensor2d(FINE,[161,1]),tf.slice(o,[0,0],[1,k]),tf.slice(o,[0,k],[1,k]),tf.clipByValue(tf.slice(o,[0,2*k],[1,k]),-5,2))),[161]),p1:tf.reshape(tf.exp(math.mixture(tf.tensor2d([[x1]]),p.p1_logits,p.p1_mu,tf.clipByValue(p.p1_log_sigma,-5,2))),[])};}),x1,fine:FINE};
    }
    if(this.kind==='vae'){
      if(what==='decode')return read(()=>({mean:tf.reshape(math.mlp(p,'dec',tf.tensor2d([[z1,z2]])),[2]),sigma:tf.exp(math.decoderLogs(p))}));
      if(what==='inspect')return this.inspect([x1,x2],Math.max(24,Math.min(120,q('resolution')||64)));
      if(what==='encode')return read(()=>{const [mu,lv]=math.encode(p,tf.tensor2d([[x1,x2]])),rng=new Random(7),eps=normals(rng,40),z=tf.add(mu,tf.mul(tf.exp(tf.mul(lv,0.5)),eps));return {mu:tf.reshape(mu,[2]),sigma:tf.reshape(tf.exp(tf.mul(lv,0.5)),[2]),kl:tf.reshape(math.klPoints(mu,lv),[]),recon:tf.reshape(math.mlp(p,'dec',mu),[2]),sigma_x:tf.exp(math.decoderLogs(p)),eps,z,xhat:math.mlp(p,'dec',z)};});
    }
    if(this.kind==='flow'&&what==='trace')return read(()=>{const n=this.cfg.layers??3,perm=this.cfg.permute??true,z=math.pull(p,tf.tensor2d([[x1,x2]]),q('stage'),perm)[0];return {points:tf.stack(Array.from({length:n+1},(_,k)=>tf.reshape(math.push(p,z,0,k,perm),[2])))};});
    if(this.kind==='gan')return read(()=>{const x=what==='generate'?math.mlp(p,'G',tf.tensor2d([[z1,z2]]),true):tf.tensor2d([[x1,x2]]);return {p_real:tf.reshape(tf.sigmoid(math.mlp(p,'D',x,true)),[]),...(what==='generate'?{x:tf.reshape(x,[2])}:{})};});
    throw new Error('Unknown model query');
  }
  async inspect(point:number[],resolution:number){
    const p=this.model.p,span=4,delta=2*span/resolution,zaxis=Array.from({length:resolution},(_,i)=>-span+(i+0.5)*delta),rng=new Random(91);
    const out=await read(()=>{
      const z=pts(zaxis),x=tf.tensor2d([point]),decoded=math.mlp(p,'dec',z),logs=math.decoderLogs(p),[mu,lv]=math.encode(p,x);
      const loglike=tf.neg(tf.sum(tf.add(tf.add(tf.mul(tf.square(tf.div(tf.sub(x,decoded),tf.exp(logs))),0.5),logs),LOG2PI/2),1));
      const logprior=tf.sub(tf.mul(tf.sum(tf.square(z),1),-0.5),LOG2PI);
      const qs=tf.add(mu,tf.mul(tf.exp(tf.mul(lv,0.5)),normals(rng,2048)));
      const loglikq=tf.neg(tf.sum(tf.add(tf.add(tf.mul(tf.square(tf.div(tf.sub(x,math.mlp(p,'dec',qs)),tf.exp(logs))),0.5),logs),LOG2PI/2),1));
      const qlog=tf.neg(tf.sum(tf.add(tf.add(tf.mul(tf.square(tf.div(tf.sub(z,mu),tf.exp(tf.mul(lv,0.5)))),0.5),tf.mul(lv,0.5)),LOG2PI/2),1));
      return {loglike,logjoint:tf.add(loglike,logprior),q:tf.exp(qlog),loglikq,kl:tf.reshape(math.klPoints(mu,lv),[]),mu:tf.reshape(mu,[2]),sigma:tf.reshape(tf.exp(tf.mul(lv,.5)),[2])};
    });
    const max=Math.max(...out.logjoint),mass=out.logjoint.map((v:number)=>Math.exp(v-max)),logp=max+Math.log(mass.reduce((a:number,b:number)=>a+b,0)*delta*delta);
    const mean=out.loglikq.reduce((a:number,b:number)=>a+b,0)/2048,se=Math.sqrt(out.loglikq.reduce((a:number,b:number)=>a+(b-mean)**2,0)/(2047*2048)),elbo=mean-out.kl;
    const rows=(a:number[])=>Array.from({length:resolution},(_,i)=>a.slice(i*resolution,(i+1)*resolution)),maxLike=Math.max(...out.loglike);
    return {generation:this.generation,step:this.step,x:point,axis:zaxis,likelihood:rows(out.loglike.map((v:number)=>Math.exp(v-maxLike))),posterior:rows(out.logjoint.map((v:number)=>Math.exp(v-logp))),q:rows(out.q),mu:out.mu,sigma:out.sigma,logp,elbo,gap:logp-elbo,mc_se:se,samples:2048,domain:[-span,span],resolution};
  }
  dispose(){this.stop();this.model.dispose();this.data.dispose();}
}
