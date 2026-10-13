import type {Config, Family} from './math';

/** Explicit RNG streams: training and probes never share state. */
export class Random {
  private state: number; private spare?: number;
  constructor(seed=0) { this.state=seed>>>0; }
  uniform() { let t=this.state=(this.state+0x6d2b79f5)>>>0; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296; }
  normal() { if(this.spare!==undefined){const x=this.spare;this.spare=undefined;return x;}const r=Math.sqrt(-2*Math.log(Math.max(this.uniform(),1e-12))),t=2*Math.PI*this.uniform();this.spare=r*Math.sin(t);return r*Math.cos(t); }
  normals(n: number) { return Float32Array.from({length:n},()=>this.normal()); }
}

export function initialize(kind: Family,cfg: Config): Record<string,{shape:number[];data:Float32Array}> {
  const r=new Random(0),p:Record<string,{shape:number[];data:Float32Array}>={};
  const put=(key:string,shape:number[],fn:()=>number)=>{p[key]={shape,data:Float32Array.from({length:shape.reduce((a,b)=>a*b,1)},fn)};};
  const mlp=(prefix:string,input:number,output:number,zeroLast=false)=>{
    [[input,64],[64,64],[64,output]].forEach(([i,o],n)=>{const v=()=>zeroLast&&n===2?0:(2*r.uniform()-1)/Math.sqrt(i);put(`${prefix}.W${n}`,[i,o],v);put(`${prefix}.b${n}`,[o],v);});
  };
  if(kind==='ar'){const k=cfg.k??8;put('p1_logits',[k],()=>0);let i=0;put('p1_mu',[k],()=>k===1?-2:-2+4*(i++)/(k-1));put('p1_log_sigma',[k],()=>-0.5);mlp('cond',1,3*k);}
  if(kind==='vae'){mlp('enc',2,4);mlp('dec',2,2);put('log_sigma_x',[2],()=>Math.log(0.3));}
  if(kind==='flow')for(let n=0;n<(cfg.layers??3);n++){put(`l${n}.a`,[1],()=>0);put(`l${n}.b`,[1],()=>0);mlp(`l${n}.net`,1,2,true);}
  if(kind==='gan'){mlp('G',2,2);mlp('D',2,1);}
  return p;
}

