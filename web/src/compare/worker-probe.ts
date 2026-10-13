import {setupJax,JaxModel,arrays,array,np,valueAndGrad,tree} from './jax';
import {unpack,type PackedParams} from '../runtime/arrays';
const relative=(a:Record<string,ArrayLike<number>>,b:PackedParams)=>{let worst=0;for(const k in b){const ref=unpack(b[k]);let d=0,s=1e-12;for(let i=0;i<ref.length;i++){d=Math.max(d,Math.abs(a[k][i]-ref[i]));s=Math.max(s,Math.abs(ref[i]));}worst=Math.max(worst,d/s);}return worst;};
self.onmessage=async()=>{
  try{
    await setupJax('wasm');const response=await fetch(new URL(`${import.meta.env.BASE_URL}runtime-fixtures.json`,self.location.origin));const fixtures=await response.json(),results=[];
    for(const fx of fixtures){const m=new JaxModel(fx.kind,fx.config,fx.initial),xs=fx.batches.map(array),nzs=fx.noise.map(arrays);const c=await m.check(xs[0],nzs[0]);let gradRel=0,curveRel=0;for(const g in c)gradRel=Math.max(gradRel,relative(c[g].grads,fx.checks[g].grads));for(let i=0;i<12;i++){const l=await m.train(xs[i],nzs[i]);curveRel=Math.max(curveRel,Math.abs(l-fx.curve[i])/Math.max(1,Math.abs(fx.curve[i])));}const paramsRel=relative(await m.values(),fx.final);results.push({kind:fx.kind,gradRel,curveRel,paramsRel,pass:gradRel<1e-4&&curveRel<1e-3&&paramsRel<1e-3});m.dispose();tree.dispose([xs,nzs]);}
    const [v,g]=valueAndGrad((x:any)=>np.square(x).sum())(np.array(new Float32Array([1,2])));const gradient=Array.from(await g.data());v.dispose();
    self.postMessage({backend:'wasm',version:'0.1.26',worker:true,results,inputGradient:gradient,pass:results.every(r=>r.pass)&&gradient[0]===2&&gradient[1]===4});
  }catch(e){self.postMessage({error:String(e)});}
};
