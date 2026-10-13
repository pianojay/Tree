import {Model,tf,tensor,type PackedParams,type ArraySpec,setup} from '../runtime/tf';
import type {Family,Config} from '../runtime/math';
export type Fixture={kind:Family;config:Config;initial:PackedParams;batches:ArraySpec[];noise:PackedParams[];checks:Record<string,{loss:number;grads:PackedParams}>;curve:number[];final:PackedParams;torch:{version:string;threads:number;ms_step:number[]}};
const relative=(a:Record<string,ArrayLike<number>>,b:PackedParams)=>{
  let worst=0;for(const k in b){const ref=tensor(b[k]);const v=ref.dataSync();ref.dispose();let d=0,s=1e-12;for(let i=0;i<v.length;i++){d=Math.max(d,Math.abs(a[k][i]-v[i]));s=Math.max(s,Math.abs(v[i]));}worst=Math.max(worst,d/s);}return worst;
};
export async function compare(fixtures:Fixture[],lib:string,backend:string){
  const t0=performance.now(),jax=lib==='jax-js'?await import('./jax'):undefined;
  if(jax)await jax.setupJax(backend);else await setup(backend);
  const startup=performance.now()-t0,results:any[]=[];
  for(const fx of fixtures){
    let model:any,inputs:any[]=[],noise:any[]=[];
    const create=()=>{
      model=jax?new jax.JaxModel(fx.kind,fx.config,fx.initial):new Model(fx.kind,fx.config,fx.initial);
      inputs=fx.batches.map(a=>jax?jax.array(a):tensor(a));noise=fx.noise.map(a=>jax?jax.arrays(a):Object.fromEntries(Object.entries(a).map(([k,v])=>[k,tensor(v)])));
    };
    const dispose=()=>{model.dispose();if(jax)jax.tree.dispose([inputs,noise]);else tf.dispose([...inputs,...noise.flatMap(n=>Object.values(n) as tf.Tensor[])]);};
    try{
      create();let lossDiff=0,gradRel=0;
      if(jax){const c=await model.check(inputs[0],noise[0]);for(const g in c){lossDiff=Math.max(lossDiff,Math.abs(c[g].loss-fx.checks[g].loss)/Math.max(1,Math.abs(fx.checks[g].loss)));gradRel=Math.max(gradRel,relative(c[g].grads,fx.checks[g].grads));}}
      else for(const g in fx.checks){const r=model.gradients(inputs[0],noise[0],g);const grads=Object.fromEntries(await Promise.all(model.keys(g).map(async(k:string)=>[k,await r.grads[model.p[k].name].data()])));lossDiff=Math.max(lossDiff,Math.abs((await r.value.data())[0]-fx.checks[g].loss)/Math.max(1,Math.abs(fx.checks[g].loss)));gradRel=Math.max(gradRel,relative(grads,fx.checks[g].grads));tf.dispose([r.value,...Object.values(r.grads) as tf.Tensor[]]);}
      dispose();create();let curveRel=0;
      const step=async(i:number)=>{const r=await model.train(inputs[i%inputs.length],noise[i%inputs.length],true,false);return jax?r:fx.kind==='gan'?(r[0]??0)+(r[1]??0):r[0];};
      let first=performance.now();let value=await step(0);first=performance.now()-first;curveRel=Math.abs(value-fx.curve[0])/Math.max(1,Math.abs(fx.curve[0]));
      for(let i=1;i<fx.curve.length;i++){value=await step(i);curveRel=Math.max(curveRel,Math.abs(value-fx.curve[i])/Math.max(1,Math.abs(fx.curve[i])));}
      const paramsRel=relative(await model.values(),fx.final),timings=[];
      for(let rep=0;rep<3;rep++){const start=performance.now();for(let i=0;i<60;i++)await step(i);timings.push((performance.now()-start)/60);}
      results.push({kind:fx.kind,lossDiff,gradRel,curveRel,paramsRel,pass:lossDiff<1e-4&&gradRel<1e-4&&curveRel<1e-3&&paramsRel<1e-3,firstMs:first,msStep:timings,torch:fx.torch});dispose();
    }catch(e){try{dispose();}catch{}results.push({kind:fx.kind,error:String(e)});}
  }
  return {lib,backend,startupMs:startup,ua:typeof navigator==='undefined'?'Node':navigator.userAgent,threads:typeof navigator==='undefined'?null:navigator.hardwareConcurrency,results};
}
