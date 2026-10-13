// Retained investigation of a derivative-field difference after independent training.
import './ts-loader.mjs';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
const a=await import('../src/runtime/tf.ts'),b=await import('../src/runtime/jax.ts'),{unpack}=await import('../src/runtime/arrays.ts');
await a.tf.setBackend('cpu');await a.tf.ready();await b.tensors.setup('wasm');
const baseline=JSON.parse(readFileSync(new URL('../public/runtime-baseline.json',import.meta.url))),f=JSON.parse(readFileSync(new URL('../public/runtime-fixtures.json',import.meta.url))).find(f=>f.kind==='gan');
const models=[new a.Model('gan',{},baseline.params.gan),new b.Model('gan',{},baseline.params.gan)],runtimes=[{t:a.tf,math:a.math},{t:b.tensors,math:b.math}];
async function step(i,cfg){for(let j=0;j<2;j++){const m=models[j],t=runtimes[j].t;m.cfg=cfg;const x=t.tensor(unpack(f.batches[i]),f.batches[i].shape),nz=Object.fromEntries(Object.entries(f.noise[i]).map(([k,v])=>[k,t.tensor(unpack(v),v.shape)]));try{await m.train(x,nz,true,true,()=>({x:t.clone(x),z:t.clone(nz.z_d)}));}finally{t.dispose([x,nz]);}}}
for(let i=0;i<12;i++)await step(i,{});
for(const cfg of [{freeze_g:true,d_steps:3},{freeze_d:true},{d_steps:2}])await step(0,cfg);
const rng=new a.Random(1),z=Array.from({length:500},()=>[rng.normal(),rng.normal()])[24],results=[];
for(let j=0;j<2;j++){const m=models[j],{t,math}=runtimes[j];const x=t.tidy(()=>math.mlp(m.p,'G',t.tensor2d([z]),true)),g=t.tidy(()=>t.grad(x=>t.neg(t.sum(t.softplus(t.neg(math.mlp(m.p,'D',x,true))))))(x));results.push({x:Array.from(await x.data()),gradient:Array.from(await g.data()),parameters:await m.values()});t.dispose([x,g]);}
function signs(p,x){const signs=[],preactivations=[];let h=x;for(let layer=0;layer<2;layer++){const w=p[`D.W${layer}`],bias=p[`D.b${layer}`],n=bias.length;h=Array.from({length:n},(_,j)=>{const v=bias[j]+h.reduce((s,x,i)=>s+x*w[i*n+j],0);signs.push(v>=0);preactivations.push(v);return v>=0?v:.2*v;});}return {signs,preactivations};}
const sa=signs(results[0].parameters,results[0].x),sb=signs(results[1].parameters,results[1].x),crossings=sa.signs.flatMap((s,i)=>s!==sb.signs[i]?[{unit:i,tf:sa.preactivations[i],jax:sb.preactivations[i]}]:[]);
let maxParameterDifference=0;for(const k in results[0].parameters)results[0].parameters[k].forEach((v,i)=>maxParameterDifference=Math.max(maxParameterDifference,Math.abs(v-results[1].parameters[k][i])));
// Re-evaluate with identical TF.js weights and position to distinguish trajectory
// differences from an input-gradient translation defect.
for(const [k,v] of Object.entries(results[0].parameters)){const old=models[1].p[k];models[1].p[k]=b.tensors.tensor(v,old.shape);old.dispose();}
const matched=b.tensors.tidy(()=>b.tensors.grad(x=>b.tensors.neg(b.tensors.sum(b.tensors.softplus(b.tensors.neg(b.math.mlp(models[1].p,'D',x,true))))))(b.tensors.tensor2d([results[0].x])));
const alignedGradient=Array.from(await matched.data());matched.dispose();const output={date:'2026-10-13',case:'12 default-initialization updates plus freeze/multiple-D controls',pointIndex:24,maxParameterDifference,tf:{x:results[0].x,gradient:results[0].gradient},jax:{x:results[1].x,gradient:results[1].gradient},crossings,alignedGradient,alignedMaxDifference:Math.max(...alignedGradient.map((v,i)=>Math.abs(v-results[0].gradient[i])))};
mkdirSync(new URL('../../docs/jax-js/results/',import.meta.url),{recursive:true});writeFileSync(new URL('../../docs/jax-js/results/gan-controls.json',import.meta.url),JSON.stringify(output,null,2));console.log(JSON.stringify(output,null,2));models.forEach(m=>m.dispose());b.disposeInferenceGraphs();
