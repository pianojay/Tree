import './ts-loader.mjs';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {tensors:t,Model,math,disposeInferenceGraphs}=await import('../src/runtime/jax.ts');
const {Session}=await import('../src/runtime/jax-session.ts');
await t.setup('wasm');
const baseline=JSON.parse(readFileSync(new URL('../public/runtime-baseline.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(readFileSync(new URL('../public/runtime-fixtures.json',import.meta.url),'utf8'));
const unpack=(await import('../src/runtime/arrays.ts')).unpack;
const tensor=a=>t.tensor(unpack(a),a.shape);
const relative=(actual,expected)=>{let d=0,s=1e-12;for(let i=0;i<actual.length;i++){d=Math.max(d,Math.abs(actual[i]-expected[i]));s=Math.max(s,Math.abs(expected[i]));}return d/s;};
for(const f of fixtures){
  const model=new Model(f.kind,f.config,f.initial), x=tensor(f.batches[0]),nz=Object.fromEntries(Object.entries(f.noise[0]).map(([k,a])=>[k,tensor(a)]));
  const checks=await model.check(x,nz);
  for(const group in checks){assert.ok(Math.abs(checks[group].loss-f.checks[group].loss)/Math.max(1,Math.abs(f.checks[group].loss))<1e-4);for(const k in checks[group].grads)assert.ok(relative(checks[group].grads[k],unpack(f.checks[group].grads[k]))<1e-4,`${f.kind} ${k} gradient`);}
  t.dispose([x,nz]);
  for(let i=0;i<f.curve.length;i++){const x=tensor(f.batches[i]),nz=Object.fromEntries(Object.entries(f.noise[i]).map(([k,a])=>[k,tensor(a)]));try{const values=await model.train(x,nz,true,false),loss=values.reduce((a,b)=>a+(b??0),0);assert.ok(Math.abs(loss-f.curve[i])/Math.max(1,Math.abs(f.curve[i]))<1e-3,`${f.kind} curve ${i}`);}finally{t.dispose([x,nz]);}}
  const final=await model.values();for(const k in final)assert.ok(relative(final[k],unpack(f.final[k]))<1e-3,`${f.kind} final ${k}`);
  model.dispose();assert.equal(t.memory().numTensors,0);console.log(`PASS ${f.kind} actual production adapter loss, gradients and Adam trajectory`);
}
for(const kind of ['ar','vae','flow','gan']){
  const s=new Session(kind,baseline);await s.init();const count=t.memory().numTensors;
  for(let i=0;i<4;i++)await s.reset('multimodal');assert.equal(t.memory().numTensors,count,`${kind} reset leak`);
  const paths={ar:['conditional?x1=.1','point?x1=.1&x2=-.2','sample_one'],vae:['encode?x1=.1&x2=-.2','decode?z1=.1&z2=.2','inspect?x1=.1&x2=-.2&resolution=32','sample_one'],flow:['trace?stage=2&x1=.1&x2=-.2','sample_one'],gan:['generate?z1=.1&z2=-.2','discriminate?x1=.1&x2=-.2']};
  const refs=Object.fromEntries(Object.entries(s.model.p).map(([k,a])=>[k,a.value.refCount])),rngState=s.trainRng.state;
  for(let i=0;i<6;i++)for(const path of paths[kind]){const [what,q]=path.split('?');await s.query(what,new URL('http://test/?'+(q??'')));}
  for(const [k,a] of Object.entries(s.model.p))assert.equal(a.value.refCount,refs[k],`${kind} parameter reference leak ${k}`);
  assert.equal(s.trainRng.state,rngState,`${kind} queries consumed training randomness`);
  assert.equal(t.memory().numTensors,count,`${kind} query leak`);
  s.start();await new Promise(r=>setTimeout(r,1500));await s.serialize(()=>s.pause());assert.ok(s.step>0,`${kind} no training: ${s.state().error}`);assert.equal(s.state().error,undefined);
  const step=s.step;await new Promise(r=>setTimeout(r,100));assert.equal(s.step,step);s.dispose();assert.equal(t.memory().numTensors,0);console.log(`PASS ${kind} snapshots, queries, reset, training, pause and disposal`);
}
const close=(a,b,tol=1e-4)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const value=fn=>t.tidy(()=>fn().dataSync());
const ar=new Model('ar',{k:8},baseline.params.ar),n=4001,dx=20/(n-1),grid=Float32Array.from({length:n},(_,i)=>-10+i*dx);
const marginal=value(()=>t.exp(math.mixture(t.tensor2d(grid,[n,1]),ar.p.p1_logits,ar.p.p1_mu,t.clipByValue(ar.p.p1_log_sigma,-5,2))));close(Array.from(marginal).reduce((a,b)=>a+b,0)*dx,1,2e-4);
for(const x1 of [-2,0,2]){const conditional=value(()=>{const h=math.mlp(ar.p,'cond',t.tensor2d([[x1]]));return t.exp(math.mixture(t.tensor2d(grid,[n,1]),t.slice(h,[0,0],[1,8]),t.slice(h,[0,8],[1,8]),t.clipByValue(t.slice(h,[0,16],[1,8]),-5,2)));});close(Array.from(conditional).reduce((a,b)=>a+b,0)*dx,1,2e-4);}ar.dispose();console.log('PASS jax-js AR normalization');
const flow=new Model('flow',{layers:3,permute:true},fixtures.find(f=>f.kind==='flow').initial);
for(const point of [[.2,-.4],[1,.7],[-1.2,.3]]){const x=value(()=>math.push(flow.p,t.tensor2d([point]),0,3,true)),[z,ld]=t.tidy(()=>math.pull(flow.p,t.tensor2d(x,[1,2]),3,true));const zs=await z.data(),logs=(await ld.data())[0];point.forEach((v,i)=>close(zs[i],v,2e-5));t.dispose([z,ld]);const jac=[];for(let j=0;j<2;j++){const a=[...point],b=[...point];a[j]+=.001;b[j]-=.001;const ya=value(()=>math.push(flow.p,t.tensor2d([a]),0,3,true)),yb=value(()=>math.push(flow.p,t.tensor2d([b]),0,3,true));jac.push([(ya[0]-yb[0])/.002,(ya[1]-yb[1])/.002]);}close(Math.log(Math.abs(jac[0][0]*jac[1][1]-jac[1][0]*jac[0][1])),-logs,.002);}flow.dispose();console.log('PASS jax-js flow inverse and finite-difference Jacobian');
const session=new Session('vae',baseline);await session.init();for(const [k,p] of Object.entries(session.model.p))if(k.startsWith('enc.')||k.startsWith('dec.')){session.model.p[k]=t.tidy(()=>t.zerosLike(p));p.dispose();}
const point=[.4,-.6],logEvidence=-point.reduce((s,x)=>s+.5*(x/.3)**2+Math.log(.3)+.5*Math.log(2*Math.PI),0);for(const n of [32,64,96]){const r=await session.inspect(point,n);close(r.logp,logEvidence,.0003);close(r.elbo,logEvidence,.0001);assert.ok(r.mc_se<1e-6);}session.dispose();console.log('PASS jax-js VAE analytic evidence at three resolutions');
const f=fixtures.find(f=>f.kind==='gan'),gan=new Model('gan',{freeze_g:true,d_steps:3},baseline.params.gan),x=tensor(f.batches[0]),nz=Object.fromEntries(Object.entries(f.noise[0]).map(([k,a])=>[k,tensor(a)]));let fresh=0;
const before=await gan.values();await gan.train(x,nz,true,false,()=>{fresh++;return {x:t.clone(x),z:t.clone(nz.z_d)};});assert.equal(fresh,2);const after=await gan.values();for(const k in before)if(k.startsWith('G.'))assert.deepEqual(before[k],after[k]);assert.ok(Object.keys(before).some(k=>k.startsWith('D.')&&after[k].some((v,i)=>v!==before[k][i])));
gan.cfg={freeze_d:true};await gan.train(x,nz,true,false);const last=await gan.values();for(const k in after)if(k.startsWith('D.'))assert.deepEqual(after[k],last[k]);assert.ok(Object.keys(after).some(k=>k.startsWith('G.')&&last[k].some((v,i)=>v!==after[k][i])));
const px=[.17,-.26],grad=value(()=>t.grad(x=>t.neg(t.sum(t.softplus(t.neg(math.mlp(gan.p,'D',x,true))))))(t.tensor2d([px]))),logD=p=>value(()=>t.neg(t.sum(t.softplus(t.neg(math.mlp(gan.p,'D',t.tensor2d([p]),true))))))[0];for(let i=0;i<2;i++){const a=[...px],b=[...px];a[i]+=.001;b[i]-=.001;close(grad[i],(logD(a)-logD(b))/.002,.002);}gan.dispose();t.dispose([x,nz]);
assert.equal(t.memory().numTensors,0);disposeInferenceGraphs();console.log('PASS jax-js GAN isolation, fresh extra D batches and input gradients; all owned tensors disposed');
