import './ts-loader.mjs';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {Session:TFSession}=await import('../src/runtime/session.ts');
const {Session:JaxSession}=await import('../src/runtime/jax-session.ts');
const tfRuntime=await import('../src/runtime/tf.ts'),jax=await import('../src/runtime/jax.ts');
const {unpack}=await import('../src/runtime/arrays.ts');
await tfRuntime.tf.setBackend('cpu');await tfRuntime.tf.ready();await jax.tensors.setup('wasm');
const baseline=JSON.parse(readFileSync(new URL('../public/runtime-baseline.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(readFileSync(new URL('../public/runtime-fixtures.json',import.meta.url),'utf8'));
function same(a,b,path='root'){
  if(typeof a==='number'){assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=2e-4+1e-4*Math.max(Math.abs(a),Math.abs(b)),`${path}: ${a} != ${b}`);return;}
  if(Array.isArray(a)){assert.equal(a.length,b.length,path);a.forEach((v,i)=>same(v,b[i],`${path}[${i}]`));return;}
  if(a&&typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path);for(const k in a)same(a[k],b[k],`${path}.${k}`);return;}
  assert.equal(a,b,path);
}
const paths={ar:['conditional?x1=.1','point?x1=.1&x2=-.2','sample_one'],vae:['encode?x1=.1&x2=-.2','decode?z1=.1&z2=.2','inspect?x1=.1&x2=-.2&resolution=32','sample_one'],flow:['trace?stage=2&x1=.1&x2=-.2','sample_one'],gan:['generate?z1=.1&z2=-.2','discriminate?x1=.1&x2=-.2']};
for(const kind of ['ar','vae','flow','gan']){
  const a=new TFSession(kind,baseline),b=new JaxSession(kind,baseline);await a.init();await b.init();
  for(const phase of ['initial','trained']){
    if(phase==='trained'){
      const f=fixtures.find(f=>f.kind===kind);
      for(let i=0;i<12;i++){
        const train=async(s,runtime,t)=>{const x=t.tensor(unpack(f.batches[i]),f.batches[i].shape),nz=Object.fromEntries(Object.entries(f.noise[i]).map(([k,a])=>[k,t.tensor(unpack(a),a.shape)]));try{await s.model.train(x,nz,true,true);s.step++;}finally{t.dispose([x,nz]);}};
        await train(a,tfRuntime,tfRuntime.tf);await train(b,jax,jax.tensors);
      }
      await a.refresh();await b.refresh();
    }
    same(a.state(),b.state(),`${kind} ${phase} snapshot`);
    for(const path of paths[kind]){const [what,q]=path.split('?'),url=new URL('http://test/?'+(q??''));same(await a.query(what,url),await b.query(what,url),`${kind} ${phase} ${what}`);}
    console.log(`PASS ${kind} ${phase} complete snapshot and query parity`);
  }
  if(kind==='ar'){await a.reset('ring8',{k:3});await b.reset('ring8',{k:3});same(a.state(),b.state(),'AR alternate K/dataset');}
  if(kind==='flow'){await a.reset('two_moons',{layers:2,permute:false});await b.reset('two_moons',{layers:2,permute:false});same(a.state(),b.state(),'flow alternate layers/order/dataset');}
  if(kind==='vae'||kind==='gan'){
    const f=fixtures.find(f=>f.kind===kind),configs=kind==='vae'?[{beta:0},{beta:2},{beta:1}]:[{freeze_g:true,freeze_d:false,d_steps:3},{freeze_g:false,freeze_d:true,d_steps:1},{freeze_g:false,freeze_d:false,d_steps:2}];
    // Use the common PyTorch-warmed fixture to isolate control/optimizer behavior
    // from different float32 trajectories near GAN's leaky-ReLU boundaries.
    if(kind==='gan'){a.model.dispose();b.model.dispose();a.model=new tfRuntime.Model(kind,a.cfg,f.initial);b.model=new jax.Model(kind,b.cfg,f.initial);}
    for(const cfg of configs){
      await a.configure(cfg);await b.configure(cfg);
      const results=[];
      for(const [s,t] of [[a,tfRuntime.tf],[b,jax.tensors]]){
        const x=t.tensor(unpack(f.batches[0]),f.batches[0].shape),nz=Object.fromEntries(Object.entries(f.noise[0]).map(([k,v])=>[k,t.tensor(unpack(v),v.shape)]));
        try {results.push(await s.model.train(x,nz,true,true,()=>({x:t.clone(x),z:t.clone(nz.z_d)})));}finally{t.dispose([x,nz]);}
      }
      same(results[0],results[1],`${kind} config loss`);await a.refresh();await b.refresh();
      same(a.state(),b.state(),`${kind} configured snapshot`);
      if(kind==='vae')assert.equal(b.model.memory().compiledGraphs,1,'VAE retained obsolete beta graphs');
    }
    console.log(`PASS ${kind} live control changes, optimizer state and configured snapshots`);
  }
  a.dispose();b.dispose();assert.equal(tfRuntime.tf.memory().numTensors,0);assert.equal(jax.tensors.memory().numTensors,0);
}
jax.disposeInferenceGraphs();console.log('PASS alternate configurations; all owned tensors disposed');
