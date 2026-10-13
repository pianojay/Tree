import './ts-loader.mjs';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {tf,Model,math,Random,tensor}=await import('../src/runtime/tf.ts');
const {Session}=await import('../src/runtime/session.ts');
await tf.setBackend('cpu');await tf.ready();
const baseline=JSON.parse(readFileSync(new URL('../public/runtime-baseline.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(readFileSync(new URL('../public/runtime-fixtures.json',import.meta.url),'utf8'));
const close=(a,b,tol=1e-4)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const value=fn=>tf.tidy(()=>fn().dataSync());

const ar=new Model('ar',{k:8},baseline.params.ar);
const n=4001,dx=20/(n-1),grid=Float32Array.from({length:n},(_,i)=>-10+i*dx);
const marginal=value(()=>tf.exp(math.mixture(tf.tensor2d(grid,[n,1]),ar.p.p1_logits,ar.p.p1_mu,tf.clipByValue(ar.p.p1_log_sigma,-5,2))));
close(Array.from(marginal).reduce((a,b)=>a+b,0)*dx,1,2e-4);
for(const x1 of [-2,0,2]){const conditional=value(()=>{const k=8,h=math.mlp(ar.p,'cond',tf.tensor2d([[x1]]));return tf.exp(math.mixture(tf.tensor2d(grid,[n,1]),tf.slice(h,[0,0],[1,k]),tf.slice(h,[0,k],[1,k]),tf.clipByValue(tf.slice(h,[0,2*k],[1,k]),-5,2)));});close(Array.from(conditional).reduce((a,b)=>a+b,0)*dx,1,2e-4);}
ar.dispose();console.log('PASS AR marginal and conditional normalization');

const flow=new Model('flow',{layers:3,permute:true},fixtures.find(x=>x.kind==='flow').initial);
const points=[[0.2,-0.4],[1,0.7],[-1.2,0.3]];
for(const point of points){
  const x=value(()=>math.push(flow.p,tf.tensor2d([point]),0,3,true));
  const [z,ld]=tf.tidy(()=>math.pull(flow.p,tf.tensor2d(x,[1,2]),3,true));
  const zs=z.dataSync(),logs=ld.dataSync()[0];point.forEach((v,i)=>close(zs[i],v,2e-5));tf.dispose([z,ld]);
  const jac=[];const h=0.001;
  for(let j=0;j<2;j++){const a=[...point],b=[...point];a[j]+=h;b[j]-=h;const ya=value(()=>math.push(flow.p,tf.tensor2d([a]),0,3,true)),yb=value(()=>math.push(flow.p,tf.tensor2d([b]),0,3,true));jac.push([(ya[0]-yb[0])/(2*h),(ya[1]-yb[1])/(2*h)]);}
  const determinant=jac[0][0]*jac[1][1]-jac[1][0]*jac[0][1];close(Math.log(Math.abs(determinant)),-logs,0.002);
}
flow.dispose();console.log('PASS flow inverse, Jacobian, and change of variables');

const mu=[0.3,-0.4],lv=[Math.log(.7**2),Math.log(1.2**2)];
const got=value(()=>math.klPoints(tf.tensor2d([mu]),tf.tensor2d([lv])))[0];
const expected=.5*mu.reduce((a,x,i)=>a+x*x+Math.exp(lv[i])-1-lv[i],0);close(got,expected,1e-6);
const vae=new Model('vae',{beta:1},baseline.params.vae),fx=fixtures.find(x=>x.kind==='vae');const x=tensor(fx.batches[0]),eps=tensor(fx.noise[0].eps);
const losses=value(()=>{const a=math.vaeLoss(vae.p,x,eps,0),b=math.vaeLoss(vae.p,x,eps,1);return tf.stack([a[0],b[0],b[2]]);});close(losses[1]-losses[0],losses[2],1e-5);tf.dispose([x,eps]);vae.dispose();
console.log('PASS Gaussian KL and beta objective identity');

const session=new Session('vae',baseline);await session.init();
for(const [k,p] of Object.entries(session.model.p))if(k.startsWith('enc.')||k.startsWith('dec.'))tf.tidy(()=>p.assign(tf.zerosLike(p)));
const point=[.4,-.6],logEvidence=-point.reduce((s,x)=>s+.5*(x/.3)**2+Math.log(.3)+.5*Math.log(2*Math.PI),0);
const resolutions=[];for(const n of [32,64,96]){const r=await session.inspect(point,n);close(r.logp,logEvidence,0.0003);close(r.elbo,logEvidence,0.0001);assert.ok(r.mc_se<1e-6);resolutions.push(r.logp);}
assert.ok(Math.max(...resolutions)-Math.min(...resolutions)<.0001);session.dispose();console.log('PASS evidence quadrature convergence against analytic independent decoder');

const gan=new Model('gan',{freeze_g:true,d_steps:1},baseline.params.gan),gfix=fixtures.find(x=>x.kind==='gan');
const gx=tensor(gfix.batches[0]),nz=Object.fromEntries(Object.entries(gfix.noise[0]).map(([k,a])=>[k,tensor(a)]));
const before=await gan.values();await gan.train(gx,nz,true,false);const after=await gan.values();
for(const k in before)if(k.startsWith('G.'))assert.deepEqual(after[k],before[k]);
assert.ok(Object.keys(before).some(k=>k.startsWith('D.')&&after[k].some((v,i)=>v!==before[k][i])));
gan.cfg={freeze_d:true};const old=await gan.values();await gan.train(gx,nz,true,false);const last=await gan.values();for(const k in old)if(k.startsWith('D.'))assert.deepEqual(last[k],old[k]);assert.ok(Object.keys(old).some(k=>k.startsWith('G.')&&last[k].some((v,i)=>v!==old[k][i])));
gan.dispose();tf.dispose([gx,...Object.values(nz)]);console.log('PASS GAN player update isolation');

const field=new Model('gan',{},baseline.params.gan),pointX=[.17,-.26];
const grad=value(()=>tf.grad(x=>tf.neg(tf.sum(tf.softplus(tf.neg(math.mlp(field.p,'D',x,true))))))(tf.tensor2d([pointX])));
const logD=p=>value(()=>tf.neg(tf.sum(tf.softplus(tf.neg(math.mlp(field.p,'D',tf.tensor2d([p]),true))))))[0];
for(let i=0;i<2;i++){const a=[...pointX],b=[...pointX];a[i]+=.001;b[i]-=.001;close(grad[i],(logD(a)-logD(b))/.002,.002);}field.dispose();console.log('PASS discriminator input gradient against finite differences');

const multi=new Model('gan',{d_steps:3,freeze_g:true},baseline.params.gan),mx=tensor(gfix.batches[0]),mn=Object.fromEntries(Object.entries(gfix.noise[0]).map(([k,a])=>[k,tensor(a)]));let fresh=0;
await multi.train(mx,mn,true,false,()=>{fresh++;return {x:tf.clone(mx),z:tf.clone(mn.z_d)};});assert.equal(fresh,2);multi.dispose();tf.dispose([mx,...Object.values(mn)]);console.log('PASS fresh inputs for additional discriminator updates');

for(const kind of ['ar','vae','flow','gan']){
  const s=new Session(kind,baseline);await s.init();const count=tf.memory().numTensors;
  for(let i=0;i<4;i++)await s.reset('multimodal');assert.equal(tf.memory().numTensors,count,`${kind} reset leak`);
  const paths={ar:'conditional?x1=0.1',vae:'encode?x1=0.1&x2=-0.2',flow:'trace?stage=2&x1=0.1&x2=-0.2',gan:'generate?z1=0.1&z2=-0.2'};
  for(let i=0;i<12;i++){const [what,q]=paths[kind].split('?');await s.query(what,new URL('http://test/?'+q));}
  assert.equal(tf.memory().numTensors,count,`${kind} query leak`);s.dispose();
}
assert.equal(tf.memory().numTensors,0);console.log('PASS four-family reset/query ownership; zero live tensors after disposal');
