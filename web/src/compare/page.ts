import {compare} from './run';
const q=new URLSearchParams(location.search),lib=q.get('lib')??'tfjs',backend=q.get('backend')??'wasm';
try{
  let result:any;
  if(q.get('probe')==='worker'){
    const worker=new Worker(new URL('./worker-probe.ts',import.meta.url),{type:'module'});
    result=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{worker.terminate();reject(new Error('jax-js worker check timed out'));},30000);worker.onmessage=e=>{clearTimeout(timer);resolve(e.data);};worker.onerror=e=>{clearTimeout(timer);reject(new Error(e.message));};worker.postMessage({});});worker.terminate();
  }else if(q.get('probe')==='readback'){
    const {setupJax,np}=await import('./jax');await setupJax(backend);
    document.querySelector('#status')!.textContent='Backend ready; checking synchronous and asynchronous readback…';
    const a=np.array(new Float32Array([1,2,3])).mul(2);const sync=Array.from(a.ref.dataSync());
    const asyncResult=await Promise.race([a.ref.data().then(v=>({values:Array.from(v)})),new Promise(r=>setTimeout(()=>r({error:'Async readback did not finish within 10 seconds'}),10000))]);
    result={lib,backend,version:'0.1.26',probe:'readback',ua:navigator.userAgent,visibility:document.visibilityState,sync,asyncResult};a.dispose();
  }else if(q.get('probe')==='leaky'){
    const {setupJax,np,nn,valueAndGrad}=await import('./jax');await setupJax(backend);
    const x=np.array(new Float32Array([-5,-1,-.1,.1,1,5]));const [v,g]=valueAndGrad((x:any)=>nn.leakyRelu(x,.2).sum())(x.ref);
    const forward=Array.from(nn.leakyRelu(x.ref,.2).dataSync()),gradient=Array.from(g.dataSync());
    result={lib,backend,probe:'leakyRelu',version:'0.1.26',ua:navigator.userAgent,forward,gradient,sum:v.dataSync()[0],expectedForward:[-1,-.2,-.02,.1,1,5],expectedGradient:[.2,.2,.2,1,1,1]};x.dispose();
  }else {const r=await fetch(`${import.meta.env.BASE_URL}runtime-fixtures.json`);if(!r.ok)throw new Error('Fixtures missing');const fixtures=await r.json(),families=q.get('families')?.split(',');result=await compare(families?fixtures.filter((f:any)=>families.includes(f.kind)):fixtures,lib,backend);}
  document.querySelector('#results')!.textContent=JSON.stringify(result,null,2);document.querySelector('#status')!.textContent='Comparison complete';
}
catch(e){document.querySelector('#status')!.textContent=`Comparison failed: ${e}`;}
