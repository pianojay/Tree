import '../style.css';
import {get,post} from '../api';
import {ARView} from '../ar';
const check=(ok:boolean,message:string)=>{if(!ok)throw new Error(message);};
const delay=(ms:number)=>new Promise(r=>setTimeout(r,ms));
const timed=async(fn:()=>Promise<unknown>)=>{const t=performance.now();await fn();return performance.now()-t;};
try{
  const startup=await timed(()=>get('/api/datasets'));
  const checks:any[]=[];
  for(const kind of ['ar','vae','flow','gan']){
    const first:any=await get(`/api/${kind}/state`);
    check(first.step===0,'Initial state not zero');
    await post(`/api/${kind}/control`,{action:'start'});await delay(1200);await post(`/api/${kind}/control`,{action:'pause'});
    const stopped:any=await get(`/api/${kind}/state`);check(stopped.step>0&&!stopped.running,`${kind} training/pause failed`);
    const tensors:any=await get('/api/_runtime');
    await delay(200);const stable:any=await get(`/api/${kind}/state`);check(stable.step===stopped.step,`${kind} continued after pause`);
    const query=kind==='ar'?'conditional?x1=.2':kind==='vae'?'encode?x1=.2&x2=.4':kind==='flow'?'trace?stage=2&x1=.2&x2=.4':'generate?z1=.2&z2=.4';
    const times=[];for(let i=0;i<20;i++)times.push(await timed(()=>get(`/api/${kind}/${query}`)));
    const after:any=await get('/api/_runtime');check(after.tensors===tensors.tensors,`${kind} query leak`);
    await post(`/api/${kind}/control`,{action:'reset',dataset:'multimodal'});
    const reset:any=await get(`/api/${kind}/state`);check(reset.step===0&&reset.generation>first.generation,`${kind} reset failed`);
    const count:any=await get('/api/_runtime');for(let i=0;i<4;i++)await post(`/api/${kind}/control`,{action:'reset',dataset:'multimodal'});
    const last:any=await get('/api/_runtime');check(count.tensors===last.tensors,`${kind} reset leak`);
    if(kind==='ar'||kind==='flow')check(stopped.heldout_nll<first.heldout_nll,`${kind} held-out NLL did not improve`);
    checks.push({kind,step:stopped.step,heldoutBefore:first.heldout_nll,heldoutAfter:stopped.heldout_nll,queryMs:times,stableTensors:after.tensors,resetTensors:last.tensors,pass:true});
  }
  const v=new ARView(document.querySelector('#test-view')!);await v.activate();let frames=0,done=false;const raf=()=>{if(done)return;frames++;requestAnimationFrame(raf);};requestAnimationFrame(raf);
  await post('/api/ar/control',{action:'start'});const begin:any=await get('/api/ar/state'),t=performance.now();await delay(5000);await post('/api/ar/control',{action:'pause'});const end:any=await get('/api/ar/state'),secs=(performance.now()-t)/1000;done=true;
  const renderMs=Array.from(document.querySelectorAll<HTMLElement>('[data-render-ms]')).map(e=>Number(e.dataset.renderMs));v.deactivate();
  const r={startupMs:startup,ua:navigator.userAgent,visibility:document.visibilityState,checks,arDrawing:{fps:frames/secs,stepsPerSec:(end.step-begin.step)/secs,canvasDrawMs:renderMs},runtime:await get('/api/_runtime')};
  check(r.arDrawing.fps>=30,'Drawing fell below 30 fps in the five-second check');
  document.querySelector('#results')!.textContent=JSON.stringify(r,null,2);document.querySelector('#status')!.textContent='Verification passed';
}catch(e){document.querySelector('#status')!.textContent=`Verification failed: ${e}`;}
