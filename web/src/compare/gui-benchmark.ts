import '../style.css';
import {get,post,ENGINE} from '../api';
import renderer from '../draw';
import {ARView} from '../ar';
import {VAEView} from '../vae';
import {FlowView} from '../flow';
import {GANView} from '../gan';
const delay=(ms:number)=>new Promise(r=>setTimeout(r,ms));
const q=new URLSearchParams(location.search),round=Number(q.get('round')??1);
const warmMs=2000,measureMs=6000;
let recording=false,drawMs:number[]=[],viewMs:number[]=[],pollMs:number[]=[];
const original=renderer.react;
renderer.react=(...args:Parameters<typeof original>)=>{const t=performance.now(),r=original(...args);if(recording)drawMs.push(performance.now()-t);return r;};
const timed=async(fn:()=>Promise<unknown>)=>{const t=performance.now();await fn();return performance.now()-t;};
try{
  const coldStartupMs=await timed(()=>get('/api/datasets')),checks:any[]=[];
  const families=['ar','vae','flow','gan'];const order=[...families.slice((round-1)%4),...families.slice(0,(round-1)%4)];
  for(const kind of order){
    document.querySelector('#status')!.textContent=`${ENGINE}, round ${round}: ${kind} warm-up`;
    const Class=({ar:ARView,vae:VAEView,flow:FlowView,gan:GANView} as any)[kind],v=new Class(document.querySelector('#test-view'));
    const render=v.render.bind(v);v.render=async()=>{const t=performance.now();await render();if(recording)viewMs.push(performance.now()-t);};
    const poll=v.poll.bind(v);v.poll=async(force=false)=>{const t=performance.now();await poll(force);if(recording)pollMs.push(performance.now()-t);};
    await v.activate();await post(`/api/${kind}/control`,{action:'reset',dataset:'multimodal'});await v.poll(true);
    await post(`/api/${kind}/control`,{action:'start'});await delay(warmMs);await post(`/api/${kind}/control`,{action:'pause'});await v.poll(true);
    const before:any=await get(`/api/${kind}/state`),metricsBefore:any=await get(`/api/${kind}/metrics`);
    if(before.error)throw new Error(before.error);
    let done=false,previous:number|undefined;const frames:number[]=[];
    const raf=(t:number)=>{if(done)return;if(previous!==undefined)frames.push(t-previous);previous=t;requestAnimationFrame(raf);};
    drawMs=[];viewMs=[];pollMs=[];recording=true;requestAnimationFrame(raf);
    document.querySelector('#status')!.textContent=`${ENGINE}, round ${round}: ${kind} measuring`;
    const start=performance.now();await post(`/api/${kind}/control`,{action:'start'});await delay(measureMs);await post(`/api/${kind}/control`,{action:'pause'});
    const elapsedMs=performance.now()-start;recording=false;done=true;
    const after:any=await get(`/api/${kind}/state`),metricsAfter:any=await get(`/api/${kind}/metrics`);
    if(after.error||after.step<=before.step)throw new Error(`${kind} training failed: ${after.error}`);
    const hover=kind==='ar'?'conditional?x1=.2':kind==='vae'?'encode?x1=.2&x2=.4':kind==='flow'?'trace?stage=2&x1=.2&x2=.4':'generate?z1=.2&z2=.4';
    const coldQueryMs=await timed(()=>get(`/api/${kind}/${hover}`)),queries=[];
    for(let i=0;i<12;i++)queries.push(await timed(()=>get(`/api/${kind}/${hover}`)));
    let inspectMs:number[]|undefined;
    if(kind==='vae'){await get('/api/vae/inspect?x1=.2&x2=.4&resolution=64');inspectMs=[];for(let i=0;i<3;i++)inspectMs.push(await timed(()=>get('/api/vae/inspect?x1=.2&x2=.4&resolution=64')));}
    const snapCount=metricsAfter.snapshot_count-metricsBefore.snapshot_count;
    checks.push({kind,warmupStep:before.step,endStep:after.step,elapsedMs,updatesPerSec:(after.step-before.step)*1000/elapsedMs,
      rafFps:frames.length*1000/frames.reduce((a,b)=>a+b,0),frameIntervalsMs:frames,canvasDrawMs:drawMs,viewRenderMs:viewMs,pollMs,
      renderedViews:viewMs.length,snapshotCount:snapCount,snapshotMeanMs:(metricsAfter.snapshot_total_ms-metricsBefore.snapshot_total_ms)/snapCount,
      snapshotIntervalMs:metricsAfter.snapshot_interval_ms,coldQueryMs,queryMs:queries,inspectMs,
      heldoutBefore:before.heldout_nll,heldoutAfter:after.heldout_nll,snapshotJsonBytes:new TextEncoder().encode(JSON.stringify(after)).length});
    v.deactivate();await post(`/api/${kind}/control`,{action:'pause'});await delay(100);
  }
  const result={engine:ENGINE,round,date:'2026-10-13',warmMs,measureMs,visibility:document.visibilityState,ua:navigator.userAgent,viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,coldStartupMs,runtime:await get('/api/_runtime'),checks};
  document.querySelector('#results')!.textContent=JSON.stringify(result,null,2);document.querySelector('#status')!.textContent='Benchmark complete';
}catch(e){document.querySelector('#status')!.textContent=`Benchmark failed: ${e}`;}
