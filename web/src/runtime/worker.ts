import {setup,tf} from './tf';
import {Session,type Baseline} from './session';
import type {Family} from './math';
const sessions=new Map<Family,Session>();
let baseline:Baseline;
const ready=(async()=>{await setup('wasm');const r=await fetch(new URL(`${import.meta.env.BASE_URL}runtime-baseline.json`,self.location.origin));if(!r.ok)throw new Error('Runtime data could not load');baseline=await r.json();})();
let queue=Promise.resolve();
self.onmessage=e=>{
  const {id,method,url,body}=e.data;
  queue=queue.then(async()=>{
    try{
      await ready;const u=new URL(url,self.location.origin),[, ,kind,what]=u.pathname.split('/');
      if(kind==='datasets'){self.postMessage({id,result:Object.keys(baseline.datasets)});return;}
      if(kind==='_runtime'){self.postMessage({id,result:{backend:tf.getBackend(),version:tf.version_core,tensors:tf.memory().numTensors,bytes:tf.memory().numBytes}});return;}
      if(!['ar','vae','flow','gan'].includes(kind))throw new Error('Unknown family');
      const k=kind as Family;let s=sessions.get(k);if(!s){s=new Session(k,baseline);await s.init();sessions.set(k,s);}
      const session=s;
      // Stop promptly, then wait for any already-running chunk before reset/pause.
      if(method==='POST'&&['reset','pause'].includes(body.action))session.stop();
      const result=await session.serialize(async()=>{
        if(method==='POST'){
          if(body.action==='reset')await session.reset(body.dataset??session.dataset,body.config);
          else if(body.action==='pause')await session.pause();
          else if(body.action==='start')session.start();
          else if(body.action==='config')await session.configure(body.config??{});
          else throw new Error('Unknown control');
          return {ok:true};
        }
        if(what==='state')return session.state();
        return {...await session.query(what,u),generation:session.generation,step:session.step};
      });self.postMessage({id,result});
    }catch(e){self.postMessage({id,error:String(e)});}
  });
};
