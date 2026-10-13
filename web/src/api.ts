export const ENGINE = new URLSearchParams(location.search).get('engine') === 'server' ? 'server' : 'tfjs';
let worker: Worker | undefined;
let next = 0;
const pending = new Map<number, {ok:(v:any)=>void;fail:(e:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
function failAll(error:Error) { for(const p of pending.values()){clearTimeout(p.timer);p.fail(error);}pending.clear();worker?.terminate();worker=undefined; }
function call<T>(method:string,url:string,body?:unknown):Promise<T> {
  if(!worker){worker=new Worker(new URL('./runtime/worker.ts',import.meta.url),{type:'module'});worker.onmessage=e=>{const p=pending.get(e.data.id);if(!p)return;pending.delete(e.data.id);clearTimeout(p.timer);if(e.data.error)p.fail(new Error(e.data.error));else p.ok(e.data.result);};worker.onerror=e=>failAll(new Error(e.message||'Training worker failed'));}
  const id=next++;
  return new Promise((ok,fail)=>{const timer=setTimeout(()=>{failAll(new Error('Training worker timed out. Reload to restart.'));},45000);pending.set(id,{ok,fail,timer});worker!.postMessage({id,method,url,body});});
}
export async function get<T>(url: string): Promise<T> {
  if(ENGINE==='tfjs')return call<T>('GET',url);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

/** Wrap an async handler so rapid calls collapse: the latest argument always runs. */
export function latest<A>(fn: (arg: A) => Promise<void>) {
  let busy = false, pending = false, want: A;
  return async (arg: A) => {
    want = arg; pending = true;
    if (busy) return;
    busy = true;
    try { while (pending) { pending = false; await fn(want); } }
    finally { busy = false; }
  };
}

export async function post(url: string, body: unknown): Promise<void> {
  if(ENGINE==='tfjs')return call<void>('POST',url,body);
  const r=await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if(!r.ok)throw new Error(`${url}: ${r.status}`);
}
