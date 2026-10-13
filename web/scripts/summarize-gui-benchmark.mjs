import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const base=new URL('../../docs/jax-js/results/',import.meta.url),runs=JSON.parse(readFileSync(new URL('gui-three-way.json',base)));
const median=a=>{a=[...a].sort((x,y)=>x-y);const n=a.length;return n%2?a[n>>1]:(a[n/2-1]+a[n/2])/2;};
const percentile=(a,p)=>[...a].sort((x,y)=>x-y)[Math.ceil(a.length*p)-1];
assert.equal(runs.length,9);
const rows=[];
for(const engine of ['server','tfjs','jax-js'])for(const kind of ['ar','vae','flow','gan']){
  const r=runs.filter(r=>r.engine===engine);assert.equal(r.length,3);assert.deepEqual(r.map(r=>r.round).sort(),[1,2,3]);
  const c=r.map(r=>{assert.equal(r.visibility,'visible');assert.equal(r.checks.length,4);assert.equal(r.warmMs,2000);assert.equal(r.measureMs,6000);return r.checks.find(c=>c.kind===kind);});
  c.forEach(c=>{assert.ok(c.updatesPerSec>0&&c.renderedViews>0&&c.snapshotCount>0);assert.equal(c.snapshotIntervalMs,500);});
  rows.push({engine,kind,updatesPerSec:median(c.map(c=>c.updatesPerSec)),updatesRange:c.map(c=>c.updatesPerSec).sort((a,b)=>a-b),rafFps:median(c.map(c=>c.rafFps)),frameP95Ms:median(c.map(c=>percentile(c.frameIntervalsMs,.95))),snapshotMeanMs:median(c.map(c=>c.snapshotMeanMs)),viewRenderMedianMs:median(c.flatMap(c=>c.viewRenderMs)),canvasMedianMs:median(c.flatMap(c=>c.canvasDrawMs)),queryMedianMs:median(c.flatMap(c=>c.queryMs)),coldQueryMedianMs:median(c.map(c=>c.coldQueryMs)),inspectionMedianMs:kind==='vae'?median(c.flatMap(c=>c.inspectMs)):undefined,snapshotJsonBytes:median(c.map(c=>c.snapshotJsonBytes)),renderedViews:c.map(c=>c.renderedViews)});
}
const result={date:'2026-10-13',protocol:'Three counterbalanced runtime rounds; 2s warm-up and 6s measurement per family; actual Canvas views, 500ms polling/snapshot target; paused query timings.',rows};
writeFileSync(new URL('gui-summary.json',base),JSON.stringify(result,null,2));console.table(rows.map(r=>({engine:r.engine,family:r.kind,updates:r.updatesPerSec.toFixed(1),fps:r.rafFps.toFixed(1),snapshot:r.snapshotMeanMs.toFixed(1),render:r.viewRenderMedianMs.toFixed(1),query:r.queryMedianMs.toFixed(1),inspect:r.inspectionMedianMs?.toFixed(1)})));
