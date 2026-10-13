/** Small Canvas renderer for Tree's existing trace descriptions. No model mathematics here. */
import {INK} from './colors';
function bounds(traces:any[],key:string):[number,number]{const a=traces.flatMap(t=>t.type==='heatmap'?(t[key]??[]):t[key]??[]).filter((x:any)=>typeof x==='number'&&Number.isFinite(x));if(!a.length)return [0,1];let lo=Math.min(...a),hi=Math.max(...a);if(lo===hi){lo-=1;hi+=1;}const pad=(hi-lo)*0.06;return [lo-pad,hi+pad];}
function react(el:HTMLElement,traces:any[],layout:any,_config?:unknown):Promise<void>{
  const start=performance.now(),w=layout.width??400,h=layout.height??250,m=layout.margin??{l:46,r:12,t:8,b:40},ratio=devicePixelRatio||1;
  let c=el.querySelector('canvas');if(!c){el.replaceChildren();c=document.createElement('canvas');c.setAttribute('role','img');el.append(c);}
  c.width=Math.round(w*ratio);c.height=Math.round(h*ratio);c.style.width=`${w}px`;c.style.height=`${h}px`;c.setAttribute('aria-label',`${layout.xaxis?.title?.text??'x'} versus ${layout.yaxis?.title?.text??'y'}`);
  const ctx=c.getContext('2d')!;ctx.scale(ratio,ratio);ctx.fillStyle=layout.paper_bgcolor??'#fff';ctx.fillRect(0,0,w,h);
  const xr=layout.xaxis?.range??bounds(traces,'x'),yr=layout.yaxis?.range??bounds(traces,'y'),pw=w-m.l-m.r,ph=h-m.t-m.b;
  const X=(x:number)=>m.l+(x-xr[0])*pw/(xr[1]-xr[0]),Y=(y:number)=>m.t+ph-(y-yr[0])*ph/(yr[1]-yr[0]);
  (el as any)._fullLayout={xaxis:{_offset:m.l,_length:pw,p2d:(x:number)=>xr[0]+x*(xr[1]-xr[0])/pw},yaxis:{_offset:m.t,_length:ph,p2d:(y:number)=>yr[1]-y*(yr[1]-yr[0])/ph}};
  ctx.save();ctx.beginPath();ctx.rect(m.l,m.t,pw,ph);ctx.clip();
  for(const t of traces)if(t.type==='heatmap'){
    const rows=t.z?.length??0,cols=t.z?.[0]?.length??0,values=t.z.flat().filter(Number.isFinite),min=t.zmin??Math.min(...values),max=t.zmax??Math.max(...values,min+1e-9);
    const image=new OffscreenCanvas(cols,rows),ic=image.getContext('2d')!,pixels=ic.createImageData(cols,rows);
    const stops=t.colorscale??[[0,'#fff'],[1,'#1976b9']];
    const rgb=(s:string)=>{if(s.startsWith('#')){const v=s.slice(1);return v.length===3?v.split('').map(x=>parseInt(x+x,16)):[0,2,4].map(i=>parseInt(v.slice(i,i+2),16));}return (s.match(/[\d.]+/g)??['0','0','0']).slice(0,3).map(Number);};
    for(let j=0;j<rows;j++)for(let i=0;i<cols;i++){const u=Math.max(0,Math.min(1,(t.z[j][i]-min)/(max-min)));let k=1;while(k<stops.length-1&&u>stops[k][0])k++;const a=stops[k-1],b=stops[k],v=(u-a[0])/(b[0]-a[0]),ra=rgb(a[1]),rb=rgb(b[1]),q=((rows-1-j)*cols+i)*4;for(let d=0;d<3;d++)pixels.data[q+d]=ra[d]+v*(rb[d]-ra[d]);pixels.data[q+3]=255;}
    ic.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=true;
    ctx.drawImage(image,X(t.x[0]),Y(t.y[rows-1]),X(t.x[cols-1])-X(t.x[0]),Y(t.y[0])-Y(t.y[rows-1]));
  }
  ctx.strokeStyle=INK.grid;ctx.lineWidth=0.6;
  for(let i=0;i<=6;i++){const x=m.l+i*pw/6,y=m.t+i*ph/6;ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,m.t+ph);ctx.moveTo(m.l,y);ctx.lineTo(m.l+pw,y);ctx.stroke();}
  const color=(v:any,i:number,fallback:string)=>Array.isArray(v)?v[i]:v??fallback;
  for(const t of traces)if(t.type!=='heatmap'){
    const xs=t.x??[],ys=t.y??[],mode=t.mode??'lines';ctx.globalAlpha=t.opacity??1;
    if(mode.includes('lines')||t.fill){ctx.beginPath();let open=false;for(let i=0;i<xs.length;i++){if(xs[i]===null||ys[i]===null||!Number.isFinite(xs[i])||!Number.isFinite(ys[i])){open=false;continue;}if(open)ctx.lineTo(X(xs[i]),Y(ys[i]));else ctx.moveTo(X(xs[i]),Y(ys[i]));open=true;}
      if(t.fill==='tozeroy'){ctx.lineTo(X(xs.at(-1)),Y(0));ctx.lineTo(X(xs[0]),Y(0));ctx.closePath();}if(t.fill==='tozerox'){ctx.lineTo(X(0),Y(ys.at(-1)));ctx.lineTo(X(0),Y(ys[0]));ctx.closePath();}
      if(t.fill){ctx.fillStyle=t.fillcolor??'rgba(42,120,214,.12)';ctx.fill();}ctx.strokeStyle=t.line?.color??INK.ink;ctx.lineWidth=t.line?.width??1.3;ctx.setLineDash(t.line?.dash?[3,3]:[]);ctx.stroke();ctx.setLineDash([]);
    }
    if(mode.includes('markers'))for(let i=0;i<xs.length;i++){if(!Number.isFinite(xs[i])||!Number.isFinite(ys[i]))continue;ctx.globalAlpha=(t.opacity??1)*(t.marker?.opacity??1);ctx.beginPath();ctx.arc(X(xs[i]),Y(ys[i]),Number(color(t.marker?.size,i,'4'))/2,0,2*Math.PI);ctx.fillStyle=color(t.marker?.color,i,INK.ink);ctx.fill();if(t.marker?.line){ctx.strokeStyle=color(t.marker.line.color,i,'#fff');ctx.lineWidth=t.marker.line.width??1;ctx.stroke();}}
  }
  ctx.globalAlpha=1;
  for(const s of layout.shapes??[]){ctx.strokeStyle=s.line?.color??INK.ink;ctx.lineWidth=s.line?.width??1;ctx.setLineDash(s.line?.dash?[3,3]:[]);ctx.beginPath();ctx.moveTo(X(s.x0),Y(s.y0));ctx.lineTo(X(s.x1),Y(s.y1));ctx.stroke();ctx.setLineDash([]);}
  ctx.restore();ctx.fillStyle=INK.ink2;ctx.font='11px Segoe UI, sans-serif';
  for(let i=0;i<=6;i++){ctx.textAlign='center';ctx.fillText((xr[0]+i*(xr[1]-xr[0])/6).toFixed(1),m.l+i*pw/6,m.t+ph+16);ctx.textAlign='right';ctx.fillText((yr[1]-i*(yr[1]-yr[0])/6).toFixed(1),m.l-6,m.t+i*ph/6+4);}
  ctx.textAlign='center';ctx.fillText(layout.xaxis?.title?.text??'',m.l+pw/2,h-5);ctx.save();ctx.translate(12,m.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(layout.yaxis?.title?.text??'',0,0);ctx.restore();
  if(layout.showlegend){ctx.font='11px Segoe UI, sans-serif';ctx.textAlign='left';let x=m.l;for(const t of traces.filter(t=>t.name)){ctx.fillStyle=t.line?.color??INK.ink;ctx.fillText(t.name,x,13);x+=ctx.measureText(t.name).width+15;}}
  el.dataset.renderMs=(performance.now()-start).toFixed(2);
  return Promise.resolve();
}
export default {react};
