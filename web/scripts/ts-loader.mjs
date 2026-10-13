import {registerHooks,stripTypeScriptTypes,createRequire} from 'node:module';
import {existsSync,readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
// TF.js's unbundled Node platform uses require('util') even in its ESM sources.
globalThis.require ??= createRequire(import.meta.url);
registerHooks({
  resolve(specifier,context,next){
    if(['@tensorflow/tfjs-core','@tensorflow/tfjs-backend-cpu','@tensorflow/tfjs-backend-wasm'].includes(specifier))return {url:new URL(`../node_modules/${specifier}/dist/index.js`,import.meta.url).href,shortCircuit:true};
    if(specifier.startsWith('.')&&context.parentURL){const u=new URL(specifier,context.parentURL);if(!u.pathname.endsWith('.ts')&&existsSync(fileURLToPath(u)+'.ts'))return {url:u.href+'.ts',shortCircuit:true};if(context.parentURL.includes('/@tensorflow/')&&!u.pathname.endsWith('.js')&&existsSync(fileURLToPath(u)+'.js'))return {url:u.href+'.js',shortCircuit:true};}
    return next(specifier,context);
  },
  load(url,context,next){if(url.endsWith('.ts'))return {format:'module',source:stripTypeScriptTypes(readFileSync(new URL(url),'utf8'),{mode:'transform'}),shortCircuit:true};if(url.includes('/@tensorflow/')&&url.endsWith('.js')&&!url.endsWith('.node.js'))return {format:'module',source:readFileSync(new URL(url),'utf8'),shortCircuit:true};return next(url,context);},
});
