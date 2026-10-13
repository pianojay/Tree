import {copyFileSync,mkdirSync,readdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const from=root+'node_modules/@tensorflow/tfjs-backend-wasm/dist/';
const into=root+'public/wasm/';mkdirSync(into,{recursive:true});
for(const f of readdirSync(from))if(f.endsWith('.wasm'))copyFileSync(from+f,into+f);
