import {existsSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
const checks=[];
const add=(name,ok,detail)=>checks.push({name,ok,detail});
add('Node',Number(process.versions.node.split('.')[0])>=24,process.version);
let chromium;
try{const require=createRequire(import.meta.url);({chromium}=require(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES?process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES+'/playwright':'playwright'));add('Playwright',true,'Installed');}catch{add('Playwright',false,'Run npm install --no-save playwright');}
if(chromium){const path=process.env.ATLAS_CHROMIUM_PATH||chromium.executablePath();add('Chromium binary',existsSync(path),path);}
try{execFileSync(process.env.ATLAS_PYTHON||'python3',['-c','import docx,openpyxl,pptx'],{stdio:'pipe'});add('Office libraries',true,'Installed');}catch{add('Office libraries',false,'Install requirements-artifacts.txt in your Python environment');}
try{const endpoint=process.env.OLLAMA_URL||'http://127.0.0.1:11434';const r=await fetch(endpoint+'/api/tags',{signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error('HTTP '+r.status);const data=await r.json();const model=process.env.ATLAS_MODEL||'qwen3.5:0.8b';add('Ollama model',data.models?.some(m=>m.name===model)||false,model);}catch(e){add('Ollama model',false,'Not reachable: '+e.message);}
console.log(JSON.stringify({ready:checks.every(c=>c.ok),checks,accuracy:'NOT MEASURED by environment diagnostics'},null,2));if(checks.some(c=>!c.ok))process.exitCode=1;
