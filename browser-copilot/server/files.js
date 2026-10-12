import {resolve,relative,dirname,extname,basename} from 'node:path';
import {mkdirSync,lstatSync,readdirSync,readFileSync,writeFileSync,renameSync,existsSync,realpathSync,unlinkSync} from 'node:fs';
import {randomUUID,createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const hash=b=>createHash('sha256').update(b).digest('hex');
const TEXT=new Set(['.txt','.md','.csv','.json','.html','.xml','.yaml','.yml','.log']);
export function createFileTools(folder){
 mkdirSync(folder,{recursive:true,mode:0o700});const root=realpathSync(folder),pending=new Map();
 function path(name){
  if(typeof name!=='string'||!name||name.length>240||name.includes('\\')||name.startsWith('/')||name.split('/').some(p=>!p||p==='..'||p.startsWith('.'))||/\.(pem|key|p12)$/i.test(name))throw Error('Use a relative, non-hidden workspace path');
  const full=resolve(root,name);if(relative(root,full).startsWith('..'))throw Error('Path escapes workspace');
  let current=root;for(const part of name.split('/')){current=resolve(current,part);if(existsSync(current)&&lstatSync(current).isSymbolicLink())throw Error('Symlinks are not allowed');}
  return full;
 }
 function digest(name){const full=path(name);if(existsSync(full)&&lstatSync(full).size>20000000)throw Error('File exceeds the 20 MB operation limit');return existsSync(full)?hash(readFileSync(full)):'absent';}
 function list(){const entries=[];function walk(dir,prefix='',depth=0){if(depth>4)return;for(const e of readdirSync(dir,{withFileTypes:true})){if(entries.length>=100)return;if(e.name.startsWith('.')||e.isSymbolicLink())continue;const name=prefix+e.name;if(e.isDirectory())walk(resolve(dir,e.name),name+'/',depth+1);else if(e.isFile())entries.push({path:name,bytes:lstatSync(resolve(dir,e.name)).size});}}walk(root);return {root,files:entries};}
 function read(name){const full=path(name);if(!TEXT.has(extname(full).toLowerCase()))throw Error('Only text, Markdown, CSV, JSON, HTML, XML and YAML reads are supported');if(lstatSync(full).size>200000)throw Error('File exceeds 200 KB; split it first');return {path:name,text:readFileSync(full,'utf8'),sha256:digest(name)};}
 function preview(action){
  if(!['write_file','rename_file','create_artifact'].includes(action.action))throw Error('Unsupported file operation');
  path(action.target);if(typeof action.value!=='string'||action.value.length>100000)throw Error('Invalid file contents');
  if(action.action==='write_file'&&!TEXT.has(extname(action.target).toLowerCase()))throw Error('Use a supported text extension');
  if(action.action==='rename_file'){path(action.value);if(!existsSync(path(action.target)))throw Error('Source does not exist');if(existsSync(path(action.value)))throw Error('Destination already exists');}
  if(action.action==='create_artifact'){
   if(!['.xlsx','.docx','.pptx'].includes(extname(action.target).toLowerCase()))throw Error('Artifact must be xlsx, docx or pptx');
   const spec=JSON.parse(action.value);if(typeof spec.title!=='string'||spec.title.length>200)throw Error('Artifact needs a short title');
  }
  if(pending.size>100)pending.clear();const ticket=randomUUID();const item={action:structuredClone(action),before:digest(action.target),expires:Date.now()+300000};pending.set(ticket,item);
  return {ticket,path:action.target,operation:action.action,willOverwrite:item.before!=='absent',preview:action.value,expiresAt:item.expires};
 }
 async function commit(ticket){
  const item=pending.get(ticket);pending.delete(ticket);if(!item||item.expires<Date.now())throw Error('Approval expired or already consumed');
  const {action,before}=item;if(digest(action.target)!==before)throw Error('File changed since preview');const full=path(action.target);mkdirSync(dirname(full),{recursive:true});path(action.target);
  if(action.action==='rename_file'){const destination=path(action.value);if(existsSync(destination))throw Error('Destination already exists');mkdirSync(dirname(destination),{recursive:true});path(action.value);renameSync(full,destination);return {path:action.value,sha256:digest(action.value)};}
  const temporary=resolve(dirname(full),'.atlas-'+randomUUID()+extname(full));
  try{
   if(action.action==='write_file')writeFileSync(temporary,action.value,{flag:'wx',mode:0o600});
   else await new Promise((res,rej)=>{const child=spawn(process.env.ATLAS_PYTHON||'python3',[fileURLToPath(new URL('./artifacts.py',import.meta.url)),temporary],{stdio:['pipe','ignore','pipe'],timeout:30000});let error='';child.stderr.on('data',b=>error+=b.toString().slice(0,2000));child.on('error',rej);child.on('close',code=>code===0?res():rej(Error('Artifact renderer failed: '+error)));child.stdin.end(action.value);});
   if(digest(action.target)!==before)throw Error('File changed while generating output');renameSync(temporary,full);return {path:action.target,bytes:lstatSync(full).size,sha256:digest(action.target)};
  }finally{if(existsSync(temporary))unlinkSync(temporary);}
 }
 return {list,read,preview,commit};
}
