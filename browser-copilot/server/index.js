import http from 'node:http';
import {randomBytes,timingSafeEqual} from 'node:crypto';
import {mkdirSync,existsSync,readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {DatabaseSync} from 'node:sqlite';
import {renderDashboard} from './dashboard.js';
import {summarizeCSV} from './csv.js';
import {createFileTools} from './files.js';
import {plan,skills,makePlan} from './planner.js';
export function createApp({dataDir=resolve('.data'),token,model=process.env.ATLAS_MODEL||'qwen3.5:0.8b',ollama=process.env.OLLAMA_URL||'http://127.0.0.1:11434',extensionId=process.env.ATLAS_EXTENSION_ID||''}={}) {
 const endpoint=new URL(ollama);
 if(!['localhost','127.0.0.1','[::1]'].includes(endpoint.hostname)||endpoint.protocol!=='http:') throw Error('Ollama must be a loopback HTTP endpoint');
 mkdirSync(dataDir,{recursive:true,mode:0o700});
 const tokenPath=resolve(dataDir,'token');
 token ||= existsSync(tokenPath)?readFileSync(tokenPath,'utf8').trim():randomBytes(32).toString('hex');
 if(!existsSync(tokenPath)) writeFileSync(tokenPath,token,{mode:0o600});
 const db=new DatabaseSync(resolve(dataDir,'atlas.sqlite'));
 db.exec('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS state (id TEXT PRIMARY KEY, version INTEGER NOT NULL, body TEXT NOT NULL);');
 const files=createFileTools(process.env.ATLAS_WORKSPACE||resolve(dataDir,'workspace'));
 let busy=false;
 const server=http.createServer(async(req,res)=>{
  const origin=req.headers.origin||'';
  const allowed=extensionId ? origin==='chrome-extension://'+extensionId : /^chrome-extension:\/\/[a-p]{32}$/.test(origin);
  const send=(status,body)=>{res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store','X-Content-Type-Options':'nosniff',...(allowed?{'Access-Control-Allow-Origin':origin,'Vary':'Origin'}:{})});res.end(JSON.stringify(body));};
  if(!/^((127\.0\.0\.1|localhost)(:\d+)?|\[::1\](:\d+)?)$/.test(req.headers.host||'')) return send(403,{error:'Invalid Host'});
  if(origin&&!allowed) return send(403,{error:'Origin not allowed'});
  if(req.method==='OPTIONS') {if(!allowed)return send(403,{error:'Origin not allowed'});res.writeHead(204,{'Access-Control-Allow-Origin':origin,'Access-Control-Allow-Methods':'GET,PUT,POST,DELETE,OPTIONS','Access-Control-Allow-Headers':'Authorization,Content-Type','Vary':'Origin'});return res.end();}
  if(req.url==='/health'&&req.method==='GET')return send(200,{ok:true,service:'atlas-local'});
  const supplied=Buffer.from((req.headers.authorization||'').replace(/^Bearer /,''));const expected=Buffer.from(token);
  if(supplied.length!==expected.length||!timingSafeEqual(supplied,expected))return send(401,{error:'Pairing token required'});
  try {
   let body={};
   if(['POST','PUT'].includes(req.method)) {let raw='';for await(const chunk of req){raw+=chunk;if(Buffer.byteLength(raw)>1500000)throw Error('Request too large');}body=JSON.parse(raw||'{}');}
   if(req.url==='/v1/info'&&req.method==='GET') {
    const r=await fetch(ollama+'/api/tags',{signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error('Cannot list Ollama models');const result=await r.json();
    return send(200,{workspace:files.list().root,model,installed:(result.models||[]).map(m=>m.name),skills});
   }
   if(req.url==='/v1/files'&&req.method==='GET')return send(200,files.list());
   if(req.url==='/v1/files/analyze-csv'&&req.method==='POST'){const file=files.read(body.path);return send(200,{path:file.path,sha256:file.sha256,summary:summarizeCSV(file.text)});}
   if(req.url==='/v1/files/verify'&&req.method==='POST')return send(200,{files:files.verify(body.files)});
   if(req.url==='/v1/files/dashboard-preview'&&req.method==='POST'){if(!body.target?.endsWith('.html'))throw Error('Dashboard output must end in .html');const source=files.read(body.source);return send(200,files.preview({action:'write_file',target:body.target,value:renderDashboard(source.text,source.path)}));}
   if(req.url==='/v1/files/read'&&req.method==='POST')return send(200,files.read(body.path));
   if(req.url==='/v1/files/preview'&&req.method==='POST')return send(200,files.preview(body.action));
   if(req.url==='/v1/files/commit'&&req.method==='POST')return send(200,await files.commit(body.ticket));
   if(req.url==='/v1/plan'&&req.method==='POST'){if(busy)return send(429,{error:'Model is busy'});busy=true;try{return send(200,await makePlan(body,{model,ollama}));}finally{busy=false;}}
   if(req.url==='/v1/step'&&req.method==='POST') {
    if(busy)return send(429,{error:'Another model request is running; try again shortly'});
    busy=true;try{return send(200,await plan(body,{model,ollama}));}finally{busy=false;}
   }
   if(req.url==='/v1/state'&&req.method==='GET') {const row=db.prepare('SELECT version,body FROM state WHERE id=?').get('workspace');return send(200,row?{version:row.version,state:JSON.parse(row.body)}:{version:0,state:{sessions:[]}});}
   if(req.url==='/v1/state'&&req.method==='PUT') {
    if(!Number.isInteger(body.version)||!Array.isArray(body.state?.sessions)||body.state.sessions.length>100)throw Error('Invalid workspace');
    db.exec('BEGIN IMMEDIATE');
    try {const row=db.prepare('SELECT version FROM state WHERE id=?').get('workspace');const version=row?.version||0;
     if(version!==body.version){db.exec('ROLLBACK');return send(409,{error:'Workspace changed in another browser. Refresh before saving.'});}
     db.prepare('INSERT INTO state VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET version=excluded.version,body=excluded.body').run('workspace',version+1,JSON.stringify(body.state));db.exec('COMMIT');return send(200,{version:version+1});
    }catch(e){try{db.exec('ROLLBACK')}catch{}throw e;}
   }
   if(req.url==='/v1/state'&&req.method==='DELETE'){db.prepare("INSERT INTO state VALUES ('workspace',1,'{\"sessions\":[]}') ON CONFLICT(id) DO UPDATE SET version=version+1,body=excluded.body").run();return send(200,{ok:true});}
   send(404,{error:'Not found'});
  }catch(e){send(400,{error:e.message});}
 });
 server.on('close',()=>db.close());
 return {server,token};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
 const {server,token}=createApp();const port=Number(process.env.PORT||8765);
 server.listen(port,'127.0.0.1',()=>console.log(`Atlas listening on http://127.0.0.1:${port}\nPairing token (paste in extension Settings): ${token}\nModel: ${process.env.ATLAS_MODEL||'qwen3.5:0.8b'}\nKeep this token private. Set ATLAS_EXTENSION_ID to restrict pairing to your extension.`));
}
