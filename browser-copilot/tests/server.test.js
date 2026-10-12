import http from 'node:http';import test from 'node:test';import assert from 'node:assert/strict';import {mkdtempSync,rmSync} from 'node:fs';import {tmpdir} from 'node:os';import {join} from 'node:path';import {createApp} from '../server/index.js';
test('API authentication, CORS, durable history and optimistic concurrency',async()=>{const dir=mkdtempSync(join(tmpdir(),'atlas-'));const {server}=createApp({dataDir:dir,token:'test-token'});await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;const headers={Authorization:'Bearer test-token','Content-Type':'application/json'};
 try{
  assert.equal((await fetch(base+'/health')).status,200);
  assert.equal((await fetch(base+'/v1/state')).status,401);
  assert.equal((await fetch(base+'/v1/state',{headers:{...headers,Origin:'https://evil.test'}})).status,403);
  assert.equal(await new Promise((resolve,reject)=>{const req=http.get(base+'/v1/state',{headers:{...headers,Host:'evil.test'}},res=>{res.resume();resolve(res.statusCode)});req.on('error',reject)}),403);
  const preflight=await fetch(base+'/v1/state',{method:'OPTIONS',headers:{Origin:'chrome-extension://'+'a'.repeat(32)}});assert.equal(preflight.status,204);
  const post=async(path,data)=>{const response=await fetch(base+path,{method:'POST',headers,body:JSON.stringify(data)});assert.equal(response.status,200);return response.json();};
  const source=await post('/v1/files/preview',{action:{action:'write_file',target:'sales.csv',value:'month,revenue\nJune,20\nJuly,22'}});await post('/v1/files/commit',{ticket:source.ticket});
  const dashboard=await post('/v1/files/dashboard-preview',{source:'sales.csv',target:'dashboard.html'});assert.equal(dashboard.willOverwrite,false);assert.ok(dashboard.preview.includes('42'));
  const output=await post('/v1/files/commit',{ticket:dashboard.ticket});const checked=await post('/v1/files/verify',{files:[output]});assert.equal(checked.files[0].verified,true);
  let r=await fetch(base+'/v1/state',{headers});assert.deepEqual(await r.json(),{version:0,state:{sessions:[]}});
  const body=JSON.stringify({version:0,state:{sessions:[{id:'one',title:'Research'}]}});
  r=await fetch(base+'/v1/state',{method:'PUT',headers,body});assert.equal((await r.json()).version,1);
  assert.equal((await fetch(base+'/v1/state',{method:'PUT',headers,body})).status,409);
  r=await fetch(base+'/v1/state',{headers});assert.equal((await r.json()).state.sessions[0].id,'one');
  await fetch(base+'/v1/state',{method:'DELETE',headers});r=await fetch(base+'/v1/state',{headers});assert.deepEqual(await r.json(),{version:2,state:{sessions:[]}});
  assert.equal((await fetch(base+'/v1/state',{method:'PUT',headers,body})).status,409);
 }finally{await new Promise(r=>server.close(r));rmSync(dir,{recursive:true,force:true});}
});
test('remote model endpoints are rejected',()=>assert.throws(()=>createApp({ollama:'https://cloud.example.com'}),/loopback/));
