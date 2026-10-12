// Uses only a local fictional CRM. Never attach this test to a real website/session.
import {createRequire} from 'node:module';
import {mkdtempSync,cpSync,readFileSync,writeFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';import {join,resolve} from 'node:path';
import http from 'node:http';import assert from 'node:assert/strict';
import {createApp} from '../server/index.js';import {createCRM} from './fixtures/crm.js';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES?process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES+'/playwright':'playwright');
const live=process.env.ATLAS_LIVE_MODEL==='1';
const expected='Hello Ada, Atlas Local helps operations teams create reviewed reports from documents and browser pages. Would a short demonstration be useful?';
const temp=mkdtempSync(join(tmpdir(),'atlas-outreach-'));let context;let apiServer;
const report={mode:live?'live-qwen':'scripted-model',passed:false,realContacts:0,startedAt:new Date().toISOString()};
const {server:crm,messages}=createCRM();
const model=http.createServer(async(req,res)=>{res.setHeader('Content-Type','application/json');try{
 if(live){const response=await fetch((process.env.OLLAMA_URL||'http://127.0.0.1:11434')+req.url,{method:req.method,...(req.method==='POST'?{body:await new Promise(async resolve=>{let raw='';for await(const c of req)raw+=c;resolve(raw);}),headers:{'Content-Type':'application/json'}}:{}),signal:AbortSignal.timeout(120000)});res.statusCode=response.status;return res.end(await response.text());}
 if(req.url==='/api/tags')return res.end(JSON.stringify({models:[{name:'qwen3.5:0.8b'}]}));
 let raw='';for await(const c of req)raw+=c;const request=JSON.parse(raw);
 if(request.format.properties.steps)return res.end(JSON.stringify({message:{content:JSON.stringify({steps:['Open the matching Ada Example profile','Draft the specified message','Review recipient and exact text before one send attempt']})}}));
 const payload=JSON.parse(request.messages[1].content),o=payload.observation;
 const act=(action,target='',value='')=>({action,target,value,reason:'Follow the controlled outreach scenario'});
 let action;
 if(o.url.endsWith('/leads'))action=act('navigate','',o.elements.find(e=>e.label.includes('View Ada')).href);
 else if(o.elements.some(e=>e.label==='Message'))action=act('click',o.elements.find(e=>e.label==='Message').id);
 else if(payload.recentActions.at(-1)?.action==='type')action=act('click',o.elements.find(e=>e.label==='Send message').id);
 else action=act('type',o.elements.find(e=>e.tag==='textarea').id,expected);
 res.end(JSON.stringify({message:{content:JSON.stringify(action)}}));
 }catch(e){res.statusCode=500;res.end(JSON.stringify({error:e.message}));}});
try{
 await new Promise(r=>crm.listen(0,'127.0.0.1',r));await new Promise(r=>model.listen(0,'127.0.0.1',r));
 const app=createApp({dataDir:join(temp,'data'),token:'fixture-only-token',ollama:`http://127.0.0.1:${model.address().port}`});apiServer=app.server;await new Promise((r,reject)=>{apiServer.once('error',reject);apiServer.listen(8765,'127.0.0.1',r);});
 const extension=join(temp,'extension');cpSync(resolve('extension'),extension,{recursive:true});const manifest=JSON.parse(readFileSync(join(extension,'manifest.json')));manifest.host_permissions.push('http://127.0.0.1/*');writeFileSync(join(extension,'manifest.json'),JSON.stringify(manifest));
 context=await chromium.launchPersistentContext(join(temp,'profile'),{headless:process.env.ATLAS_HEADFUL!=='1',...(process.env.ATLAS_CHROMIUM_PATH?{executablePath:process.env.ATLAS_CHROMIUM_PATH}:{channel:'chromium'}),args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`]});
 const worker=context.serviceWorkers()[0]||await context.waitForEvent('serviceworker');const id=new URL(worker.url()).host;
 const lead=await context.newPage();await lead.goto(`http://127.0.0.1:${crm.address().port}/leads`);
 const panel=await context.newPage();await panel.goto(`chrome-extension://${id}/panel.html?workspace=1`);
 await panel.locator('#token').fill('fixture-only-token');await panel.locator('#connect').click();await panel.waitForFunction(()=>document.getElementById('connection').textContent.includes('Connected'));
 await panel.locator('#settingsToggle').click();await panel.locator('#productBrief').fill('Atlas Local creates reviewed reports from documents and browser pages. No quantified performance claims supplied.');await panel.locator('#leadCriteria').fill('Operations Director at a UK software company with 10–50 employees');await panel.locator('#savePreferences').click();await panel.locator('#settingsToggle').click();
 await panel.locator('#tabs input').first().check();await panel.locator('#skill').selectOption('leads');await panel.locator('#prompt').fill(`Find the matching lead Ada Example, open her profile, click Message, fill the composer with exactly: ${expected} Then send once after review.`);await panel.locator('#run').click();
 const deadline=Date.now()+600000;let approvedPlan=false;
 while(Date.now()<deadline){
  if(!approvedPlan&&await panel.locator('#planReview').isVisible()){await panel.locator('#acceptPlan').click();approvedPlan=true;}
  if(await panel.locator('#approval').isVisible()){
   if(await panel.locator('#sendReview').isVisible()){
    assert.equal(messages.length,0,'No message may be sent before final review');
    assert.ok((await panel.locator('#approvalDetails').textContent()).includes(expected));
    await panel.locator('#recipientName').fill('Ada Example');await panel.locator('#recipientUrl').fill(lead.url());
   }
   await panel.locator('#reviewed').check();await panel.locator('#approve').click();
  }
  if(await panel.locator('#deliveryReview').isVisible()){
   await lead.waitForFunction(()=>document.getElementById('thread').textContent.includes('Message recorded'));
   assert.equal(messages.length,1);assert.equal(messages[0].recipient,'ada');assert.equal(messages[0].message,expected);
   await panel.locator('#confirmDelivery').click();await panel.waitForFunction(()=>document.getElementById('taskState').textContent==='Send confirmed by user');report.passed=true;report.receiptId=messages[0].id;break;
  }
  const error=await panel.locator('#timeline .event.error').allTextContents();if(error.length)throw Error(error.join('\n'));
  await new Promise(r=>setTimeout(r,200));
 }
 if(!report.passed)throw Error('Scenario did not complete within 10 minutes');
}catch(e){report.error=e.message;process.exitCode=1;}finally{
 report.finishedAt=new Date().toISOString();report.messages=messages.length;console.log(JSON.stringify(report,null,2));await context?.close();for(const s of [apiServer,crm,model])if(s?.listening)await new Promise(r=>s.close(r));rmSync(temp,{recursive:true,force:true});
}
