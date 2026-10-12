import {isSendAction,buildLeadGoal,validateRecipient,chooseWorkflow} from './outreach-policy.js';
import {canDelegate,missingOutputs,parseOutputs} from './delegation.js';
import {safeURL,originPattern,needsApproval,validateProposal} from './policy.js';
const $=id=>document.getElementById(id), BASE='http://127.0.0.1:8765';
let token='',version=0,state={sessions:[]},session=null,tabs=[],selected=new Set(),running=false,stopped=false,approvalResolve=null,controller=null,connected=false,conflict=false,planResolve=null,activeSend=null,saving=0;
function status(s){$('status').textContent=s;}
async function api(path,method='GET',body,signal){const r=await fetch(BASE+path,{method,headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},...(body?{body:JSON.stringify(body)}:{}),signal});const result=await r.json();if(!r.ok){if(r.status===409)conflict=true;throw Error(result.error||'Backend unavailable');}return result;}
async function persist(){
 saving++;
 try{
  await chrome.storage.local.set({workspace:state});
  if(conflict)throw Error('History changed elsewhere. Reconnect before continuing.');
  if(connected){const r=await api('/v1/state','PUT',{version,state});version=r.version;}
 }finally{saving--;}
}
async function syncHistory(silent=false){
 if(!connected||running||saving)return;
 const remote=await api('/v1/state');
 if(running||saving)return;
 if(remote.version>version){const currentId=session?.id;state=remote.state;version=remote.version;session=state.sessions.find(s=>s.id===currentId)||state.sessions[0]||null;await chrome.storage.local.set({workspace:state});render();}
 if(!silent)status('History is up to date.');
}
async function activePage(){
 const active=(await chrome.tabs.query({active:true,currentWindow:true}))[0];
 $('currentPage').textContent=active?.title||'Open a webpage to begin';$('currentPage').title=active?.url||'';
 if(!running&&$('followTab').checked&&/^https?:/.test(active?.url||''))selected=new Set([active.id]);
 await refreshTabs();
}
function scopeUI(){$('fileSection').hidden=$('scope').value==='browser';$('outputOptions').hidden=$('scope').value==='browser';}

function renderSessions(){const el=$('sessions');el.replaceChildren();if(!state.sessions.length){el.add(new Option('No saved sessions',''));return;}for(const s of state.sessions)el.add(new Option(s.title,s.id));el.value=session?.id||state.sessions[0].id;}
function render(){$('deliveryReview').hidden=!session?.awaitingDelivery;$('taskState').textContent=session?.status||'Ready';const outputs=session?.deliverables||[];$('deliverablesSection').hidden=!outputs.length;$('deliverables').replaceChildren();for(const output of outputs){const row=document.createElement('div');row.className='deliverable';row.textContent=output.path+' · '+(output.verified?'File verified':'Saved; verification pending');$('deliverables').append(row);}renderSessions();$('welcome').hidden=!!session?.events.length;$('timeline').replaceChildren();for(const e of session?.events||[]){const item=document.createElement('div');item.className='event '+e.kind;const kind=document.createElement('div');kind.className='kind';kind.textContent=e.kind+' · '+new Date(e.at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});const body=document.createElement('div');body.className='body';body.textContent=e.text;item.append(kind,body);$('timeline').append(item);}const sources=session?.evidence||[];$('evidenceSection').hidden=!sources.length;$('sources').replaceChildren();for(const [i,e]of sources.entries()){if(e.url.startsWith('atlas://')){const entry=document.createElement('p');entry.textContent=`${i+1}. ${e.title} (workspace file)`;$('sources').append(entry);continue;}const link=document.createElement('a');link.textContent=`${i+1}. ${e.title||e.url}`;try{link.href=safeURL(e.url)}catch{continue;}link.target='_blank';link.rel='noopener noreferrer';$('sources').append(link);}}
async function log(kind,text){session.events.push({kind,text:String(text),at:Date.now()});session.events=session.events.slice(-120);render();await persist();}
function newSession(goal='New task'){session={id:crypto.randomUUID(),title:goal.slice(0,65),goal,events:[],evidence:[],actions:[],createdAt:Date.now()};state.sessions.unshift(session);state.sessions=state.sessions.slice(0,30);render();}
async function refreshTabs(){tabs=(await chrome.tabs.query({currentWindow:true})).filter(t=>/^https?:/.test(t.url||''));if(!selected.size){const active=tabs.find(t=>t.active);if(active)selected.add(active.id);}selected=new Set([...selected].filter(id=>tabs.some(t=>t.id===id)));$('tabs').replaceChildren();for(const t of tabs){const label=document.createElement('label');label.className='tab';const input=document.createElement('input');input.type='checkbox';input.checked=selected.has(t.id);input.disabled=running;input.addEventListener('change',()=>{$('followTab').checked=false;input.checked?selected.add(t.id):selected.delete(t.id);});const name=document.createElement('span');name.textContent=t.title||t.url;name.title=t.url;label.append(input,name);$('tabs').append(label);}}
async function permitted(id){const t=await chrome.tabs.get(id);safeURL(t.url);if(!await chrome.permissions.contains({origins:[originPattern(t.url)]}))throw Error('Grant access to this site using “Allow selected sites”.');return t;}
async function refreshFiles(){const inventory=await api('/v1/files');$('workspacePath').textContent=inventory.root;$('files').textContent=inventory.files.map(f=>`${f.path} · ${f.bytes} bytes`).join('\n')||'Your folder is empty. Add source files here, or ask Atlas to create a report.';return inventory;}
async function observe(id){const o=id===null?{url:'atlas://workspace',title:'Local workspace',text:'',elements:[],fingerprint:'files'}:await snapshot(id);if($('scope').value!=='browser'){const inv=await refreshFiles();o.files=inv.files;o.text+='\nWorkspace files: '+JSON.stringify(inv.files);}return o;}
async function snapshot(id){await permitted(id);await chrome.scripting.executeScript({target:{tabId:id},files:['content.js']});const results=await chrome.scripting.executeScript({target:{tabId:id},func:()=>globalThis.__atlas.snapshot()});if(!results[0]?.result)throw Error('Cannot read this page');return results[0].result;}
function waitForReview(action,o,id){$('sendReview').hidden=!activeSend;if(activeSend){$('recipientName').value='';$('recipientUrl').value='';}$('approval').hidden=false;$('reviewed').checked=false;$('approvalTitle').textContent=action.action==='click'?'Review this click':'Review '+action.action.replace('_',' ');const target=o.elements.find(e=>e.id===action.target);$('approvalDetails').textContent=`Page: ${o.title}\nURL: ${o.url}\nTarget: ${target?.label||action.target||'New page'}\nAction: ${action.action}\nValue: ${action.value||'—'}\nReason: ${action.reason}\n\nApproval applies only to this action on this snapshot.`;$('approval').scrollIntoView({behavior:'smooth',block:'center'});if(id!==null)chrome.tabs.update(id,{active:true}).catch(()=>{});return new Promise(resolve=>{approvalResolve=resolve;});}
function resolveApproval(ok){$('approval').hidden=true;const resolve=approvalResolve;approvalResolve=null;resolve?.(ok);}
async function settled(id){const end=Date.now()+15000;while(Date.now()<end){if(stopped)throw Error('Stopped');const t=await chrome.tabs.get(id);if(t.status==='complete'){await new Promise(r=>setTimeout(r,500));return;}await new Promise(r=>setTimeout(r,250));}throw Error('Page did not finish loading');}
async function execute(action,o,id){
 if(action.action==='analyze_csv'){const result=await api('/v1/files/analyze-csv','POST',{path:action.target});session.evidence.push({url:'atlas://workspace/'+result.path,title:result.path+' (computed summary)',text:JSON.stringify(result.summary),at:Date.now()});session.evidence=session.evidence.slice(-20);await log('analysis',JSON.stringify(result.summary,null,2));return id;}
 if(action.action==='read_file'){const file=await api('/v1/files/read','POST',{path:action.target});session.evidence.push({url:'atlas://workspace/'+file.path,title:file.path,text:file.text,at:Date.now()});session.evidence=session.evidence.slice(-20);await log('source','Read '+file.path+' · SHA256 '+file.sha256);return id;}
 if(['write_file','rename_file','create_artifact','create_dashboard'].includes(action.action)){
  const preview=action.action==='create_dashboard'?await api('/v1/files/dashboard-preview','POST',{target:action.target,source:action.value}):await api('/v1/files/preview','POST',{action});
  const delegated=session.delegateOutputs&&canDelegate(action,preview,session.requiredOutputs);
  if(!delegated&&!await waitForReview({...action,reason:action.reason+(preview.willOverwrite?' · REPLACES EXISTING FILE':'')},o,null)){delete session.pending;await persist();throw Error('File operation cancelled');}
  if(stopped)throw Error('Stopped');const result=await api('/v1/files/commit','POST',{ticket:preview.ticket});
  session.deliverables=(session.deliverables||[]).filter(f=>f.path!==result.path);session.deliverables.push({...result,verified:false});
  await log('file','Saved '+result.path+'\nSHA256 '+result.sha256);await refreshFiles();return id;
 }
 let sendIntent=null;
 if(isSendAction(action,o)){
  const result=await chrome.scripting.executeScript({target:{tabId:id},func:p=>globalThis.__atlas.draft(p),args:[{snapshotId:o.snapshotId,fingerprint:o.fingerprint}]});
  const draft=result[0]?.result;if(!draft?.message)throw Error('Unable to read the exact message for review');
  activeSend={observation:o,draft};
  const accepted=await waitForReview({...action,value:draft.message,reason:'SEND MESSAGE — '+action.reason},o,id);
  const reviewed=activeSend;activeSend=null;
  if(!accepted){delete session.pending;await persist();throw Error('Send cancelled');}
  if(stopped)throw Error('Stopped');
  sendIntent=await api('/v1/outreach/prepare','POST',{...reviewed.recipient,message:draft.message});
 }else if(needsApproval(action)&&!await waitForReview(action,o,id)){delete session.pending;await persist();throw Error('Action cancelled');}
 if(stopped)throw Error('Stopped');
 if(['navigate','open_tab'].includes(action.action)){
  const current=await snapshot(id);if(current.url!==o.url||current.fingerprint!==o.fingerprint)throw Error('Page changed during review; run again to replan.');
  if(!await chrome.permissions.contains({origins:[originPattern(action.value)]}))throw Error('Destination needs site access. Open it manually, then allow the site.');
  if(action.action==='open_tab'){const t=await chrome.tabs.create({url:safeURL(action.value),active:false});selected.add(t.id);await settled(t.id);await refreshTabs();return t.id;}
  await chrome.tabs.update(id,{url:safeURL(action.value)});await settled(id);return id;
 }
 if(action.action==='switch_tab'){const target=Number(action.target);await permitted(target);return target;}
 if(action.action==='extract'){session.evidence.push({url:o.url,title:o.title,text:o.text,at:Date.now()});session.evidence=session.evidence.slice(-20);await log('source',`Captured ${o.title}\n${o.url}`);return id;}
 if(['click','type','select','scroll'].includes(action.action)){
  if(sendIntent){await api('/v1/outreach/claim','POST',{id:sendIntent.id});session.awaitingDelivery={id:sendIntent.id,url:o.url};session.status='Delivery unverified';await persist();}
  const r=await chrome.scripting.executeScript({target:{tabId:id},func:p=>globalThis.__atlas.act(p),args:[{snapshotId:o.snapshotId,fingerprint:o.fingerprint,action}]});
  if(!r[0]?.result?.executed)throw Error('Action was not executed');
  await log('verification',r[0].result.note);await settled(id);return id;
 }
 return id;
}
function setRunning(value){running=value;for(const id of ['run','capture','connect','newSession','sessions','grant','refreshTabs','refreshFiles','savePreferences','settingsToggle','scope','skill','openWorkspace','syncHistory','followTab'])$(id).disabled=value;$('stop').hidden=!value;document.querySelectorAll('#tabs input').forEach(e=>e.disabled=value);}
async function run(){
 if(running)return;if(!connected)throw Error('Connect your backend first.');if(session?.pending)throw Error('An earlier action has an uncertain outcome. Check the page/files, then start a New session to avoid repeating it.');if(session?.awaitingDelivery)throw Error('Resolve the previous send outcome before continuing.');if($('followTab').checked)await activePage();const inputGoal=$('prompt').value.trim();const workflow=chooseWorkflow($('skill').value,inputGoal);const goal=workflow==='leads'?buildLeadGoal(inputGoal,state.preferences?.outreach):inputGoal;if(!goal)throw Error('Describe your goal.');if($('scope').value!=='files'&&!selected.size)throw Error('Select a tab.');
 if(!session)newSession(goal);if(!['Needs your input','Paused','Stopped'].includes(session.status))session.deliverables=[];session.status='Planning';session.goal=goal;session.prompt=inputGoal;session.skill=workflow;session.scope=$('scope').value;session.title=goal.slice(0,65);stopped=false;setRunning(true);
 try{await log('user',goal);status('Building a task plan…');controller=new AbortController();const proposed=await api('/v1/plan','POST',{goal,scope:$('scope').value},controller.signal);controller=null;if(stopped)return;$('planSteps').value=proposed.steps.join('\n');$('requiredOutputs').value=$('scope').value==='browser'?'':(session.requiredOutputs||[]).join('\n');scopeUI();$('delegateOutputs').checked=false;session.status='Awaiting plan review';render();$('planReview').hidden=false;status('Review or edit your plan.');const accepted=await new Promise(r=>{planResolve=r});$('planReview').hidden=true;if(!accepted||stopped)throw Error('Plan cancelled');session.plan=$('planSteps').value.split('\n').map(s=>s.trim()).filter(Boolean).slice(0,8);if(!session.plan.length)throw Error('Plan cannot be empty');session.requiredOutputs=parseOutputs($('requiredOutputs').value);session.delegateOutputs=$('delegateOutputs').checked;session.status='Running';await log('plan',session.plan.map((s,i)=>`${i+1}. ${s}`).join('\n'));let id=$('scope').value==='files'?null:[...selected][0],last='',repeats=0;
  for(let step=0;step<20;step++){
   if(stopped)break;$('stepCount').textContent=`Step ${step+1} / 20`;status('Reading page…');if(id!==null)await settled(id);const o=await observe(id);
   status('Local model is choosing the next step…');controller=new AbortController();
   const result=await api('/v1/step','POST',{goal,skill:session.skill,observation:o,tabs:tabs.filter(t=>selected.has(t.id)).map(t=>({id:t.id,title:t.title,url:t.url})),history:session.actions,evidence:session.evidence,conversation:session.events.filter(e=>['user','answer','question'].includes(e.kind)).slice(-6),instructions:state.preferences?.instructions||'',readOnly:!!state.preferences?.readOnly,taskPlan:session.plan||[],requiredOutputs:session.requiredOutputs||[],delivered:session.deliverables||[],scope:$('scope').value},controller.signal);controller=null;if(stopped)break;
   const a=validateProposal(result.action,o,tabs.filter(t=>selected.has(t.id)),goal);const fileAction=['read_file','analyze_csv','write_file','rename_file','create_artifact','create_dashboard'].includes(a.action);if(($('scope').value==='browser'&&fileAction)||($('scope').value==='files'&&!fileAction&&!['done','ask'].includes(a.action)))throw Error('Action outside selected task scope');if(state.preferences?.readOnly&&!['extract','scroll','switch_tab','read_file','analyze_csv','done','ask'].includes(a.action))throw Error('Action disabled in read-only mode');const key=JSON.stringify(a)+'|'+o.fingerprint;repeats=key===last?repeats+1:0;last=key;if(repeats>=2)throw Error('No progress detected. Narrow the task or navigate manually.');
   if(a.action==='done'){
    const absent=missingOutputs(session.requiredOutputs||[],session.deliverables||[]);
    if(absent.length){await log('verification','Task is not finished. Missing outputs: '+absent.join(', '));session.actions.push({action:'verification',result:'Create these missing files before done: '+absent.join(', ')});if(step>=19)throw Error('Step limit reached with missing deliverables');continue;}
    const checked=await api('/v1/files/verify','POST',{files:session.deliverables||[]});session.deliverables=checked.files;
    if(checked.files.some(f=>!f.verified))throw Error('A delivered file is missing or changed. Review it before completing this task.');
    session.status='Completed';await log('answer',a.value);status('Task finished. Output files checked; review their contents.');return;
   }
   if(a.action==='ask'){session.status='Needs your input';await log('question',a.value);status('Your input is needed.');return;}
   await log('plan',`${a.action.replace('_',' ')} · ${a.reason}`);status(needsApproval(a)?'Waiting for your review…':'Working…');session.pending={action:a,url:o.url,at:Date.now()};await persist();id=await execute(a,o,id);delete session.pending;if(session.awaitingDelivery){session.status='Delivery unverified';await log('verification','A send control was dispatched. Verify the actual conversation; no automatic resend will occur.');status('Check the conversation and record the outcome below.');return;}
   const after=await observe(id);const changed=after.fingerprint!==o.fingerprint;
   session.actions.push({action:a.action,target:a.target,value:a.value.slice(0,500),result:changed?'Page changed; not proof of task success':'No page change',url:after.url});session.actions=session.actions.slice(-30);await persist();
   if(a.action==='click'&&!changed){await log('question','The click produced no observable page change. Check the page before continuing; Atlas will not retry it automatically.');status('Paused for outcome verification.');return;}
  }status(stopped?'Stopped. No more actions will run.':'Step limit reached. Review progress before continuing.');
 }finally{if(['Running','Planning','Awaiting plan review'].includes(session?.status))session.status=stopped?'Stopped':'Paused';render();try{await persist();}catch(error){status(error.message);}controller=null;$('planReview').hidden=true;resolveApproval(false);setRunning(false);$('stepCount').textContent='Ready when you are';}
}
async function capture(){if(running)return;if(!connected)throw Error('Connect first');if(!selected.size)throw Error('Select pages to capture');if(!session)newSession('Multi-tab research');setRunning(true);stopped=false;try{const ids=[...selected];for(let i=0;i<ids.length&&!stopped;i+=3){const results=await Promise.allSettled(ids.slice(i,i+3).map(snapshot));for(const result of results){if(stopped)break;if(result.status==='rejected'){await log('error',result.reason.message);continue;}const o=result.value;session.evidence.push({url:o.url,title:o.title,text:o.text,at:Date.now()});session.evidence=session.evidence.slice(-20);await log('source','Captured '+o.title+'\n'+o.url);}}status('Sources captured. Ask a question to compare or summarize them.');}finally{setRunning(false);}}

async function connect(){token=$('token').value.trim();if(!token)throw Error('Paste the pairing token');const info=await api('/v1/info');const remote=await api('/v1/state');version=remote.version;state=remote.state;$('productBrief').value=state.preferences?.outreach?.product||'';$('leadCriteria').value=state.preferences?.outreach?.criteria||'';$('leadCTA').value=state.preferences?.outreach?.cta||'';$('instructions').value=state.preferences?.instructions||'';$('readOnly').checked=!!state.preferences?.readOnly;session=state.sessions[0]||null;conflict=false;connected=true;await chrome.storage.local.set({token});$('connection').textContent='Connected · '+info.model;$('led').classList.add('online');$('modelInfo').textContent=info.installed.includes(info.model)?'Model installed and ready.':'Configured model is not installed. Available: '+info.installed.join(', ');$('settings').hidden=true;render();await refreshFiles();status('Connected. Select pages or switch to Files.');}
function handler(id,fn){$(id).addEventListener('click',()=>Promise.resolve().then(fn).catch(async e=>{status(e.message);if(session){session.events.push({kind:'error',text:e.message,at:Date.now()});render();await chrome.storage.local.set({workspace:state});}setRunning(false);}));}
handler('settingsToggle',()=>{$('settings').hidden=!$('settings').hidden;});handler('connect',connect);handler('refreshTabs',refreshTabs);
// Permission requests must execute synchronously within a user gesture.
$('grant').addEventListener('click',()=>{const origins=[...new Set(tabs.filter(t=>selected.has(t.id)).map(t=>originPattern(t.url)))];if(!origins.length)return status('Select a tab first');chrome.permissions.request({origins}).then(ok=>status(ok?'Site access granted.':'Site access declined.')).catch(e=>status(e.message));});
handler('savePreferences',async()=>{if(!connected)throw Error('Connect first');state.preferences={instructions:$('instructions').value.slice(0,1500),readOnly:$('readOnly').checked,outreach:{product:$('productBrief').value.slice(0,1600),criteria:$('leadCriteria').value.slice(0,800),cta:$('leadCTA').value.slice(0,300)}};await persist();status('Workflow preferences saved and synced.');});
handler('refreshFiles',refreshFiles);handler('syncHistory',syncHistory);
$('scope').addEventListener('change',scopeUI);$('followTab').addEventListener('change',()=>activePage().catch(e=>status(e.message)));
chrome.tabs.onActivated.addListener(()=>{if(!running)activePage().catch(()=>{});});
chrome.tabs.onUpdated.addListener((id,change,tab)=>{if(!running&&(change.status==='complete'||change.title)&&tab.active)activePage().catch(()=>{});});
chrome.tabs.onRemoved.addListener(()=>{if(!running)refreshTabs().catch(()=>{});});
$('acceptPlan').addEventListener('click',()=>{planResolve?.(true);planResolve=null;});$('cancelPlan').addEventListener('click',()=>{planResolve?.(false);planResolve=null;});
async function exclusive(fn){return navigator.locks.request('atlas-execution',{ifAvailable:true},lock=>{if(!lock)throw Error('A task is running in another Atlas window. Return there or stop it first.');return fn();});}
handler('run',()=>exclusive(run));handler('capture',()=>exclusive(capture));handler('newSession',async()=>{newSession();$('prompt').value='';await persist();});
$('sessions').addEventListener('change',()=>{session=state.sessions.find(s=>s.id===$('sessions').value);$('prompt').value=session?.prompt||session?.goal||'';$('scope').value=session?.scope||'browser';$('skill').value=session?.skill||'research';scopeUI();render();});
$('approve').addEventListener('click',()=>{if(!$('reviewed').checked)return status('Review the target on the page and tick the checkbox.');if(activeSend){try{activeSend.recipient=validateRecipient($('recipientName').value,$('recipientUrl').value,activeSend.observation);}catch(error){return status(error.message);}}resolveApproval(true);});$('reject').addEventListener('click',()=>resolveApproval(false));
$('stop').addEventListener('click',()=>{stopped=true;planResolve?.(false);planResolve=null;controller?.abort();resolveApproval(false);status('Stopping. An already-dispatched action cannot be undone.');});
for(const button of document.querySelectorAll('[data-prompt]'))button.addEventListener('click',()=>{$('prompt').value=button.dataset.prompt;$('skill').value=button.dataset.skill;$('prompt').focus();});
handler('export',()=>{if(!session)return;const report=`# ${session.title}\n\n`+session.events.map(e=>`## ${e.kind}\n${e.text}\n`).join('\n')+'\n## Source evidence\n'+session.evidence.map(e=>`### ${e.title}\n${e.url}\nCaptured: ${new Date(e.at).toISOString()}\n\n${e.text}`).join('\n\n');const a=document.createElement('a');const url=URL.createObjectURL(new Blob([report],{type:'text/markdown'}));a.href=url;a.download='atlas-research.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
async function recordDelivery(outcome){if(!session?.awaitingDelivery)return;await api('/v1/outreach/resolve','POST',{id:session.awaitingDelivery.id,status:outcome});delete session.awaitingDelivery;delete session.pending;session.status=outcome==='user_confirmed'?'Send confirmed by user':'Send outcome uncertain';await log('verification',outcome==='user_confirmed'?'User verified the recipient and sent-message record. This is a human attestation, not a platform delivery receipt.':'Send outcome is uncertain. Exact-message retries remain blocked.');status('Outcome recorded. Review before starting another outreach task.');}
handler('confirmDelivery',()=>recordDelivery('user_confirmed'));handler('uncertainDelivery',()=>recordDelivery('uncertain'));
handler('openWorkspace',async()=>{const url=chrome.runtime.getURL('panel.html?workspace=1');const existing=(await chrome.tabs.query({})).find(t=>t.url===url);const tab=existing?await chrome.tabs.update(existing.id,{active:true}):await chrome.tabs.create({url});await chrome.tabs.update(tab.id,{autoDiscardable:false});});
if(new URLSearchParams(location.search).has('workspace')){document.body.classList.add('workspace');$('openWorkspace').hidden=true;}
window.addEventListener('beforeunload',event=>{if(running){event.preventDefault();event.returnValue='';}});
$('extensionId').textContent=chrome.runtime.id;
const saved=await chrome.storage.local.get(['token','workspace']);token=saved.token||'';$('token').value=token;if(saved.workspace){state=saved.workspace;session=state.sessions[0]||null;}scopeUI();render();await activePage();if(token){try{await connect();}catch(e){status(e.message);$('settings').hidden=false;}}else $('settings').hidden=false;

setInterval(()=>{if(!running&&!saving&&connected&&!conflict&&$('settings').hidden)syncHistory(true).catch(()=>{});},15000);
