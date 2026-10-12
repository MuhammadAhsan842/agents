// Injected in the isolated world. No executable code is accepted from the model.
(()=>{
 if(globalThis.__atlas)return;
 let targets=new Map(),snapshotId='',signature='';
 const visible=e=>{const r=e.getBoundingClientRect(),s=getComputedStyle(e);return r.width>0&&r.height>0&&s.visibility!=='hidden'&&s.display!=='none';};
 const label=e=>(e.getAttribute('aria-label')||e.labels?.[0]?.innerText||e.getAttribute('placeholder')||e.innerText||e.getAttribute('title')||e.name||'').replace(/\s+/g,' ').trim().slice(0,180);
 const sensitive=e=>e.type==='password'||/password|credit.?card|card.?number|cvv|otp|one.?time|secret|token/i.test([e.name,e.id,e.autocomplete,label(e)].join(' '));
 const hash=s=>{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return (h>>>0).toString(16);};
 function text(){
  if(!document.body)return '';
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let node,count=0,total=0;const parts=[];
  while((node=walker.nextNode())&&count++<6000&&total<14000){const e=node.parentElement;if(!e||e.closest('script,style,noscript,svg,input,textarea,[contenteditable],[hidden]')||!visible(e))continue;const value=node.textContent.replace(/\s+/g,' ').trim();if(value){parts.push(value);total+=value.length;}}
  return parts.join('\n').slice(0,14000);
 }
 function fingerprint(){return hash(location.href+'\n'+text()+'\n'+[...targets.values()].map(e=>label(e)+'|'+e.getAttribute('href')+'|'+e.isConnected+'|'+(!sensitive(e)?(e.value||e.isContentEditable&&e.textContent||''):'')).join('\n'));}
 function snapshot(){
  const nodes=[...document.querySelectorAll('a[href],button,input:not([type=hidden]),textarea,select,[role=button],[role=link],[contenteditable=true]')].filter(e=>{const r=e.getBoundingClientRect();return visible(e)&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth;});
  targets=new Map();
  const elements=nodes.slice(0,70).map((e,i)=>{const id='e'+i;targets.set(id,e);let href='';try{const u=new URL(e.getAttribute('href'),location.href);if(e.hasAttribute('href')&&['http:','https:'].includes(u.protocol)&&!u.username&&!u.password)href=u.href;}catch{}
   return {id,tag:e.tagName.toLowerCase(),role:e.getAttribute('role')||'',label:label(e),href,editable:e.matches('input,textarea,[contenteditable=true]'),sensitive:sensitive(e),disabled:!!e.disabled||e.getAttribute('aria-disabled')==='true',...(e.tagName==='SELECT'?{options:[...e.options].slice(0,30).map(o=>({value:o.value,label:o.text}))}:{})};});
  snapshotId=crypto.randomUUID();signature=fingerprint();return {snapshotId,fingerprint:signature,url:location.href,title:document.title,text:text(),elements,truncated:nodes.length>70};
 }
 async function act(p){
  if(p.snapshotId!==snapshotId||p.fingerprint!==signature||fingerprint()!==signature)throw Error('Page changed. Take a new snapshot before acting.');
  const a=p.action,e=targets.get(a.target);
  if(['click','type','select'].includes(a.action)&&(!e?.isConnected||!visible(e)||sensitive(e)||e.disabled||e.getAttribute('aria-disabled')==='true'))throw Error('Target is unavailable or sensitive');
  if(a.action==='click') {e.scrollIntoView({block:'center'});e.click();}
  else if(a.action==='type'){
   if(!e.matches('input,textarea,[contenteditable=true]')||e.readOnly||['file','checkbox','radio','button','submit'].includes(e.type))throw Error('Cannot type into this element');
   e.focus();if(e.isContentEditable)e.textContent=a.value;else{const proto=e.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(e,a.value);}
   e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));
   if((e.isContentEditable?e.textContent:e.value)!==a.value)throw Error('Field did not retain the requested text');
  }else if(a.action==='select') {if(e.tagName!=='SELECT'||![...e.options].some(o=>o.value===a.value))throw Error('Invalid option');e.value=a.value;e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));if(e.value!==a.value)throw Error('Selection failed');}
  else if(a.action==='scroll'){if(!['up','down'].includes(a.value))throw Error('Invalid direction');window.scrollBy(0,(a.value==='up'?-1:1)*innerHeight*.8);}
  else throw Error('Unsupported DOM action');
  snapshotId='';return {executed:true,verified:['type','select'].includes(a.action),note:['type','select'].includes(a.action)?'Field value checked':'Dispatched; inspect the next snapshot to verify the outcome'};
 }
 function draft(p){
  if(p.snapshotId!==snapshotId||p.fingerprint!==signature||fingerprint()!==signature)throw Error('Page changed before message review');
  const candidates=[...targets.entries()].filter(([,e])=>e.matches('textarea,[contenteditable=true]')&&!sensitive(e)&&visible(e)&&(e.isContentEditable?e.textContent:e.value).trim());
  if(candidates.length!==1)throw Error('Cannot identify one unambiguous message composer. Close other composers and try again.');
  const [id,e]=candidates[0];const message=e.isContentEditable?e.textContent:e.value;
  if(message.length>4000)throw Error('Message exceeds the 4000-character review limit');
  return {composerId:id,message};
 }
 globalThis.__atlas={snapshot,act,draft};
})();
