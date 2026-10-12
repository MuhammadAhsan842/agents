const normalized=s=>String(s||'').normalize('NFKC').toLowerCase().replace(/\s+/g,' ').trim();
export function verifyContract(a,{observation,requiredOutputs=[],delivered=[]}){
 const elements=observation.elements||[],e=elements.find(e=>e.id===a.target);
 if(['click','type','select'].includes(a.action)&&e){
  const peers=elements.filter(x=>!x.disabled&&!x.sensitive&&x.tag===e.tag&&normalized(x.label)===normalized(e.label));
  // IDs distinguish DOM nodes, but do not prove which identically labelled recipient is intended.
  if(peers.length>1)throw Error('Ambiguous duplicate control labels. Ask the user to isolate the intended control.');
  if(a.action==='select'&&!(e.options||[]).some(o=>o.value===a.value&&!o.disabled))throw Error('Select value must match an observed enabled option');
 }
 if(['read_file','analyze_csv'].includes(a.action)&&!(observation.files||[]).some(f=>f.path===a.target))throw Error('Read target must exist in the supplied file inventory');
 if(a.action==='analyze_csv'&&!/\.csv$/i.test(a.target))throw Error('Numeric analysis requires a CSV file');
 if(a.action==='done'){
  const missing=requiredOutputs.filter(path=>!delivered.some(d=>d.path===path));
  if(missing.length)throw Error('Required outputs are not delivered: '+missing.join(', '));
 }
 return a;
}
// Full action-specific branches keep type targets, select values and tab IDs separate.
export function constrainedSchema(base,allowed,o,tabs,goal){
 const active=o.elements.filter(e=>!e.disabled&&!e.sensitive);
 const branches=[];
 const branch=(action,target,value=base.properties.value)=>branches.push({type:'object',additionalProperties:false,required:base.required,properties:{...base.properties,action:{type:'string',enum:[action]},target,value}});
 const empty={type:'string',enum:['']};
 for(const action of allowed){
  if(['click','type'].includes(action)){
   const ids=active.filter(e=>action!=='type'||e.editable).map(e=>e.id);
   if(ids.length)branch(action,{type:'string',enum:ids},action==='click'?empty:base.properties.value);
  }else if(action==='select'){
   for(const e of active.filter(e=>e.tag==='select')){const values=(e.options||[]).filter(v=>!v.disabled).map(v=>v.value);if(values.length)branch(action,{type:'string',enum:[e.id]},{type:'string',enum:values});}
  }else if(action==='switch_tab'){
   if(tabs.length)branch(action,{type:'string',enum:tabs.map(t=>String(t.id))},empty);
  }else if(action==='scroll')branch(action,empty,{type:'string',enum:['up','down']});
  else if(['read_file','analyze_csv'].includes(action)){
   const paths=(o.files||[]).map(f=>f.path).filter(p=>action!=='analyze_csv'||/\.csv$/i.test(p));if(paths.length)branch(action,{type:'string',enum:paths},empty);
  }else if(['navigate','open_tab'].includes(action)){
   const urls=[...o.elements.map(e=>e.href),...(goal.match(/https?:\/\/[^\s<>"']+/g)||[])].flatMap(s=>{try{const u=new URL(s);return ['http:','https:'].includes(u.protocol)&&!u.username&&!u.password?[u.href]:[];}catch{return [];}});
   if(urls.length)branch(action,empty,{type:'string',enum:[...new Set(urls)]});
  }else branch(action,['extract','done','ask'].includes(action)?empty:base.properties.target);
 }
 return {...base,anyOf:branches};
}
