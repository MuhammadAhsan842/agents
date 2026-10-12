export function safeURL(s){const u=new URL(s);if(!['http:','https:'].includes(u.protocol)||u.username||u.password)throw Error('Only HTTP(S) pages are supported');return u.href;}
export function originPattern(url){return new URL(safeURL(url)).origin+'/*';}
export function needsApproval(action){return ['click','type','select','navigate','open_tab','write_file','rename_file','create_artifact'].includes(action.action);}
export function validateProposal(a,o,tabs,goal){
 if(!a||!['click','type','select','navigate','open_tab','switch_tab','extract','scroll','done','ask','read_file','analyze_csv','write_file','rename_file','create_artifact'].includes(a.action))throw Error('Unknown browser action');
 if(['click','type','select'].includes(a.action)){
  const e=o.elements.find(e=>e.id===a.target);if(!e||e.disabled||e.sensitive)throw Error('Target is not actionable');
  if(a.action==='type'&&!e.editable)throw Error('Target is not editable');
  if(a.action==='select'&&e.tag!=='select')throw Error('Target is not a select');
 }
 if(['navigate','open_tab'].includes(a.action)){
  const url=safeURL(a.value);const urls=(goal.match(/https?:\/\/[^\s<>"']+/g)||[]);
  if(!o.elements.some(e=>e.href===url)&&!urls.includes(url))throw Error('Unobserved navigation URL');
 }
 if(a.action==='switch_tab'&&!tabs.some(t=>String(t.id)===a.target))throw Error('Tab is outside this workspace');
 if(a.action==='scroll'&&!['up','down'].includes(a.value))throw Error('Invalid scroll direction');
 return a;
}
