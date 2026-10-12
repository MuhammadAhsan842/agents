import {plan} from '../server/planner.js';
const observation={url:'https://example.test/',title:'Demo CRM',text:'Contact: Ada. Revenue: 42 USD. Period: June.',elements:[{id:'e0',tag:'input',label:'Full name',editable:true},{id:'e1',tag:'button',label:'Save'},{id:'e2',tag:'a',label:'Details',href:'https://example.test/details'}]};
const cases=[
 {goal:'Fill Full name with Ahsan.',expected:'type',target:'e0'},
 {goal:'Click Save.',expected:'click',target:'e1'},
 {goal:'Scroll down.',expected:'scroll',value:'down'},
 {goal:'Scroll up.',expected:'scroll',value:'up'},
 {goal:'Capture this page as evidence.',expected:'extract'},
 {goal:'Switch to the tab titled Pricing.',expected:'switch_tab',target:'2'},
 {goal:'Read the local file brief.md.',scope:'files',expected:'read_file',target:'brief.md'},
 {goal:'Compute numeric column summaries for revenue.csv.',scope:'files',expected:'analyze_csv',target:'revenue.csv'}
];
const config={model:process.env.ATLAS_MODEL||'qwen3.5:0.8b',ollama:process.env.OLLAMA_URL||'http://127.0.0.1:11434'};
const repeats=Number(process.env.ATLAS_EVAL_REPEATS||3);
if(!Number.isInteger(repeats)||repeats<1||repeats>100)throw Error('ATLAS_EVAL_REPEATS must be 1–100');
const results=[];
const measuredAvailable=()=>results.some(r=>r.status!=='error');
for(let repeat=0;repeat<repeats;repeat++)for(const c of cases){
 const start=Date.now();
 try{
  const r=await plan({goal:c.goal,scope:c.scope||'browser',observation:{...observation,files:[{path:'brief.md',bytes:100},{path:'revenue.csv',bytes:200}]},tabs:[{id:2,title:'Pricing',url:'https://example.test/pricing'}]},config);
  const ok=r.action.action===c.expected&&(!c.target||r.action.target===c.target)&&(!c.value||r.action.value===c.value);
  results.push({repeat,goal:c.goal,status:ok?'passed':r.action.action==='ask'?'abstained':'failed',action:r.action,metrics:r.metrics,ms:Date.now()-start});
 }catch(e){results.push({repeat,goal:c.goal,status:'error',error:e.message,ms:Date.now()-start});}
 // An unavailable model is not a measured zero-percent accuracy score.
 if(!measuredAvailable()&&results[0]?.status==='error'){repeat=repeats;break;}
}
const measured=results.filter(r=>r.status!=='error'),passed=results.filter(r=>r.status==='passed').length;
const times=measured.map(r=>r.ms).sort((a,b)=>a-b),pct=p=>times.length?times[Math.max(0,Math.ceil(times.length*p)-1)]:null;
console.log(JSON.stringify({metric:'single-step fixture selection; NOT end-to-end browser success',model:config.model,repeats,planned:cases.length*repeats,completed:measured.length,passed,abstained:results.filter(r=>r.status==='abstained').length,errors:results.filter(r=>r.status==='error').length,rate:measured.length?passed/measured.length:null,latencyMs:{p50:pct(.5),p95:pct(.95)},results},null,2));
if(passed!==cases.length*repeats)process.exitCode=1;
