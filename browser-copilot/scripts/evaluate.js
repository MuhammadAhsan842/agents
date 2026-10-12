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
let passed=0;const results=[];
for(const c of cases){const start=Date.now();try{const r=await plan({goal:c.goal,scope:c.scope||'browser',observation:{...observation,files:[{path:'brief.md',bytes:100},{path:'revenue.csv',bytes:200}]},tabs:[{id:2,title:'Pricing',url:'https://example.test/pricing'}]},{model:process.env.ATLAS_MODEL||'qwen3.5:0.8b',ollama:process.env.OLLAMA_URL||'http://127.0.0.1:11434'});const ok=r.action.action===c.expected&&(!c.target||r.action.target===c.target)&&(!c.value||r.action.value===c.value);passed+=Number(ok);results.push({goal:c.goal,ok,action:r.action,ms:Date.now()-start});}catch(e){results.push({goal:c.goal,ok:false,error:e.message});}}
console.log(JSON.stringify({metric:'single-step fixture selection; NOT end-to-end browser success',model:process.env.ATLAS_MODEL||'qwen3.5:0.8b',passed,total:cases.length,rate:passed/cases.length,results},null,2));
if(passed!==cases.length)process.exitCode=1;
