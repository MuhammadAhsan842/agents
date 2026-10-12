// Sequential isolated processes: avoid CPU contention and keep JSON on stdout.
import {spawnSync} from 'node:child_process';
const profiles=['legacy','flat','compact','compact-flat'];
const reports=[];
for(const profile of profiles){
 console.error('Evaluating controller profile:',profile);
 const r=spawnSync(process.execPath,['scripts/evaluate.js'],{encoding:'utf8',env:{...process.env,ATLAS_CONTROLLER_PROFILE:profile,ATLAS_EVAL_REPEATS:process.env.ATLAS_EVAL_REPEATS||'1'},maxBuffer:16*1024*1024});
 if(r.error)throw r.error;
 let report;try{report=JSON.parse(r.stdout);}catch{throw Error('Evaluation did not return JSON: '+r.stderr);}
 reports.push(report);
 if(!report.measured){console.error('No measured responses; stopping comparison.');break;}
}
console.log(JSON.stringify({experiment:'2x2 prompt/schema ablation; same model, temperature, context, token budget and validators',limitations:'Development-set comparison. No winner is automatically deployed. Profiles run sequentially; timing includes order/cache effects. Rerun in different orders for latency conclusions.',reports},null,2));
if(reports.length!==profiles.length||reports.some(r=>r.errors))process.exitCode=1;
