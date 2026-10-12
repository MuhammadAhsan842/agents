// Two-sided 95% Wilson interval. Only use on independently sampled task outcomes.
export function wilson(successes,total,z=1.959963984540054){
 if(!Number.isInteger(total)||!Number.isInteger(successes)||total<0||successes<0||successes>total)throw Error('Invalid counts');
 if(!total)return null;
 const p=successes/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,h=z*Math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d;
 return {lower:Math.max(0,c-h),upper:Math.min(1,c+h)};
}
export function summarize(results,planned,distinctCases){
 const measured=results.filter(r=>r.status!=='error');
 const passed=measured.filter(r=>r.status==='passed').length;
 const executable=measured.filter(r=>r.expected!=='ask');
 const completed=executable.filter(r=>r.status==='passed').length;
 const times=measured.map(r=>r.ms).sort((a,b)=>a-b),pct=p=>times.length?times[Math.max(0,Math.ceil(times.length*p)-1)]:null;
 // Repeated seeds are not new independent tasks. Use only the first run per case.
 const first=results.filter(r=>r.repeat===0),firstPassed=first.filter(r=>r.status==='passed').length;
 return {planned,measured:measured.length,passed,errors:results.length-measured.length,
  decisionAccuracy:measured.length?passed/measured.length:null,
  completionRate:executable.length?completed/executable.length:null,
  coverage:executable.length?executable.filter(r=>r.action.action!=='ask').length/executable.length:null,
  abstained:executable.filter(r=>r.action.action==='ask').length,
  firstRunInterval:first.length===distinctCases&&first.every(r=>r.status!=='error')?wilson(firstPassed,distinctCases):null,
  intervalAssumption:'Distinct tasks must be independently sampled and representative; repeats are excluded.',
  latencyMs:{p50:pct(.5),p95:pct(.95)},productionAccuracyEstablished:false};
}
