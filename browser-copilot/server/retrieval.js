// BM25 with positive Robertson IDF, length normalization and stable tie breaking.
// Scores rank evidence; they are NOT probabilities or model confidence.
const tokens=s=>String(s).normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}_]+/gu)||[];
export function rankBM25(texts,query,{k1=1.2,b=.75}={}){
 const docs=texts.map(tokens),terms=[...new Set(tokens(query))];
 const average=docs.reduce((n,d)=>n+d.length,0)/(docs.length||1)||1;
 const df=new Map(terms.map(t=>[t,docs.filter(d=>d.includes(t)).length]));
 return docs.map((d,i)=>{const counts=new Map();for(const t of d)counts.set(t,(counts.get(t)||0)+1);
  const score=terms.reduce((sum,t)=>{const f=counts.get(t)||0;return sum+Math.log(1+(docs.length-df.get(t)+.5)/(df.get(t)+.5))*f*(k1+1)/(f+k1*(1-b+b*d.length/average));},0);
  return {i,score};
 }).sort((a,b)=>b.score-a.score||a.i-b.i);
}
