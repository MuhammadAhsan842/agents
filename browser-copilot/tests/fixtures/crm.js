import http from 'node:http';
import {randomUUID} from 'node:crypto';
const shell = body => `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Atlas Test CRM — fictional data</title><style>body{font:18px system-ui;margin:36px;max-width:800px}a,button,input,textarea{display:block;margin:16px 0;font:inherit}textarea{width:600px;height:100px}article{border:1px solid #bbb;padding:16px;margin:16px 0}</style></head><body><p>LOCAL TEST FIXTURE · NO REAL CONTACTS</p>${body}</body></html>`;
export function createCRM() {
  const messages = [];
  const server = http.createServer(async (req,res) => {
    if (req.method === 'POST' && req.url === '/api/messages') {
      let raw='';for await(const chunk of req){raw+=chunk;if(raw.length>10000){res.writeHead(413);return res.end();}}
      try {const body=JSON.parse(raw);if(body.recipient!=='ada'||typeof body.message!=='string'||!body.message.trim())throw Error('Invalid message');
        const existing=messages.find(m=>m.key===body.key);const record=existing||{id:randomUUID(),recipient:'ada',message:body.message,key:body.key};if(!existing)messages.push(record);
        res.setHeader('Content-Type','application/json');res.end(JSON.stringify(record));
      }catch{res.writeHead(400);res.end('Invalid fixture request');}return;
    }
    res.setHeader('Content-Type','text/html');
    if(req.url==='/leads/ada')return res.end(shell(`<h1>Ada Example</h1><p>Operations Director · Cedar Software · United Kingdom · 35 employees</p><p>Team handles customer support knowledge and document search.</p><button id="compose" onclick="document.getElementById('composer').hidden=false;this.hidden=true">Message</button><div id="composer" hidden><h2>Message to Ada Example</h2><label for="message">Your message</label><textarea id="message"></textarea><button id="send">Send message</button></div><section id="thread"></section><script>
const key=crypto.randomUUID();document.getElementById('send').onclick=async()=>{const button=document.getElementById('send');button.disabled=true;try{const response=await fetch('/api/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recipient:'ada',message:document.getElementById('message').value,key})});if(!response.ok)throw Error('Send failed');const record=await response.json();const entry=document.createElement('article');entry.textContent='Message recorded for Ada Example. Receipt: '+record.id+' — '+record.message;document.getElementById('thread').append(entry);document.getElementById('composer').hidden=true;}catch(error){document.getElementById('thread').textContent=error.message;}};
</script>`));
    if(req.url==='/leads')return res.end(shell('<h1>Lead search results</h1><article><h2>Ada Example</h2><p>Operations Director · Cedar Software · United Kingdom · 35 employees</p><a href="/leads/ada">View Ada Example profile</a></article><article><h2>Ben Sample</h2><p>Designer · Large Retail · Canada · 2000 employees</p></article>'));
    res.writeHead(404);res.end(shell('<h1>Not found</h1>'));
  });
  return {server,messages};
}
