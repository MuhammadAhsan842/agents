import {createHash,randomUUID} from 'node:crypto';
export function createOutreachLedger(db) {
  db.exec(`CREATE TABLE IF NOT EXISTS outreach (
    id TEXT PRIMARY KEY, dedupe TEXT UNIQUE NOT NULL, recipient TEXT NOT NULL,
    profile_url TEXT NOT NULL, message TEXT NOT NULL, status TEXT NOT NULL,
    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
  )`);
  function prepare({recipient,profileUrl,message}) {
    if (typeof recipient !== 'string' || recipient.trim().length < 3 || recipient.length > 160) throw Error('Invalid recipient');
    if (typeof message !== 'string' || !message.trim() || message.length > 4000) throw Error('Message must contain 1–4000 characters');
    const url = new URL(profileUrl);
    if (!['https:','http:'].includes(url.protocol) || url.username || url.password) throw Error('Invalid profile URL');
    url.hash = '';
    const dedupe = createHash('sha256').update(url.href+'\n'+message).digest('hex');
    const previous = db.prepare('SELECT id,status FROM outreach WHERE dedupe=?').get(dedupe);
    if (previous && previous.status !== 'prepared') throw Error('This exact message already has a send attempt. Check the conversation; automatic retry is blocked.');
    if (previous) return previous;
    const id = randomUUID(), now = Date.now();
    db.prepare('INSERT INTO outreach VALUES (?,?,?,?,?,?,?,?)').run(id,dedupe,recipient.trim(),url.href,message,'prepared',now,now);
    return {id,status:'prepared'};
  }
  function claim(id) {
    const result = db.prepare("UPDATE outreach SET status='dispatching',updated_at=? WHERE id=? AND status='prepared'").run(Date.now(),id);
    if (result.changes !== 1) throw Error('Send attempt already claimed or missing. Do not retry an uncertain send.');
    return {id,status:'dispatching'};
  }
  function resolve({id,status}) {
    if (!['user_confirmed','uncertain'].includes(status)) throw Error('Delivery must be user-confirmed or uncertain');
    const result = db.prepare("UPDATE outreach SET status=?,updated_at=? WHERE id=? AND status IN ('dispatching','uncertain')").run(status,Date.now(),id);
    if (result.changes !== 1) throw Error('No unresolved send attempt found');
    return {id,status};
  }
  return {prepare,claim,resolve};
}
