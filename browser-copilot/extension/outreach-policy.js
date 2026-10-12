export function isSendAction(action, observation) {
  const target = observation.elements?.find(item => item.id === action.target);
  return action.action === 'click' && /\bsend\b/i.test(target?.label || '');
}
export function buildLeadGoal(goal, brief = {}) {
  if (!goal?.trim()) throw Error('Describe your product and who you want to reach in the prompt.');
  const saved = brief?.product || brief?.criteria ? `\nOPTIONAL SAVED BRIEF: ${String(brief.product||'').slice(0,1600)}\nOPTIONAL SAVED CRITERIA: ${String(brief.criteria||'').slice(0,800)}\nOPTIONAL CTA: ${String(brief.cta||'').slice(0,300)}` : '';
  return `${goal}${saved}\nUse the user's product details and target criteria from this prompt and conversation. Current instructions override saved preferences. Ask only for essential details that are genuinely missing. Find one matching lead at a time. Cite observed profile evidence. Never invent proof, customer results, or contact details. Draft before sending. Stop after one send attempt and let the user verify delivery.`;
}

export function validateRecipient(recipient, profileUrl, observation) {
  const name = recipient.trim();
  if (name.length < 3 || name.length > 160) throw Error('Enter the recipient’s full name.');
  const normalize = value => value.toLowerCase().replace(/\s+/g, ' ').trim();
  if (!normalize(observation.text).includes(normalize(name))) throw Error('Recipient name is not present in the observed page. Open their profile first.');
  const profile = new URL(profileUrl);
  const page = new URL(observation.url);
  profile.hash = ''; page.hash = '';
  if (!['https:', 'http:'].includes(profile.protocol) || profile.username || profile.password || profile.href !== page.href) {
    throw Error('Recipient profile URL must match the current page. Verify the conversation recipient on that page.');
  }
  return {recipient:name, profileUrl:profile.href};
}

export function chooseWorkflow(selected, goal) {
  if (selected !== 'auto') return selected;
  if (/\b(leads?|prospects?|outreach|sales navigator)\b|linkedin.*(?:message|find)|(?:ceos?|founders?).*message/i.test(goal)) return 'leads';
  if (/\b(dashboard|metrics|portal|analytics)\b/i.test(goal)) return 'portal';
  return 'research';
}
