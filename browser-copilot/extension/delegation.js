// Delegation is a user-selected capability, never a model-selected confidence score.
export function canDelegate(action, preview, approvedOutputs = []) {
  return ['write_file', 'create_artifact', 'create_dashboard'].includes(action.action)
    && preview?.willOverwrite === false
    && approvedOutputs.includes(action.target);
}
export function missingOutputs(expected, delivered) {
  const paths = new Set(delivered.map(item => item.path));
  return expected.filter(path => !paths.has(path));
}
export function parseOutputs(text) {
  const paths = [...new Set(text.split('\n').map(s => s.trim()).filter(Boolean))];
  if (paths.length > 12 || paths.some(p => p.length > 240 || p.startsWith('/') || p.includes('\\') || p.split('/').some(s => !s || s.startsWith('.')))) {
    throw Error('Use up to 12 relative output paths, one per line, without hidden or parent directories.');
  }
  return paths;
}
