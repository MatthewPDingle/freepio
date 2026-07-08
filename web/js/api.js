// Thin API layer.

async function req(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status}`);
  }
  return res.json();
}

export const api = {
  buildSpot: cfg => req('POST', '/api/spot', cfg),
  solve: opts => req('POST', '/api/solve', opts),
  stop: () => req('POST', '/api/stop'),
  status: () => req('GET', '/api/status'),
  node: path => req('POST', '/api/node', { path }),
  exploit: (path, exploiter) => req('POST', '/api/exploit', { path, exploiter }),
  pfBuild: cfg => req('POST', '/api/preflop/spot', cfg),
  pfEstimate: cfg => req('POST', '/api/preflop/estimate', cfg),
  pfSolve: opts => req('POST', '/api/preflop/solve', opts || {}),
  pfStop: () => req('POST', '/api/preflop/stop'),
  pfStatus: () => req('GET', '/api/preflop/status'),
  pfNode: path => req('POST', '/api/preflop/node', { path }),
  pfExport: path => req('POST', '/api/preflop/export', { path }),
  pfTable: seats => req('POST', '/api/preflop/table', { seats }),
  pfGenerate: (seat, stats, name) => req('POST', '/api/preflop/generate', { seat, stats, name }),
  pfArchetypes: () => req('GET', '/api/preflop/archetypes'),
  pfHero: seat => req('POST', '/api/preflop/hero', { seat }),
  pfProfiles: () => req('GET', '/api/preflop/profiles'),
  pfProfileSave: (name, profile) => req('POST', '/api/preflop/profiles/save', { name, profile }),
  pfProfileGet: name => req('POST', '/api/preflop/profiles/get', { name }),
  pfSaveGame: name => req('POST', '/api/preflop/save', { name }),
  pfLoadGame: name => req('POST', '/api/preflop/load', { name }),
  pfSavedGames: () => req('GET', '/api/preflop/saves'),
  lock: (path, mode, label) => req('POST', '/api/lock', { path, mode, label }),
  profileLocks: (player, stats, aggressor) =>
    req('POST', '/api/profile_locks', { player, stats, aggressor }),
  profileLocksClear: () => req('DELETE', '/api/profile_locks'),
  reportsRun: body => req('POST', '/api/reports/run', body),
  reportsStatus: () => req('GET', '/api/reports/status'),
  reportsStop: () => req('POST', '/api/reports/stop'),
  reportsList: () => req('GET', '/api/reports'),
  reportsGet: name => req('POST', '/api/reports/get', { name }),
  reportsDelete: name => req('POST', '/api/reports/delete', { name }),
  unlock: path => req('POST', '/api/unlock', { path }),
  locks: () => req('GET', '/api/locks'),
  runouts: path => req('POST', '/api/runouts', { path }),
  parseRange: text => req('POST', '/api/range/parse', { text }),
  presets: () => req('GET', '/api/presets'),
  save: name => req('POST', '/api/save', { name }),
  load: name => req('POST', '/api/load', { name }),
  saves: () => req('GET', '/api/saves'),
};

let toastTimer = null;
export function toast(msg, isError = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = isError ? 'show error' : 'show';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, isError ? 5000 : 2500);
}
