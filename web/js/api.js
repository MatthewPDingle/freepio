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
  lock: (path, mode, label) => req('POST', '/api/lock', { path, mode, label }),
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
