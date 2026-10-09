export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

/** Wrap an async handler so rapid calls collapse: the latest argument always runs. */
export function latest<A>(fn: (arg: A) => Promise<void>) {
  let busy = false, pending = false, want: A;
  return async (arg: A) => {
    want = arg; pending = true;
    if (busy) return;
    busy = true;
    try { while (pending) { pending = false; await fn(want); } }
    finally { busy = false; }
  };
}

export async function post(url: string, body: unknown): Promise<void> {
  await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
