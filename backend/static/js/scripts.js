export async function api(url, opts={} ){const r=await fetch(url,opts); if(!r.ok) throw new Error(await r.text()); return r.json();}
export function toast(msg,type='info'){alert(msg)}
