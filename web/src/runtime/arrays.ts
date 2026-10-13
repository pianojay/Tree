export type ArraySpec = {shape:number[];b64:string};
export type PackedParams = Record<string,ArraySpec>;
export function unpack(a:ArraySpec):Float32Array {
  const bytes=Uint8Array.from(atob(a.b64),c=>c.charCodeAt(0));
  return new Float32Array(bytes.buffer);
}
