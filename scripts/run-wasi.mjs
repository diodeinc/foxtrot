// Run a wasm32-wasip1 binary under Node's WASI implementation.
// Usage: node run-wasi.mjs <module.wasm> [args...]
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { argv, env } from "node:process";

const [, , wasmPath, ...args] = argv;
const wasi = new WASI({
  version: "preview1",
  args: [wasmPath, ...args],
  env,
  preopens: { "/Users/lenny": "/Users/lenny" },
});

const wasm = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
wasi.start(instance);
