// Run a wasm32-wasip1 binary under Node's WASI implementation.
// Usage: node run-wasi.mjs <module.wasm> [args...]
//
// The current directory and the parent directory of every existing
// path-like argument are preopened, so absolute input paths work without
// granting access to the whole filesystem.
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { WASI } from "node:wasi";
import { argv, cwd, env } from "node:process";

const [, , wasmPath, ...rawArgs] = argv;

// Absolutize path-like arguments (the WASI guest has no notion of the host
// working directory) and preopen their parent directories.
const preopens = { [cwd()]: cwd() };
const args = rawArgs.map((arg) => {
  const abs = resolve(arg);
  if (!existsSync(abs)) {
    return arg;
  }
  const dir = dirname(abs);
  preopens[dir] = dir;
  return abs;
});

const wasi = new WASI({
  version: "preview1",
  args: [wasmPath, ...args],
  env,
  preopens,
});

const wasm = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
wasi.start(instance);
