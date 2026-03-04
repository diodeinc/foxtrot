importScripts("wasm.js");

const { step_to_triangle_buf_with_timeout, init_log } = wasm_bindgen;
const DEFAULT_TIMEOUT_MS = 30_000;
async function run() {
    await wasm_bindgen();
    init_log();

    onmessage = function(e) {
        try {
            var triangles = step_to_triangle_buf_with_timeout(e.data, DEFAULT_TIMEOUT_MS);
            postMessage({ ok: true, triangles });
        } catch (err) {
            postMessage({ ok: false, error: String(err) });
        }
    }
}
run();
