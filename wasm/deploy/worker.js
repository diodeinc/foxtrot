importScripts("wasm.js");

const { tessellate_step, init_log } = wasm_bindgen;
async function run() {
    await wasm_bindgen();
    init_log();

    onmessage = function(e) {
        try {
            const outcome = tessellate_step(e.data);
            postMessage(outcome, [outcome.triangles.buffer]);
        } catch (error) {
            postMessage({schema: 2, completion: "failed", failures: [{
                kind: "execution_error", message: String(error)
            }]});
        }
    }
}
run();
