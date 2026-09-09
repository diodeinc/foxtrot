importScripts("wasm.js");

const { tessellate_step, init_log } = wasm_bindgen;
const ready = wasm_bindgen({module_or_path: "wasm_bg.wasm"}).then(init_log);

onmessage = async function(e) {
    try {
        await ready;
        const outcome = tessellate_step(e.data);
        postMessage(outcome, [outcome.triangles.buffer]);
    } catch (error) {
        postMessage({schema: 2, completion: "failed", failures: [{
            kind: "execution_error", message: String(error)
        }]});
    }
}
