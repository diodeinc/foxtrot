/// Takes STEP text and returns an expanded triangle buffer.
///
/// Vertices are packed into rows of 9 floats, representing
/// - Position
/// - Normal
/// - Color
///
/// Each consecutive group of three vertices forms one triangle.
///
use wasm_bindgen::prelude::*;
use log::{Level};

#[wasm_bindgen]
pub fn init_log() {
    console_log::init_with_level(Level::Info).expect("Failed to initialize log");
}

#[wasm_bindgen]
pub fn tessellate_step(data: String) -> Result<JsValue, JsValue> {
    use step::step_file::StepFile;
    use triangulate::triangulate::triangulate;

    let flat = StepFile::strip_flatten(data.as_bytes())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let step = StepFile::parse(&flat)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (mesh, stats) = triangulate(&step);
    let buffer = mesh.to_triangle_buffer();
    let out = js_sys::Object::new();
    js_sys::Reflect::set(&out, &"schema".into(), &2.into())?;
    js_sys::Reflect::set(&out, &"completion".into(),
        &if stats.is_complete() { "complete" } else { "partial" }.into())?;
    js_sys::Reflect::set(&out, &"numFaces".into(), &(stats.num_faces as f64).into())?;
    js_sys::Reflect::set(&out, &"numShells".into(), &(stats.num_shells as f64).into())?;
    js_sys::Reflect::set(&out, &"failures".into(),
        &serde_wasm_bindgen::to_value(&stats.failures).map_err(|e| JsValue::from_str(&e.to_string()))?)?;
    // Keep geometry in a typed array; never serialize millions of JS numbers.
    js_sys::Reflect::set(&out, &"triangles".into(), &js_sys::Float32Array::from(buffer.as_slice()))?;
    Ok(out.into())
}
