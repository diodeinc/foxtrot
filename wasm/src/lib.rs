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
pub fn step_to_triangle_buf(data: String) -> Result<Vec<f32>, JsValue> {
    use step::step_file::StepFile;
    use triangulate::triangulate::triangulate;

    let flat = StepFile::strip_flatten(data.as_bytes())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let step = StepFile::parse(&flat)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (mesh, _stats) = triangulate(&step);
    Ok(mesh.to_triangle_buffer())
}
