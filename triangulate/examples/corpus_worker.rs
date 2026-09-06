//! One process per model. The Python harness owns timeouts and repetition.
use std::time::Instant;

use serde::Serialize;
use step::step_file::StepFile;
use triangulate::triangulate::triangulate;
use triangulate::stats::TessellationFailure;

#[derive(Serialize)]
struct WorkerReport {
    schema: u32,
    status: &'static str,
    message: Option<String>,
    read_ms: f64, parse_ms: f64, triangulate_ms: f64, export_ms: f64,
    triangles: usize, vertices: usize, faces: usize, shells: usize,
    completion: &'static str,
    failures: Vec<TessellationFailure>,
    degenerate_f64: usize, browser_nonfinite: usize, browser_triangles: usize,
    browser_degenerate: usize, browser_zero_normals: usize,
}

fn write_input_error(path: &str, read_ms: f64, message: String) -> Result<(), Box<dyn std::error::Error>> {
    let report = WorkerReport { schema: 2, status: "input_error", message: Some(message), read_ms,
        parse_ms: 0., triangulate_ms: 0., export_ms: 0., triangles: 0, vertices: 0,
        faces: 0, shells: 0, completion: "failed", failures: vec![], degenerate_f64: 0,
        browser_nonfinite: 0, browser_triangles: 0, browser_degenerate: 0, browser_zero_normals: 0 };
    std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn collinear(points: [nalgebra_glm::DVec3; 3]) -> bool {
    // A rounded cross product can vanish for a noncollinear thin triangle.
    // Use the same adaptive orientation predicates as the triangulator.
    (0..3).all(|i| {
        let [a, b, c] = points.map(|p| robust::Coord { x: p[i], y: p[(i + 1) % 3] });
        robust::orient2d(a, b, c) == 0.
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 && args.len() != 5 {
        return Err("usage: corpus_worker INPUT.step METRICS.json OUTPUT.stl|- [BROWSER.bin]".into());
    }
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let start = Instant::now();
    let data = match std::fs::read(&args[1]) {
        Ok(data) => data,
        Err(e) => return write_input_error(&args[2], 0., e.to_string()),
    };
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let flat = match StepFile::strip_flatten(&data) {
        Ok(flat) => flat,
        Err(e) => return write_input_error(&args[2], read_ms, e.to_string()),
    };
    let step = match StepFile::parse(&flat) {
        Ok(step) => step,
        Err(e) => return write_input_error(&args[2], read_ms, e.to_string()),
    };
    let parse_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let (mesh, stats) = triangulate(&step);
    let triangulate_ms = start.elapsed().as_secs_f64() * 1000.0;
    // Distinguish tessellation defects from precision lost by binary STL's
    // f32 coordinates. The harness independently validates the exported mesh.
    let degenerate_f64 = mesh.triangles.iter().filter(|triangle| {
        let a = mesh.verts[triangle.verts.x as usize].pos;
        let b = mesh.verts[triangle.verts.y as usize].pos;
        let c = mesh.verts[triangle.verts.z as usize].pos;
        collinear([a, b, c])
    }).count();
    let start = Instant::now();
    let browser = mesh.to_triangle_buffer();
    let browser_nonfinite = browser.iter().filter(|v| !v.is_finite()).count();
    let browser_zero_normals = browser.chunks_exact(9)
        .filter(|v| v[3..6].iter().all(|&x| x == 0.)).count();
    let mut browser_degenerate = 0;
    for triangle in browser.chunks_exact(27) {
        let [a, b, c] = [0, 9, 18].map(|i| nalgebra_glm::DVec3::new(
            triangle[i] as f64, triangle[i + 1] as f64, triangle[i + 2] as f64));
        if [a, b, c].iter().all(|p| p.iter().all(|x| x.is_finite())) {
            browser_degenerate += usize::from(collinear([a, b, c]));
        }
    }
    if args[3] != "-" {
        mesh.save_stl(&args[3])?;
    }
    if let Some(path) = args.get(4) {
        std::fs::write(path, browser.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    }
    let export_ms = start.elapsed().as_secs_f64() * 1000.0;
    let report = WorkerReport { schema: 2, status: "ok", message: None, read_ms, parse_ms,
        triangulate_ms, export_ms, triangles: mesh.triangles.len(), vertices: mesh.verts.len(),
        faces: stats.num_faces, shells: stats.num_shells,
        completion: if stats.is_complete() { "complete" } else { "partial" },
        failures: stats.failures, degenerate_f64, browser_nonfinite,
        browser_triangles: browser.len() / 27, browser_degenerate, browser_zero_normals };
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_glm::DVec3;

    #[test]
    fn collinearity_does_not_round_away_thin_triangles() {
        let a = DVec3::zeros();
        let b = DVec3::new(1., 1. + f64::EPSILON, 0.);
        let c = DVec3::new(1. - f64::EPSILON, 1., 0.);
        assert_eq!((b - a).cross(&(c - a)), DVec3::zeros());
        assert!(!collinear([a, b, c]));
        assert!(collinear([a, b, b]));
        assert!(collinear([a, b, 2. * b]));
    }
}
