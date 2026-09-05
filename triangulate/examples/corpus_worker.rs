//! One process per model. The Python harness owns timeouts and repetition.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

static WARNINGS: AtomicUsize = AtomicUsize::new(0);
static ERRORS: AtomicUsize = AtomicUsize::new(0);

struct Logger(env_logger::Logger);
impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.0.enabled(metadata)
    }
    fn log(&self, record: &log::Record) {
        match record.level() {
            log::Level::Warn => {
                WARNINGS.fetch_add(1, Ordering::Relaxed);
            }
            log::Level::Error => {
                ERRORS.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        self.0.log(record);
    }
    fn flush(&self) {
        self.0.flush();
    }
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
    if args.len() != 4 {
        return Err("usage: corpus_worker INPUT.step METRICS.json OUTPUT.stl".into());
    }
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).build();
    let max_level = logger.filter().max(log::LevelFilter::Warn);
    log::set_boxed_logger(Box::new(Logger(logger)))?;
    // Always count warnings/errors, even when RUST_LOG suppresses their display.
    log::set_max_level(max_level);

    let start = Instant::now();
    let data = std::fs::read(&args[1])?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let flat = StepFile::strip_flatten(&data)?;
    let step = StepFile::parse(&flat)?;
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
    mesh.save_stl(&args[3])?;
    let export_ms = start.elapsed().as_secs_f64() * 1000.0;
    // Only numeric fields: strings and report serialization belong to the harness.
    std::fs::write(&args[2], format!(
        "{{\"read_ms\":{},\"parse_ms\":{},\"triangulate_ms\":{},\"export_ms\":{},\"triangles\":{},\"vertices\":{},\"faces\":{},\"shells\":{},\"errors\":{},\"panics\":{},\"log_warn\":{},\"log_error\":{},\"degenerate_f64\":{}}}",
        read_ms, parse_ms, triangulate_ms, export_ms, mesh.triangles.len(),
        mesh.verts.len(), stats.num_faces, stats.num_shells, stats.num_errors,
        stats.num_panics, WARNINGS.load(Ordering::Relaxed), ERRORS.load(Ordering::Relaxed),
        degenerate_f64,
    ))?;
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
