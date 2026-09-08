//! Single-process, single-thread browser-pipeline measurements. No mesh export.
//! Usage: profile INPUT.step OUTPUT.json

use serde_json::{json, Value};
use std::time::Instant;
use triangulate::mesh::{Triangle, Vertex};

fn usage() -> (f64, Option<u64>) {
    #[cfg(unix)]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            let cpu = usage.ru_utime.tv_sec as f64 + usage.ru_stime.tv_sec as f64
                + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 / 1e6;
            let bytes = usage.ru_maxrss as u64 * if cfg!(target_os = "macos") { 1 } else { 1024 };
            return (cpu * 1000., Some(bytes));
        }
    }
    (0., None)
}

fn stage<T>(name: &str, stages: &mut Vec<Value>, run: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let cpu = usage().0;
    let result = run();
    let (end_cpu, rss) = usage();
    stages.push(json!({"name": name, "wall_ms": start.elapsed().as_secs_f64() * 1000.,
        "cpu_ms": end_cpu - cpu, "peak_rss_bytes": rss}));
    result
}

fn profile(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    triangulate::timing::reset();
    let start = Instant::now();
    let mut stages = Vec::new();
    let data = stage("read", &mut stages, || std::fs::read(input))?;
    let flat = stage("flatten", &mut stages, || step::step_file::StepFile::strip_flatten(&data))
        .map_err(|e| e.to_string())?;
    let parsed = stage("parse", &mut stages, || step::step_file::StepFile::parse(&flat))
        .map_err(|e| e.to_string())?;
    let (mesh, stats) = stage("tessellate", &mut stages, || triangulate::triangulate::triangulate(&parsed));
    let browser = stage("browser_buffer", &mut stages, || mesh.to_triangle_buffer());
    let total_ms = start.elapsed().as_secs_f64() * 1000.;
    let phases: Vec<_> = triangulate::timing::snapshot().into_iter()
        .map(|(name, seconds, calls)| json!({"name": name, "seconds": seconds, "calls": calls}))
        .collect();
    let report = json!({
        "schema": 1, "total_ms": total_ms, "stages": stages, "phases": phases,
        "mesh": {"vertices": mesh.verts.len(), "triangles": mesh.triangles.len(),
            "faces": stats.num_faces, "shells": stats.num_shells,
            "completion": stats.completion(), "failures": stats.failures},
        "storage": {"input_bytes": data.len(), "flattened_bytes": flat.len(),
            "entities": parsed.0.len(),
            "mesh_bytes": mesh.verts.len() * std::mem::size_of::<Vertex>()
                + mesh.triangles.len() * std::mem::size_of::<Triangle>(),
            "mesh_capacity_bytes": mesh.verts.capacity() * std::mem::size_of::<Vertex>()
                + mesh.triangles.capacity() * std::mem::size_of::<Triangle>(),
            "browser_bytes": browser.len() * std::mem::size_of::<f32>()}
    });
    std::fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("usage: profile INPUT.step OUTPUT.json".into()); }
    env_logger::init();
    // Timing accumulators are thread-local. Run both tessellation and snapshot
    // on this worker, not on the caller outside Rayon's pool.
    rayon::ThreadPoolBuilder::new().num_threads(1).build()?.install(|| profile(&args[1], &args[2]))
}
