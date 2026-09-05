//! Coarse phase-level profiling harness for STEP tessellation.
//!
//! Usage: cargo run --release --example profile -- <file.step>

use std::time::Instant;

/// Current max RSS in MB (native unix only; returns 0 elsewhere).
fn peak_rss_mb() -> f64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            // ru_maxrss is bytes on macOS, KB on Linux
            let scale = if cfg!(target_os = "macos") { 1e6 } else { 1e3 };
            return usage.ru_maxrss as f64 / scale;
        }
    }
    0.0
}

fn main() {
    env_logger::init();
    let path = std::env::args().nth(1).expect("usage: profile <file.step>");
    let data = std::fs::read(&path).expect("could not read input file");
    eprintln!("input: {} ({:.1} MB)", path, data.len() as f64 / 1e6);

    let t0 = Instant::now();
    let flat = step::step_file::StepFile::strip_flatten(&data)
        .expect("could not preprocess STEP file");
    let t1 = Instant::now();
    eprintln!("strip_flatten: {:.3}s ({:.1} MB flattened) [rss {:.0} MB]",
        (t1 - t0).as_secs_f64(), flat.len() as f64 / 1e6, peak_rss_mb());

    let parsed = step::step_file::StepFile::parse(&flat)
        .expect("could not parse STEP file");
    let t2 = Instant::now();
    eprintln!("parse:         {:.3}s ({} entities) [rss {:.0} MB]",
        (t2 - t1).as_secs_f64(), parsed.0.len(), peak_rss_mb());

    let (mesh, stats) = triangulate::triangulate::triangulate(&parsed);
    let t3 = Instant::now();
    eprintln!("triangulate:   {:.3}s ({} verts, {} tris; shells={} faces={} errors={} panics={}) [rss {:.0} MB]",
        (t3 - t2).as_secs_f64(), mesh.verts.len(), mesh.triangles.len(),
        stats.num_shells, stats.num_faces, stats.num_errors, stats.num_panics,
        peak_rss_mb());

    let tess = triangulate::colored_mesh::group_mesh_by_color(&mesh)
        .expect("group_mesh_by_color failed");
    let t4 = Instant::now();
    let total_pos: usize = tess.submeshes.iter().map(|s| s.positions.len()).sum();
    let total_idx: usize = tess.submeshes.iter().map(|s| s.indices.len()).sum();
    eprintln!("group_color:   {:.3}s ({} submeshes, {} positions, {} indices) [rss {:.0} MB]",
        (t4 - t3).as_secs_f64(), tess.submeshes.len(), total_pos, total_idx,
        peak_rss_mb());

    eprintln!("TOTAL:         {:.3}s", (t4 - t0).as_secs_f64());

    eprintln!("-- internal phase timings --");
    for (name, secs, calls) in triangulate::timing::snapshot() {
        eprintln!("{:32} {:8.3}s  ({} calls)", name, secs, calls);
    }

    // Quality metrics: per-color triangle count, total area, bbox.
    // Stable across vertex-ordering changes; written to stdout for diffing.
    if std::env::var("MESH_METRICS").is_ok() {
        let mut rows: Vec<String> = tess.submeshes.iter().map(|sm| {
            let mut area = 0.0f64;
            let mut bb_min = [f64::INFINITY; 3];
            let mut bb_max = [f64::NEG_INFINITY; 3];
            for tri in sm.indices.chunks(3) {
                let p = |i: usize| {
                    let v = sm.positions[tri[i] as usize];
                    [v[0] as f64, v[1] as f64, v[2] as f64]
                };
                let (a, b, c) = (p(0), p(1), p(2));
                for q in [&a, &b, &c] {
                    for k in 0..3 {
                        bb_min[k] = bb_min[k].min(q[k]);
                        bb_max[k] = bb_max[k].max(q[k]);
                    }
                }
                let u = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
                let v = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
                let cx = u[1]*v[2] - u[2]*v[1];
                let cy = u[2]*v[0] - u[0]*v[2];
                let cz = u[0]*v[1] - u[1]*v[0];
                area += 0.5 * (cx*cx + cy*cy + cz*cz).sqrt();
            }
            format!(
                "color=({:.3},{:.3},{:.3}) tris={} area={:.1} bbox=({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2})",
                sm.color[0], sm.color[1], sm.color[2],
                sm.indices.len() / 3, area,
                bb_min[0], bb_min[1], bb_min[2],
                bb_max[0], bb_max[1], bb_max[2],
            )
        }).collect();
        rows.sort();
        for r in rows {
            println!("{}", r);
        }
    }
}
