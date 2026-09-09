use log::{Level, LevelFilter, Log, Metadata, Record};
use serde_json::{Value, json};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default, Clone, Copy)]
struct Fox {
    models: u64,
    faces: u64,
    errors: u64,
    panics: u64,
    failed_instances: u64,
    skipped_instances: u64,
    call_ms: f64,
    calls: u64,
}
struct BenchLogger {
    fox: Mutex<Fox>,
    messages: Mutex<Vec<String>>,
}
static LOGGER: BenchLogger = BenchLogger {
    fox: Mutex::new(Fox {
        models: 0,
        faces: 0,
        errors: 0,
        panics: 0,
        failed_instances: 0,
        skipped_instances: 0,
        call_ms: 0.0,
        calls: 0,
    }),
    messages: Mutex::new(Vec::new()),
};

impl Log for BenchLogger {
    fn enabled(&self, m: &Metadata) -> bool {
        m.level() <= Level::Warn || m.target() == "editor_pcb3d_scene"
    }
    fn log(&self, r: &Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let message = r.args().to_string();
        if let Some(ms) = message.strip_prefix("foxtrot_bench_call_ms ") {
            let mut total = self.fox.lock().unwrap();
            total.call_ms += ms
                .parse::<f64>()
                .expect("Foxtrot call timing format changed");
            total.calls += 1;
        }
        if let Some(rest) = message.strip_prefix("foxtrot_bench_diagnostics ") {
            let mut event = Fox::default();
            for item in rest.split_whitespace() {
                let mut p = item.splitn(2, '=');
                let (Some(k), Some(v)) = (p.next(), p.next()) else {
                    continue;
                };
                let n = v.parse().expect("Foxtrot diagnostic format changed");
                match k {
                    "faces" => event.faces = n,
                    "errors" => event.errors = n,
                    "panics" => event.panics = n,
                    _ => {}
                }
            }
            event.models = 1;
            let mut total = self.fox.lock().unwrap();
            total.models += 1;
            total.faces += event.faces;
            total.errors += event.errors;
            total.panics += event.panics;
        }
        if message.starts_with("tessellation summary: ") {
            let words: Vec<_> = message.split_whitespace().collect();
            let mut total = self.fox.lock().unwrap();
            total.failed_instances = words[4].parse().expect("Diode summary format changed");
            total.skipped_instances = words[6].parse().expect("Diode summary format changed");
        }
        if r.level() <= Level::Warn || message.contains("failed") || message.contains("skipped") {
            let mut messages = self.messages.lock().unwrap();
            if messages.len() < 1000 {
                messages.push(format!("{} {}: {}", r.level(), r.target(), message));
            }
        }
        eprintln!("{} {}: {}", r.level(), r.target(), message);
    }
    fn flush(&self) {}
}

fn usage() -> (f64, Option<u64>) {
    #[cfg(unix)]
    unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut u) == 0 {
            let cpu = u.ru_utime.tv_sec as f64
                + u.ru_stime.tv_sec as f64
                + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) as f64 / 1e6;
            return (
                cpu * 1000.0,
                Some(u.ru_maxrss as u64 * if cfg!(target_os = "macos") { 1 } else { 1024 }),
            );
        }
    }
    (0.0, None)
}

fn stage<T>(name: &str, stages: &mut Vec<Value>, f: impl FnOnce() -> T) -> T {
    let wall = Instant::now();
    let cpu = usage().0;
    let value = f();
    let (cpu2, rss) = usage();
    stages.push(json!({"name":name,"wall_ms":wall.elapsed().as_secs_f64()*1000.0,"cpu_ms":cpu2-cpu,"peak_rss_bytes":rss}));
    value
}

fn run(input: &str) -> anyhow::Result<Value> {
    let total = Instant::now();
    let mut stages = Vec::new();
    let text = stage("read", &mut stages, || std::fs::read_to_string(input))?;
    let board = stage("parse_kicad_pcb", &mut stages, || {
        editor_kicad::parse_kicad_pcb(&text)
    })?;
    let instances = stage("placements_with_dnp", &mut stages, || {
        editor_kicad::placements_with_dnp(&board, true)
    });
    let scene = stage("prepare_scene", &mut stages, || {
        editor_pcb3d_scene::prepare_scene(&board, &instances, &text, false)
    })?;
    let serialized = stage("serialize_scene", &mut stages, || {
        editor_pcb3d_scene::serialize_scene(&scene)
    });
    let total_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut nonfinite = 0usize;
    let mut invalid = 0usize;
    let mut empty = 0usize;
    let mut nonfinite_instances = 0usize;
    let (mut vertices, mut triangles, mut all_instances) = (0usize, 0usize, 0usize);
    let (mut mesh_bytes, mut capacity_bytes) = (0usize, 0usize);
    stage("validate", &mut stages, || {
        for b in &scene.batches {
            vertices += b.vertices.len();
            triangles += b.indices.len() / 3;
            all_instances += b.instances.len();
            mesh_bytes += b.vertices.len() * std::mem::size_of::<editor_pcb3d_scene::Vertex>()
                + b.indices.len() * 4
                + b.instances.len() * std::mem::size_of::<editor_pcb3d_scene::InstanceRaw>();
            capacity_bytes += b.vertices.capacity()
                * std::mem::size_of::<editor_pcb3d_scene::Vertex>()
                + b.indices.capacity() * 4
                + b.instances.capacity() * std::mem::size_of::<editor_pcb3d_scene::InstanceRaw>();
            nonfinite += b
                .vertices
                .iter()
                .filter(|v| v.pos.iter().chain(v.normal.iter()).any(|x| !x.is_finite()))
                .count();
            nonfinite_instances += b
                .instances
                .iter()
                .filter(|i| {
                    [
                        i.model_0,
                        i.model_1,
                        i.model_2,
                        i.model_3,
                        i.color,
                        i.inst_flags,
                    ]
                    .iter()
                    .flatten()
                    .any(|x| !x.is_finite())
                })
                .count();
            invalid += b
                .indices
                .iter()
                .filter(|&&i| i as usize >= b.vertices.len())
                .count();
            invalid += usize::from(b.indices.len() % 3 != 0);
            if b.vertices.is_empty() || b.indices.is_empty() || b.instances.is_empty() {
                empty += 1;
            }
        }
    });
    let fox = *LOGGER.fox.lock().unwrap();
    let messages = LOGGER.messages.lock().unwrap().clone();
    let partial = fox.errors > 0 || fox.panics > 0 || fox.failed_instances > 0;
    Ok(json!({"schema":1,"total_ms":total_ms,"stages":stages,
      "counts":{"footprints":scene.stats.footprints,"pads":scene.stats.pads,"vias":scene.stats.vias,"board_drawings":scene.stats.board_drawings,
        "unique_models":scene.stats.unique_models,"total_instances":scene.stats.total_instances,"resolved_instances":scene.resolved_instances,
        "placeholder_instances":scene.placeholder_instances,"batches":scene.batches.len(),"vertices":vertices,"triangles":triangles,"instances":all_instances},
      "timings":{"embedded_extraction_secs":scene.stats.embed_extract_secs,"component_tessellation_secs":scene.stats.component_tess_secs,
        "board_geometry_secs":scene.stats.board_geom_secs,"total_preparation_secs":scene.stats.total_prep_secs,
        "foxtrot_call_secs":fox.call_ms / 1000.0},
      "storage":{"mesh_bytes":mesh_bytes,"mesh_capacity_bytes":capacity_bytes},"serialized_bytes":serialized.len(),
      "validation":{"nonfinite_vertices":nonfinite,"nonfinite_instances":nonfinite_instances,"invalid_indices":invalid,"empty_batches":empty},
      "foxtrot":{"models":fox.models,"faces":fox.faces,"errors":fox.errors,"panics":fox.panics,"failed_instances":fox.failed_instances,"skipped_instances":fox.skipped_instances,"calls":fox.calls},"logs":messages,
      "completion":if partial{"partial"}else if fox.skipped_instances>0{"missing_models"}else{"complete"},"status":"ok"}))
}

fn main() {
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Info))
        .ok();
    let args: Vec<String> = std::env::args().collect();
    let output = args.get(2).cloned();
    let report = if args.len() != 3 {
        json!({"schema":1,"status":"error","completion":"failed","error":"usage: board_worker INPUT.kicad_pcb OUTPUT.json"})
    } else {
        match run(&args[1]) {
            Ok(v) => v,
            Err(e) => {
                json!({"schema":1,"status":"error","completion":"failed","error":format!("{e:#}")})
            }
        }
    };
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(path) = output {
        if let Err(e) = std::fs::write(path, bytes) {
            eprintln!("failed to write report: {e}");
            std::process::exit(2);
        }
    } else {
        eprintln!("{}", String::from_utf8_lossy(&bytes));
    }
    if report["status"] != "ok" {
        std::process::exit(1);
    }
}
