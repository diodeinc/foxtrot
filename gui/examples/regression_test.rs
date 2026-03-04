use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::{App, Arg};
use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

static INFO_COUNT: AtomicUsize = AtomicUsize::new(0);
static WARN_COUNT: AtomicUsize = AtomicUsize::new(0);
static ERROR_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A logger that counts messages by level and forwards to env_logger.
struct CountingLogger {
    inner: env_logger::Logger,
}

impl log::Log for CountingLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        match record.level() {
            log::Level::Error => { ERROR_COUNT.fetch_add(1, Ordering::Relaxed); }
            log::Level::Warn => { WARN_COUNT.fetch_add(1, Ordering::Relaxed); }
            log::Level::Info => { INFO_COUNT.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
        self.inner.log(record);
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

fn reset_log_counts() -> (usize, usize, usize) {
    let info = INFO_COUNT.swap(0, Ordering::Relaxed);
    let warn = WARN_COUNT.swap(0, Ordering::Relaxed);
    let error = ERROR_COUNT.swap(0, Ordering::Relaxed);
    (info, warn, error)
}

struct FileResult {
    name: String,
    duration_ms: f64,
    triangles: usize,
    faces: usize,
    errors: usize,
    panics: usize,
    log_info: usize,
    log_warn: usize,
    log_error: usize,
}

fn run_tests(files: &[std::path::PathBuf]) -> Vec<FileResult> {
    let mut results = Vec::new();

    for path in files {
        let name = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        eprint!("Testing {}... ", name);
        let _ = reset_log_counts();
        let start = Instant::now();

        let result = std::panic::catch_unwind(|| {
            let data = std::fs::read(path).expect("Could not open file");
            let flat = StepFile::strip_flatten(&data);
            let step = StepFile::parse(&flat);
            let (mesh, stats) = triangulate(&step);
            (mesh.triangles.len(), stats)
        });

        let duration = start.elapsed();
        let (log_info, log_warn, log_error) = reset_log_counts();

        match result {
            Ok((tri_count, stats)) => {
                eprintln!("{:.1}ms, {} tris, {} faces, {} err, {} panic, log: {}/{}/{}",
                    duration.as_secs_f64() * 1000.0,
                    tri_count, stats.num_faces, stats.num_errors, stats.num_panics,
                    log_info, log_warn, log_error);
                results.push(FileResult {
                    name, duration_ms: duration.as_secs_f64() * 1000.0,
                    triangles: tri_count, faces: stats.num_faces,
                    errors: stats.num_errors, panics: stats.num_panics,
                    log_info, log_warn, log_error,
                });
            }
            Err(_) => {
                eprintln!("HARD PANIC!");
                results.push(FileResult {
                    name, duration_ms: duration.as_secs_f64() * 1000.0,
                    triangles: 0, faces: 0, errors: 0, panics: 1,
                    log_info, log_warn, log_error,
                });
            }
        }
    }
    results
}

fn print_table(results: &[FileResult]) {
    eprintln!("\n{}", "=".repeat(120));
    eprintln!("{:<40} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "File", "Time(ms)", "Tris", "Faces", "Errs", "Panics", "Info", "Warn", "Error");
    eprintln!("{}", "-".repeat(120));

    let (mut tt, mut tri, mut fc, mut er, mut pa) = (0.0, 0, 0, 0, 0);
    for r in results {
        eprintln!("{:<40} {:>8.1} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
            r.name, r.duration_ms, r.triangles, r.faces, r.errors, r.panics,
            r.log_info, r.log_warn, r.log_error);
        tt += r.duration_ms; tri += r.triangles; fc += r.faces;
        er += r.errors; pa += r.panics;
    }

    eprintln!("{}", "-".repeat(120));
    eprintln!("{:<40} {:>8.1} {:>8} {:>6} {:>6} {:>6}",
        "TOTAL", tt, tri, fc, er, pa);
    eprintln!("{}", "=".repeat(120));
    eprintln!("\n{} files tested, {} total panics, {} total errors",
        results.len(), pa, er);
}

/// Serialize results to a simple JSON format (no serde dependency).
fn results_to_json(results: &[FileResult]) -> String {
    let mut s = String::from("{\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str(&format!(
            "  \"{}\": {{\"duration_ms\": {:.2}, \"triangles\": {}, \"faces\": {}, \"errors\": {}, \"panics\": {}, \"log_warn\": {}, \"log_error\": {}}}",
            r.name, r.duration_ms, r.triangles, r.faces, r.errors, r.panics, r.log_warn, r.log_error
        ));
        if i + 1 < results.len() { s.push(','); }
        s.push('\n');
    }
    s.push('}');
    s
}

/// Minimal JSON object parser (no serde). Returns map of name -> (duration_ms, errors, panics, log_warn, log_error).
fn parse_baseline(json: &str) -> BTreeMap<String, (f64, usize, usize, usize, usize)> {
    let mut map = BTreeMap::new();
    // Parse entries like: "name": {"duration_ms": 1.0, ... "errors": 0, "panics": 0, "log_warn": 0, "log_error": 0}
    let mut rest = json.trim();
    if rest.starts_with('{') { rest = &rest[1..]; }

    while let Some(name_start) = rest.find('"') {
        rest = &rest[name_start + 1..];
        let name_end = rest.find('"').unwrap_or(0);
        let name = rest[..name_end].to_string();
        rest = &rest[name_end + 1..];

        // Find the inner object
        let obj_start = match rest.find('{') {
            Some(i) => i,
            None => break,
        };
        let obj_end = match rest[obj_start..].find('}') {
            Some(i) => obj_start + i + 1,
            None => break,
        };
        let obj = &rest[obj_start..obj_end];
        rest = &rest[obj_end..];

        let get_f64 = |key: &str| -> f64 {
            obj.find(key).and_then(|i| {
                let after = &obj[i + key.len()..];
                let after = after.trim_start_matches(|c: char| c == '"' || c == ':' || c == ' ');
                let end = after.find(|c: char| c != '.' && c != '-' && !c.is_ascii_digit()).unwrap_or(after.len());
                after[..end].parse().ok()
            }).unwrap_or(0.0)
        };
        let get_usize = |key: &str| -> usize { get_f64(key) as usize };

        map.insert(name, (
            get_f64("duration_ms"),
            get_usize("errors"),
            get_usize("panics"),
            get_usize("log_warn"),
            get_usize("log_error"),
        ));
    }
    map
}

fn main() {
    let matches = App::new("regression_test")
        .about("Tessellation regression test harness")
        .arg(Arg::with_name("save-baseline")
            .long("save-baseline")
            .takes_value(true)
            .help("Save results as JSON baseline to this path"))
        .arg(Arg::with_name("compare")
            .long("compare")
            .takes_value(true)
            .help("Compare against a JSON baseline file"))
        .arg(Arg::with_name("timing-threshold")
            .long("timing-threshold")
            .takes_value(true)
            .default_value("0.02")
            .help("Max allowed timing regression as fraction (default: 0.02 = 2%)"))
        .get_matches();

    // Set up the counting logger
    let inner = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    ).build();
    let max_level = inner.filter();
    log::set_boxed_logger(Box::new(CountingLogger { inner }))
        .expect("Failed to set logger");
    log::set_max_level(max_level);

    // Find test assets
    let patterns = &[
        "test_assets/*.step", "test_assets/*.STEP",
        "test_assets/*.stp", "test_assets/*.STP",
    ];
    let mut files: Vec<_> = patterns.iter()
        .flat_map(|p| glob::glob(p).expect("Failed to glob pattern"))
        .filter_map(|r| r.ok())
        .collect();
    files.sort();
    files.dedup();

    if files.is_empty() {
        eprintln!("No STEP files found in test_assets/. Run scripts/extract_test_assets.sh first.");
        std::process::exit(1);
    }

    eprintln!("Found {} STEP files in test_assets/\n", files.len());

    let results = run_tests(&files);
    print_table(&results);

    // Save baseline if requested
    if let Some(path) = matches.value_of("save-baseline") {
        let json = results_to_json(&results);
        std::fs::write(path, &json).expect("Failed to write baseline");
        eprintln!("\nBaseline saved to {}", path);
    }

    // Compare against baseline if requested
    let mut failed = false;
    if let Some(path) = matches.value_of("compare") {
        let threshold: f64 = matches.value_of("timing-threshold")
            .unwrap().parse().expect("Invalid timing threshold");
        let baseline_json = std::fs::read_to_string(path).expect("Failed to read baseline");
        let baseline = parse_baseline(&baseline_json);

        eprintln!("\n{}", "=".repeat(100));
        eprintln!("Comparing against baseline: {}", path);
        eprintln!("{}", "=".repeat(100));

        let mut timing_regressions = Vec::new();
        let mut count_regressions = Vec::new();

        for r in &results {
            if let Some(&(base_time, base_err, base_panic, base_warn, base_log_err)) = baseline.get(&r.name) {
                // Check error/warning count regressions (must not increase)
                if r.errors > base_err {
                    count_regressions.push(format!(
                        "  {} errors: {} -> {} (+{})", r.name, base_err, r.errors, r.errors - base_err));
                }
                if r.panics > base_panic {
                    count_regressions.push(format!(
                        "  {} panics: {} -> {} (+{})", r.name, base_panic, r.panics, r.panics - base_panic));
                }
                if r.log_warn > base_warn {
                    count_regressions.push(format!(
                        "  {} log_warn: {} -> {} (+{})", r.name, base_warn, r.log_warn, r.log_warn - base_warn));
                }
                if r.log_error > base_log_err {
                    count_regressions.push(format!(
                        "  {} log_error: {} -> {} (+{})", r.name, base_log_err, r.log_error, r.log_error - base_log_err));
                }

                // Check timing regression (total, not per-file, is checked below)
            }
        }

        // Check total timing regression
        let base_total: f64 = baseline.values().map(|v| v.0).sum();
        let cur_total: f64 = results.iter().map(|r| r.duration_ms).sum();
        let timing_delta = (cur_total - base_total) / base_total;

        eprintln!("\nTotal timing: {:.1}ms -> {:.1}ms ({:+.1}%)",
            base_total, cur_total, timing_delta * 100.0);

        if timing_delta > threshold {
            timing_regressions.push(format!(
                "  Total: {:.1}ms -> {:.1}ms ({:+.1}%, threshold: {:.1}%)",
                base_total, cur_total, timing_delta * 100.0, threshold * 100.0));
        }

        if !count_regressions.is_empty() {
            eprintln!("\nERROR/WARNING REGRESSIONS:");
            for r in &count_regressions { eprintln!("{}", r); }
            failed = true;
        }

        if !timing_regressions.is_empty() {
            eprintln!("\nTIMING REGRESSIONS:");
            for r in &timing_regressions { eprintln!("{}", r); }
            failed = true;
        }

        if count_regressions.is_empty() && timing_regressions.is_empty() {
            eprintln!("\nNo regressions detected.");
        }
    }

    if failed {
        eprintln!("\nFAILED: regressions detected");
        std::process::exit(1);
    }

    eprintln!("\nAll tests passed!");
}
