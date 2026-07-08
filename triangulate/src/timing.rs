//! Lightweight single-threaded phase timing accumulators for profiling.
//!
//! Zero-dependency: uses `std::time::Instant` (works natively and under
//! WASI). All state is thread-local, so with the `parallel` feature the
//! numbers are per-thread and `dump()` only reports the calling thread.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static ACC: RefCell<HashMap<&'static str, (f64, u64)>> = RefCell::new(HashMap::new());
}

/// Time a closure and accumulate under `name`.
///
/// On `wasm32-unknown-unknown` (browser builds) `std::time::Instant::now()`
/// panics, so timing is compiled out and the closure runs directly.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[inline(always)]
pub fn time<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn time<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    let start = std::time::Instant::now();
    let out = f();
    let dt = start.elapsed().as_secs_f64();
    ACC.with(|acc| {
        let mut acc = acc.borrow_mut();
        let e = acc.entry(name).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    });
    out
}

/// Return accumulated (name, seconds, calls), sorted by descending time.
pub fn snapshot() -> Vec<(&'static str, f64, u64)> {
    ACC.with(|acc| {
        let mut v: Vec<_> = acc
            .borrow()
            .iter()
            .map(|(k, (t, n))| (*k, *t, *n))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v
    })
}

/// Reset all accumulators.
pub fn reset() {
    ACC.with(|acc| acc.borrow_mut().clear());
}
