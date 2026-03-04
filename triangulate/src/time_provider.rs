#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;
