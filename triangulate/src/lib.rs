pub mod colored_mesh;
pub mod curve;
pub mod mesh;
pub mod stats;
pub mod surface;
#[cfg(feature = "timeouts")]
mod time_provider;
pub mod triangulate;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Could not lower point to 2D for triangulation")]
    CouldNotLower,

    #[error("Could not convert into a Surface")]
    UnknownSurfaceType,

    #[error("Could not convert into a Curve")]
    UnknownCurveType,

    #[error("Closed NURBS and b-spline surfaces are not implemented")]
    ClosedSurface,

    #[error("Self-intersecting NURBS and b-spline surfaces are not implemented")]
    SelfIntersectingSurface,

    #[error("Closed NURBS and b-spline curves are not implemented")]
    ClosedCurve,

    #[error("Self-intersecting NURBS and b-spline curves are not implemented")]
    SelfIntersectingCurve,

    #[cfg(feature = "timeouts")]
    #[error("Tessellation timed out")]
    Timeout,

    #[cfg(feature = "timeouts")]
    #[error("Tessellation watchdog budget exceeded")]
    WatchdogBudgetExceeded,
}
