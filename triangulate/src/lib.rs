pub mod colored_mesh;
pub mod mesh;
pub mod stats;
pub mod surface;
pub mod triangulate;
pub mod curve;

#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Could not lower point to 2D for triangulation")]
    CouldNotLower,

    #[error("Invalid geometry: {0}")]
    InvalidGeometry(&'static str),

    #[error("Invalid STEP entity: {0}")]
    InvalidStepEntity(&'static str),

    #[error("Missing STEP field: {0}")]
    MissingStepField(&'static str),

    #[error("Could not invert transform: {0}")]
    SingularTransform(&'static str),

    #[error("Numeric conversion failed: {0}")]
    NumericConversion(&'static str),

    #[error("Triangulation panicked")]
    TriangulationPanic,

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
}
