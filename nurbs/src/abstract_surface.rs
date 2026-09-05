use nalgebra_glm::{DVec2, DVec3};
use crate::VecF;

/// Trait for a curve which maps from 2D (uv) to 3D
/// This trait is implement for both Bezier and NURBS surfaces, and abstracts
/// over them in the [`SampledSurface`] `struct`
pub trait AbstractSurface {
    fn point(&self, uv: DVec2) -> DVec3;

    /// Low-level function to calculate a point with a basis function hint
    /// (used as an optimization when we're re-using basis functions)
    fn point_from_basis(&self, uspan: usize, Nu: &VecF,
                               vspan: usize, Nv: &VecF) -> DVec3;

    /// Cartesian control-hull bounds for one knot cell. Rational weights
    /// must be positive for the convex-hull property to apply.
    fn control_bounds(&self, spans: [usize; 2]) -> [DVec3; 2];

    fn derivs<const E: usize>(&self, uv: DVec2) -> Vec<Vec<DVec3>> {
        self.derivs_relative_to::<E>(uv, DVec3::zeros())
    }

    /// The zeroth entry is S(uv) - reference, evaluated before restoring
    /// world coordinates. Higher derivatives are translation invariant.
    fn derivs_relative_to<const E: usize>(&self, uv: DVec2, reference: DVec3) -> Vec<Vec<DVec3>>;

    /// Evaluate the specified knot cell, including its one-sided derivatives
    /// at cell boundaries. The parameter must lie in the closed cell.
    fn derivs_in_span<const E: usize>(&self, uv: DVec2, spans: [usize; 2], reference: DVec3) -> Vec<Vec<DVec3>>;
}
