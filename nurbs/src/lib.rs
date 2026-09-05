#![allow(non_snake_case)]
// This crate is translations of algorithms from the 70s, which use awkward
// single-character names everywhere, so we're matching their convention.

mod abstract_curve;
mod abstract_surface;
mod bspline_curve;
mod bspline_surface;
mod knot_vector;
mod nd_curve;
mod nd_surface;
mod nurbs_curve;
mod nurbs_surface;
mod sampled_curve;
mod sampled_surface;

use smallvec::{SmallVec};
type VecF = SmallVec<[f64; 8]>;

// Translate homogeneous controls by the Cartesian anchor before accumulating
// the quotient. A constant Cartesian coordinate then contributes exactly zero
// to every numerator derivative, even when weights vary.
fn rational_difference(p: nalgebra_glm::DVec4, origin: nalgebra_glm::DVec4) -> nalgebra_glm::DVec4 {
    let xyz = p.xyz() - (origin.xyz() / origin.w) * p.w;
    nalgebra_glm::DVec4::new(xyz.x, xyz.y, xyz.z, p.w - origin.w)
}

pub use crate::abstract_curve::AbstractCurve;
pub use crate::abstract_surface::AbstractSurface;
pub use crate::bspline_curve::BSplineCurve;
pub use crate::bspline_surface::BSplineSurface;
pub use crate::knot_vector::KnotVector;
pub use crate::nd_curve::NDBSplineCurve;
pub use crate::nd_surface::NDBSplineSurface;
pub use crate::nurbs_curve::NURBSCurve;
pub use crate::nurbs_surface::NURBSSurface;
pub use crate::sampled_curve::SampledCurve;
pub use crate::sampled_surface::SampledSurface;
