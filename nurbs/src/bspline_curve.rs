use crate::{abstract_curve::AbstractCurve, nd_curve::NDBSplineCurve};
use nalgebra_glm::DVec3;

pub type BSplineCurve = NDBSplineCurve<3>;

impl AbstractCurve for BSplineCurve {
    fn point(&self, u: f64) -> DVec3 {
        self.curve_point(u)
    }
    fn derivs<const E: usize>(&self, u: f64) -> Vec<DVec3> {
        self.curve_derivs::<E>(u)
    }
}
