use nalgebra_glm::{DVec3};
use crate::{nd_curve::NDBSplineCurve, abstract_curve::AbstractCurve};

pub type BSplineCurve = NDBSplineCurve<3>;

impl AbstractCurve for BSplineCurve {
    fn point(&self, u: f64) -> DVec3 {
        self.curve_point(u)
    }
    fn derivs<const E: usize>(&self, u: f64) -> Vec<DVec3> {
        self.curve_derivs::<E>(u)
    }

    fn derivs_in_span<const E: usize>(&self, u: f64, span: usize) -> Vec<DVec3> {
        let (origin, mut derivatives) = self.curve_derivs_relative::<E>(u, span, |p, origin| p - origin);
        derivatives[0] += origin;
        derivatives
    }
}
