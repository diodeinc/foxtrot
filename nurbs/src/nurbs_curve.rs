use nalgebra_glm::{DVec3};
use crate::{nd_curve::NDBSplineCurve, abstract_curve::AbstractCurve};

pub type NURBSCurve = NDBSplineCurve<4>;

impl AbstractCurve for NURBSCurve {
    /// Converts a point at position t onto the 3D line, using basis functions
    /// of order `p + 1` respectively.
    ///
    /// ALGORITHM A4.1
    fn point(&self, u: f64) -> DVec3 {
        let p = self.curve_point(u);
        p.xyz() / p.w
    }

    /// Computes the derivatives of the curve of order up to and including `d` at location `t`,
    /// using basis functions of order `p + 1` respectively.
    ///
    /// ALGORITHM A4.2
    fn derivs<const E: usize>(&self, u: f64) -> Vec<DVec3> {
        let derivs = self.curve_derivs::<E>(u);
        let mut CK = vec![DVec3::zeros(); E + 1];
        for k in 0..=E {
            let mut v = derivs[k].xyz();
            for i in 1..=k {
                let b = num_integer::binomial(k, i);
                v -= b as f64 * derivs[i].w * CK[k - i];
            }
            CK[k] = v / derivs[0].w;
        }
        CK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;
    use nalgebra_glm::DVec4;

    #[test]
    fn quotient_derivatives_use_each_lower_order() {
        // x(u) = u / (1 + u²): the quadratic weight derivative is nonzero.
        let curve = NURBSCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            vec![DVec4::new(0., 0., 0., 1.), DVec4::new(0.5, 0., 0., 1.),
                 DVec4::new(1., 0., 0., 2.)]);
        for u in [0., 0.3, 1.] {
            let d = curve.derivs::<2>(u);
            let w = 1. + u * u;
            assert!((d[0].x - u / w).abs() < 1e-14);
            assert!((d[1].x - (1. - u * u) / (w * w)).abs() < 1e-14);
            assert!((d[2].x - 2. * u * (u * u - 3.) / (w * w * w)).abs() < 1e-14);
        }
    }
}
