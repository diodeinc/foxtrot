use nalgebra_glm::{DVec3};
use crate::{nd_curve::NDBSplineCurve, abstract_curve::AbstractCurve};

pub type NURBSCurve = NDBSplineCurve<4>;

impl AbstractCurve for NURBSCurve {
    /// Converts a point at position t onto the 3D line, using basis functions
    /// of order `p + 1` respectively.
    ///
    /// ALGORITHM A4.1
    fn point(&self, u: f64) -> DVec3 {
        let (origin, p) = self.curve_point_relative(u, crate::rational_difference);
        p.xyz() / (p.w + origin.w) + origin.xyz() / origin.w
    }

    /// Computes the derivatives of the curve of order up to and including `d` at location `t`,
    /// using basis functions of order `p + 1` respectively.
    ///
    /// ALGORITHM A4.2
    fn derivs<const E: usize>(&self, u: f64) -> Vec<DVec3> {
        let (origin, mut derivs) = self.curve_derivs_relative::<E>(u, crate::rational_difference);
        derivs[0].w += origin.w;
        let mut CK = vec![DVec3::zeros(); E + 1];
        for k in 0..=E {
            let mut v = derivs[k].xyz();
            for i in 1..=k {
                let b = num_integer::binomial(k, i);
                v -= b as f64 * derivs[i].w * CK[k - i];
            }
            CK[k] = v / derivs[0].w;
        }
        CK[0] += origin.xyz() / origin.w;
        CK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;
    use nalgebra_glm::DVec4;

    #[test]
    fn rational_constant_coordinates_have_zero_derivatives() {
        let z = 1.849999999971;
        let curve = NURBSCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            [1., 0.5, 1.].iter().enumerate().map(|(i, &w)|
                DVec4::new(i as f64 * 1e-16, 0., z, 1.) * w).collect());
        for i in 0..=32 {
            let u = i as f64 / 32.;
            assert_eq!(curve.point(u).z, z);
            let d = curve.derivs::<3>(u);
            assert_eq!(d[0].z, z);
            assert!(d[1..].iter().all(|d| d.z == 0.));
        }
    }

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
