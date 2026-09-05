use std::cmp::min;
use nalgebra_glm::TVec;
use crate::KnotVector;

#[derive(Debug, Clone)]
pub struct NDBSplineCurve<const D: usize> {
    pub open: bool,
    pub knots: KnotVector,
    control_points: Vec<TVec<f64, D>>,
}

/// Abstract b-spline curve with N-dimensional control points
impl<const D: usize> NDBSplineCurve<D> {
    pub fn new(
        open: bool,
        knots: KnotVector,
        control_points: Vec<TVec<f64, D>>,
    ) -> Self {
        Self {
            open,
            knots,
            control_points,
        }
    }

    pub fn min_u(&self) -> f64 {
        self.knots.min_t()
    }
    pub fn max_u(&self) -> f64 {
        self.knots.max_t()
    }

    /// Sufficient structural test for a C1 periodic cut: repeated controls
    /// and a translated knot sequence, with simple knots at both cut ends.
    /// Geometric closedness alone does not imply a smooth periodic seam.
    pub fn has_smooth_periodic_seam(&self) -> bool {
        let p = self.knots.degree();
        let n = self.control_points.len();
        if p < 2 || n <= p
            || self.knots[p - 1] == self.min_u() || self.knots[p + 1] == self.min_u()
            || self.knots[n - 1] == self.max_u() || self.knots[n + 1] == self.max_u()
            || self.control_points[..p] != self.control_points[n - p..] {
            return false;
        }
        let period = self.max_u() - self.min_u();
        let shift = n - p;
        // The two outermost knots never enter an active basis function.
        (1..self.knots.len() - shift - 1).all(|i| {
            let (a, b) = (self.knots[i], self.knots[i + shift]);
            // Only uncertainty from representing/subtracting knot values;
            // no world-space or source-geometry tolerance is used.
            (b - a - period).abs() <= 4. * f64::EPSILON * (a.abs() + b.abs() + period.abs())
        })
    }

    /// Converts a point at position t onto the 3D line, using basis functions
    /// of order `p + 1` respectively.
    ///
    /// ALGORITHM A3.1
    pub fn curve_point(&self, u: f64) -> TVec<f64, D> {
        let (origin, point) = self.curve_point_relative(u, |p, origin| p - origin);
        point + origin
    }

    pub(crate) fn curve_point_relative(&self, u: f64,
        difference: impl Fn(TVec<f64, D>, TVec<f64, D>) -> TVec<f64, D>,
    ) -> (TVec<f64, D>, TVec<f64, D>) {
        let p = self.knots.degree();

        let span = self.knots.find_span(u);
        let N = self.knots.basis_funs_for_span(span, u);

        // Anchor at the dominant basis term so endpoint interpolation does
        // not subtract and re-add an unrelated, potentially large coordinate.
        let anchor = N.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let origin = self.control_points[span - p + anchor];
        let mut C = TVec::zeros();
        for i in 0..=p {
            C += N[i] * difference(self.control_points[span - p + i], origin)
        }
        (origin, C)
    }

    /// Computes the derivatives of the curve of order up to and including `d` at location `t`,
    /// using basis functions of order `p + 1` respectively.
    ///
    /// ALGORITHM A3.2
    pub fn curve_derivs<const E: usize>(&self, u: f64) -> Vec<TVec<f64, D>> {
        let (origin, mut derivatives) = self.curve_derivs_relative::<E>(u, |p, origin| p - origin);
        derivatives[0] += origin;
        derivatives
    }

    pub(crate) fn curve_derivs_relative<const E: usize>(&self, u: f64,
        difference: impl Fn(TVec<f64, D>, TVec<f64, D>) -> TVec<f64, D>,
    ) -> (TVec<f64, D>, Vec<TVec<f64, D>>) {
        let p = self.knots.degree();

        let du = min(E, p);

        let span = self.knots.find_span(u);
        let N_derivs = self.knots.basis_funs_derivs_for_span(span, u, du);

        // Partition of unity: a constant contributes only to the position,
        // never its derivatives. Evaluate local differences before summation
        // instead of cancelling large translated control coordinates.
        let anchor = N_derivs[0].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let origin = self.control_points[span - p + anchor];
        let mut CK = vec![TVec::zeros(); E + 1];
        for k in 0..=du {
            for j in 0..=p {
                CK[k] += N_derivs[k][j] * difference(self.control_points[span - p + j], origin)
            }
        }
        (origin, CK)
    }

    pub fn as_polyline(&self, u_start: f64, u_end: f64, num_points_per_knot: usize) -> Vec<TVec<f64, D>> {
        let (u_min, u_max) = if u_start < u_end {
            (u_start, u_end)
        } else {
            (u_end, u_start)
        };

        let mut result = vec![self.curve_point(u_min)];

        // TODO this could be faster if we skip to the right start/end sections

        assert!(num_points_per_knot > 0);
        for i in 0..self.knots.len() - 1 {
            // Skip multiple knots
            if self.knots[i] == self.knots[i + 1] {
                continue;
            }
            // Iterate over a grid within this region
            for u in 0..num_points_per_knot {
                let frac = (u as f64) / (num_points_per_knot as f64);
                let u = self.knots[i] * (1.0 - frac) + self.knots[i + 1] * frac;
                if u > u_min && u < u_max {
                    result.push(self.curve_point(u));
                }
            }
        }
        result.push(self.curve_point(u_max));

        if u_start > u_end {
            result.reverse();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_glm::DVec3;

    #[test]
    fn periodic_seam_requires_matching_controls_and_translated_knots() {
        let knots = [-2., -1., 0., 1., 2., 3., 4., 5., 6.];
        let controls = [DVec3::x(), DVec3::y(), -DVec3::x(), -DVec3::y(), DVec3::x(), DVec3::y()];
        let make = |knots: &[f64], controls: &[DVec3]| NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(2, knots, &[1; 9]), controls.to_vec());
        assert!(make(&knots, &controls).has_smooth_periodic_seam());
        let mut exterior = knots;
        exterior[0] = exterior[1];
        exterior[8] = exterior[7];
        let trimmed_exterior = make(&exterior, &controls);
        assert!(trimmed_exterior.has_smooth_periodic_seam());
        for i in 0..=16 {
            let u = i as f64 / 4.;
            assert_eq!(trimmed_exterior.curve_derivs::<2>(u),
                make(&knots, &controls).curve_derivs::<2>(u));
        }
        let mut different = controls;
        different[5].x += 1e-12;
        assert!(!make(&knots, &different).has_smooth_periodic_seam());
        let mut different = knots;
        different[7] += 1e-10;
        assert!(!make(&different, &controls).has_smooth_periodic_seam());
        let corner = NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            controls[..5].to_vec());
        assert!(!corner.has_smooth_periodic_seam());
    }

    #[test]
    fn clamped_endpoint_preserves_small_coordinates() {
        let end = DVec3::new(1e-30, 2e-30, 3e-30);
        let curve = NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            vec![DVec3::repeat(1.), DVec3::repeat(0.5), end]);
        assert_eq!(curve.curve_point(1.), end);
        assert_eq!(curve.curve_derivs::<2>(1.)[0], end);
    }

    #[test]
    fn constant_coordinates_have_exactly_zero_derivatives() {
        let curve = NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(3, &[0., 0.0396321528446033, 0.0396624224503112], &[4, 3, 4]),
            (0..7).map(|i| DVec3::new(i as f64, (i * i) as f64, 8.58999999999999)).collect());
        for i in 0..=32 {
            let u = curve.max_u() * i as f64 / 32.;
            assert_eq!(curve.curve_point(u).z, 8.58999999999999);
            let derivatives = curve.curve_derivs::<3>(u);
            assert_eq!(derivatives[0].z, 8.58999999999999);
            assert!(derivatives[1..].iter().all(|d| d.z == 0.));
        }
    }
}
