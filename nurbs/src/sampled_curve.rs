use nalgebra_glm::{dot, DVec3};
use crate::{abstract_curve::AbstractCurve, nd_curve::NDBSplineCurve};

#[derive(Debug)]
pub struct SampledCurve<const N: usize> {
    curve: NDBSplineCurve<N>,
    samples: Vec<(f64, DVec3)>,
}

impl<const N: usize> SampledCurve<N>
    where NDBSplineCurve<N>: AbstractCurve
{
    pub fn new(curve: NDBSplineCurve<N>) -> Self {
        const N: usize = 8;
        let mut samples = Vec::new();
        for i in 0..curve.knots.len() - 1 {
            // Skip multiple knots
            if curve.knots[i] == curve.knots[i + 1] {
                continue;
            }
            // Iterate over a grid within this region
            for u in 0..N {
                let frac = (u as f64) / (N as f64 - 1.0);
                let u = curve.knots[i] * (1.0 - frac) + curve.knots[i + 1] * frac;

                let q = curve.point(u);
                samples.push((u, q));
            }
        }

        Self { curve, samples }
    }

    // Section 6.1 (start middle page 232)
    pub fn u_from_point_newtons_method(&self, P: DVec3, u_0: f64) -> Option<f64> {
        const TOL: f64 = 64.0 * f64::EPSILON;
        let min = self.min_u();
        let max = self.max_u();
        let range = max - min;
        let constrain = |u: f64| {
            if self.curve.open { u.clamp(min, max) }
            else if u < min || u > max { min + (u - min).rem_euclid(range) }
            else { u }
        };
        let mut u = constrain(u_0);
        for _ in 0..256 {
            let derivs = self.curve.derivs::<1>(u);
            let r = derivs[0] - P;
            let tangent = derivs[1] * range;
            let speed = tangent.norm();
            let unit = if speed > 0.0 { tangent / speed } else { DVec3::zeros() };
            let gradient = dot(&r, &unit);
            let position_scale = derivs[0].abs() + P.abs();
            // First-order stationarity, not a fixed world-space distance or
            // a signed cosine test. A normal offset cannot mask tangential error.
            if gradient.abs() <= TOL * (speed + dot(&unit.abs(), &position_scale))
                || (self.curve.open
                    && ((u == min && gradient >= 0.0) || (u == max && gradient <= 0.0))) {
                return Some(u);
            }
            // Derivative-scaled Gauss--Newton is a descent direction even
            // where the squared-distance Hessian is negative.
            let step = -gradient / speed * range;
            let mut alpha = 1.0;
            let mut accepted = None;
            for _ in 0..40 {
                let candidate = constrain(u + alpha * step);
                let candidate_r = self.curve.point(candidate) - P;
                let delta = if self.curve.open { candidate - u } else { alpha * step };
                let slope = gradient * speed * (delta / range);
                let change = 0.5 * dot(&(candidate_r - r), &(candidate_r + r));
                let roundoff = TOL * dot(&(candidate_r.abs() + r.abs()), &position_scale);
                if slope < 0.0 && change <= 1e-4 * slope + roundoff {
                    accepted = Some(candidate);
                    break;
                }
                alpha *= 0.5;
            }
            u = accepted?;
        }
        None
    }

    pub fn min_u(&self) -> f64 {
        self.curve.min_u()
    }

    pub fn max_u(&self) -> f64 {
        self.curve.max_u()
    }

    pub fn u_from_point(&self, p: DVec3) -> Option<f64> {
        use ordered_float::OrderedFloat;
        let best_u = self.samples.iter()
            .min_by_key(|(_u, pos)| OrderedFloat((pos - p).norm()))
            .unwrap().0;
        self.u_from_point_newtons_method(p, best_u)
    }

    pub fn as_polyline(&self, u_start: f64, u_end: f64, num_points_per_knot: usize) -> Vec<DVec3> {
        assert!(num_points_per_knot > 0);
        // A degree-one span is exactly a line segment, including rational
        // spans. Keep its knots, but do not manufacture redundant samples.
        let num_points_per_knot = if self.curve.knots.degree() == 1 { 1 } else { num_points_per_knot };
        let (u_min, u_max) = if u_start < u_end {
            (u_start, u_end)
        } else {
            (u_end, u_start)
        };

        let mut result = vec![self.curve.point(u_min)];

        // TODO this could be faster if we skip to the right start/end sections

        for i in 0..self.curve.knots.len() - 1 {
            // Skip multiple knots
            if self.curve.knots[i] == self.curve.knots[i + 1] {
                continue;
            }
            // Iterate over a grid within this region
            for u in 0..num_points_per_knot {
                let frac = (u as f64) / (num_points_per_knot as f64);
                let u = self.curve.knots[i] * (1.0 - frac) + self.curve.knots[i + 1] * frac;
                if u > u_min && u < u_max {
                    result.push(self.curve.point(u));
                }
            }
        }
        result.push(self.curve.point(u_max));

        if u_start > u_end {
            result.reverse();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;

    #[test]
    fn linear_spans_keep_corners_without_redundant_samples() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(1, &[0., 0.5, 1.], &[2, 1, 2]),
            vec![DVec3::new(0., 0., 0.), DVec3::new(1., 0., 0.), DVec3::new(1., 1., 0.)]));
        let points = curve.as_polyline(0.25, 0.75, 8);
        assert_eq!(points, vec![DVec3::new(0.5, 0., 0.), DVec3::new(1., 0., 0.), DVec3::new(1., 0.5, 0.)]);
        assert_eq!(curve.as_polyline(0.75, 0.25, 8), points.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn short_curves_resolve_parameters_instead_of_returning_the_nearest_sample() {
        for size in [1e-12, 1e-3, 1.0, 1e6] {
            let curve = SampledCurve::new(NDBSplineCurve::new(true,
                KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
                vec![DVec3::zeros(), DVec3::new(size, 0., 0.)]));
            for parameter in [0.123456, 0.50001, 0.50002, 0.99999] {
                for normal_offset in [0., 100.] {
                    let p = DVec3::new(size * parameter, normal_offset, 0.);
                    let actual = curve.u_from_point(p).unwrap();
                    assert!((actual - parameter).abs() < 1e-13);
                }
            }
            assert_eq!(curve.u_from_point(DVec3::new(-size, 0., 0.)), Some(0.));
            assert_eq!(curve.u_from_point(DVec3::new(2. * size, 0., 0.)), Some(1.));
        }
    }

    #[test]
    fn curved_projection_converges_at_a_normal_offset_in_different_knot_units() {
        for domain in [[0., 1.], [0., 1e-8], [100., 200.]] {
            let curve = SampledCurve::new(NDBSplineCurve::new(true,
                KnotVector::from_multiplicities(2, &domain, &[3, 3]),
                vec![DVec3::new(0., 0., 0.), DVec3::new(0.5, 0., 0.), DVec3::new(1., 1., 0.)]));
            // C(t)=(t,t²,0); this small normal offset retains a unique minimum.
            let t = 0.31;
            let p = DVec3::new(t, t * t, 0.) + 0.02 * DVec3::new(-2. * t, 1., 0.);
            let actual = curve.u_from_point(p).unwrap();
            assert!(((actual - domain[0]) / (domain[1] - domain[0]) - t).abs() < 1e-12);
        }
    }
}
