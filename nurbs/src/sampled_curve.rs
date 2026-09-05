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
            let derivs = self.curve.derivs::<2>(u);
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
            // Use distance curvature when it defines a descent direction;
            // otherwise retain Gauss--Newton. Work in derivative-normalized
            // coordinates and bound travel to one parameter domain.
            let inverse_speed = range / speed;
            let hessian = 1.0 + dot(&(derivs[2] * inverse_speed), &r) * inverse_speed;
            let metric = if hessian.is_finite() && hessian > 0.0 { hessian } else { 1.0 };
            let step = (-gradient / speed / metric).clamp(-1.0, 1.0) * range;
            // Test the full step, not a backtracked step: line-search failure
            // must not become success merely by halving until nothing moves.
            if u + step == u {
                return Some(u);
            }
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
            // Sample the trimmed span itself. Filtering a whole-span grid
            // can leave a short, curved trim with only its two endpoints.
            let a = self.curve.knots[i].max(u_min);
            let b = self.curve.knots[i + 1].min(u_max);
            if a >= b {
                continue;
            }
            // Iterate over a grid within this region
            for u in 0..num_points_per_knot {
                let frac = (u as f64) / (num_points_per_knot as f64);
                let u = a * (1.0 - frac) + b * frac;
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
    fn short_trims_sample_their_own_knot_interval() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            vec![DVec3::new(0., 0., 0.), DVec3::new(0.5, 0., 0.), DVec3::new(1., 1., 0.)]));
        let points = curve.as_polyline(0.01, 0.02, 8);
        assert_eq!(points.len(), 9);
        assert!((points[4] - DVec3::new(0.015, 0.015 * 0.015, 0.)).norm() < 1e-15);
        assert_eq!(curve.as_polyline(0.02, 0.01, 8), points.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn projection_uses_positive_distance_curvature_near_a_short_endpoint() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(3, &[0., 0.5, 1.], &[4, 1, 4]),
            vec![DVec3::new(5.48692594933471, 6.82772962686143, 2.32441744759478),
                 DVec3::new(5.47868936050619, 6.83705199546656, 2.33273309729622),
                 DVec3::new(5.46639462225168, 6.85150880217835, 2.3478102560178),
                 DVec3::new(5.4701140466715, 6.87809457408748, 2.38851300959099),
                 DVec3::new(5.47009428407746, 6.87803967799293, 2.38856983421543)]));
        let p = DVec3::new(5.47010288700796, 6.8780635750221, 2.3885957227281);
        let u = curve.u_from_point(p).unwrap();
        let d = curve.curve.derivs::<2>(u);
        let r = d[0] - p;
        assert!(u > 0.99 && u < 1.);
        assert!(dot(&r, &d[1]).abs() / d[1].norm() < 1e-12);
        assert!(d[1].norm_squared() + dot(&r, &d[2]) > 0.);
    }

    #[test]
    fn projection_stops_at_the_nearest_representable_parameter() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(1, &[-41.36699254603, -41.31699254603], &[2, 2]),
            vec![DVec3::new(0.5, 5.89, 0.05000000000001),
                 DVec3::new(0.5, 5.89, 3.552713678801e-15)]));
        let point = DVec3::new(0.5, 5.89, 0.05);
        let u = curve.u_from_point(point).unwrap();
        let error = (curve.curve.point(u) - point).norm_squared();
        assert!(u > curve.min_u(), "a resolvable first step must still be taken");
        assert!(error < 25e-30);
        for bits in [u.to_bits() - 1, u.to_bits() + 1] {
            assert!(error <= (curve.curve.point(f64::from_bits(bits)) - point).norm_squared());
        }
    }

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
