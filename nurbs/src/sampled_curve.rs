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
        for i in curve.knots.degree()..curve.knots.len() - 1 - curve.knots.degree() {
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
        let constrain = |u: f64| u.clamp(min, max);
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
                || (u == min && gradient >= 0.0) || (u == max && gradient <= 0.0) {
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
                let delta = candidate - u;
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

    pub fn is_closed(&self) -> bool {
        !self.curve.open
    }

    pub fn u_from_point(&self, p: DVec3) -> Option<f64> {
        use ordered_float::OrderedFloat;
        let best_u = self.samples.iter()
            .min_by_key(|(_u, pos)| OrderedFloat((pos - p).norm()))
            .unwrap().0;
        let alias = if !self.curve.open && best_u == self.min_u() { Some(self.max_u()) }
                    else if !self.curve.open && best_u == self.max_u() { Some(self.min_u()) }
                    else { None };
        std::iter::once(best_u).chain(alias)
            .filter_map(|seed| self.u_from_point_newtons_method(p, seed))
            .min_by_key(|&u| OrderedFloat((self.curve.point(u) - p).norm_squared()))
    }

    pub fn as_polyline(&self, ranges: &[(f64, f64)], num_points_per_knot: usize) -> Vec<DVec3> {
        assert!(num_points_per_knot > 0);
        let trim_length: f64 = ranges.iter().map(|&(a, b)| (b - a).abs()).sum();
        let smooth_seam = self.curve.has_smooth_periodic_seam();
        let knots = &self.curve.knots;
        let degree = knots.degree();
        // Ordered cells carry a sampling measure and whether their end is a
        // mandatory corner. Smooth knots and periodic cuts change the density,
        // not the sampling phase: they need not create tiny endpoint edges.
        let mut cells = Vec::new();
        for &(start, end) in ranges {
            let first = cells.len();
            for i in degree..knots.len() - degree - 1 {
                let a = knots[i].max(start.min(end));
                let b = knots[i + 1].min(start.max(end));
                if a >= b { continue; }
                let measure = (b - a) / (knots[i + 1] - knots[i]).min(trim_length);
                let cell = if start < end {
                    (a, b, measure, b == knots[i + 1] && knots[i + degree] == b)
                } else {
                    (b, a, measure, a == knots[i] && knots[i + 1 - degree] == a)
                };
                cells.push(cell);
            }
            if start > end { cells[first..].reverse(); }
            if let Some(last) = cells[first..].last_mut() { last.3 = !smooth_seam; }
        }
        let Some(last) = cells.last_mut() else {
            return ranges.first().map_or_else(Vec::new, |&(a, b)|
                vec![self.curve.point(a), self.curve.point(b)]);
        };
        last.3 = true;
        let mut result = vec![cells[0].0];
        for run in cells.split_inclusive(|cell| cell.3) {
            let measure: f64 = run.iter().map(|cell| cell.2).sum();
            let count = (measure * num_points_per_knot as f64).ceil() as usize;
            let mut cell = 0;
            let mut base = 0.;
            for i in 1..count {
                let sample = measure * (i as f64 / count as f64);
                while cell + 1 < run.len() && sample > base + run[cell].2 {
                    base += run[cell].2;
                    cell += 1;
                }
                let (a, b, width, _) = run[cell];
                let fraction = ((sample - base) / width).clamp(0., 1.);
                result.push(a * (1. - fraction) + b * fraction);
            }
            result.push(run.last().unwrap().1);
        }
        result.into_iter().map(|u| self.curve.point(u)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;

    #[test]
    fn closed_curve_projection_searches_both_bounded_seam_representatives() {
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            [(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 1e-12)].iter()
                .map(|&(x, y)| DVec3::new(x, y, 0.)).collect()));
        assert_eq!(curve.u_from_point(DVec3::new(-1e-12, 0., 0.)), Some(0.));
        assert!((curve.u_from_point(DVec3::new(0.01, 0., 0.)).unwrap() - 0.01).abs() < 1e-12);
        assert!((curve.u_from_point(DVec3::new(0., 0.01, 0.)).unwrap() - 3.99).abs() < 1e-12);
    }

    #[test]
    fn projection_samples_only_the_active_knot_domain() {
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(1, &[-1., 0., 1., 2., 3., 4.], &[1; 6]),
            vec![DVec3::zeros(), DVec3::x(), DVec3::y(), DVec3::zeros()]));
        assert!(curve.samples.iter().all(|&(u, _)| u >= curve.min_u() && u <= curve.max_u()));
    }

    #[test]
    fn smooth_periodic_cut_samples_remain_separated_from_trim_endpoints() {
        let controls = [(2., 1.), (2., 2.), (1., 2.), (1., 1.), (2., 1.), (2., 2.)];
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(2, &[-2., -1., 0., 1., 2., 3., 4., 5., 6.], &[1; 9]),
            controls.iter().map(|&(x,y)| DVec3::new(x,y,0.)).collect()));
        assert!(curve.curve.has_smooth_periodic_seam());
        for start in [1e-12, 4. - 1e-12] {
            for ranges in [[(start, 0.), (4., start)], [(start, 4.), (0., start)]] {
                let points = curve.as_polyline(&ranges, 8);
                assert_eq!(points[0], curve.curve.point(start));
                assert_eq!(points.first(), points.last());
                assert!(points.windows(2).all(|p|
                    p[0].map(|x| x as f32) != p[1].map(|x| x as f32)));
            }
        }
    }

    #[test]
    fn periodic_cut_does_not_multiply_sampling_density() {
        let corners = [DVec3::new(1., 1., 0.), DVec3::new(2., 1., 0.),
            DVec3::new(2., 2., 0.), DVec3::new(1., 2., 0.), DVec3::new(1., 1., 0.)];
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            corners.to_vec()));
        let start = 1e-6;
        let points = curve.as_polyline(&[(start, 0.), (4., start)], 8);
        assert_eq!(points[0], curve.curve.point(start));
        assert_eq!(points.first(), points.last());
        assert!(corners.iter().all(|corner| points.contains(corner)));
        assert!(points.len() <= curve.as_polyline(&[(4., 0.)], 8).len() + 1);
        assert!(points.windows(2).all(|p| p[0].map(|x| x as f32) != p[1].map(|x| x as f32)));
    }

    #[test]
    fn short_trims_sample_their_own_knot_interval() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            vec![DVec3::new(0., 0., 0.), DVec3::new(0.5, 0., 0.), DVec3::new(1., 1., 0.)]));
        let points = curve.as_polyline(&[(0.01, 0.02)], 8);
        assert_eq!(points.len(), 9);
        assert!((points[4] - DVec3::new(0.015, 0.015 * 0.015, 0.)).norm() < 1e-15);
        assert_eq!(curve.as_polyline(&[(0.02, 0.01)], 8), points.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn smooth_knots_do_not_restart_sampling_next_to_trim_endpoints() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 0.5, 1.], &[3, 1, 3]),
            vec![DVec3::zeros(), DVec3::new(0.25, 0., 0.),
                DVec3::new(0.75, 1., 0.), DVec3::new(1., 1., 0.)]));
        for (a, b) in [(0.5 - 1e-12, 1.), (1., 0.5 - 1e-12),
                       (0., 0.5 + 1e-12), (0.5 + 1e-12, 0.)] {
            let points = curve.as_polyline(&[(a, b)], 8);
            assert_eq!(points[0], curve.curve.point(a));
            assert_eq!(*points.last().unwrap(), curve.curve.point(b));
            assert!(points.len() >= 9);
            assert!(points.windows(2).all(|p|
                p[0].map(|x| x as f32) != p[1].map(|x| x as f32)));
        }
    }

    #[test]
    fn narrow_smooth_spans_retain_their_sampling_density() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(2, &[0., 1e-9, 1.], &[3, 1, 3]),
            vec![DVec3::zeros(), DVec3::new(0., 1., 0.),
                DVec3::new(1., 1., 0.), DVec3::new(1., 2., 0.)]));
        let points = curve.as_polyline(&[(0., 1.)], 8);
        assert_eq!(points.len(), 17);
        for i in 0..=8 {
            assert_eq!(points[i], curve.curve.point(1e-9 * i as f64 / 8.));
        }
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
    fn one_sample_per_span_keeps_knot_corners() {
        let curve = SampledCurve::new(NDBSplineCurve::new(true,
            KnotVector::from_multiplicities(1, &[0., 0.5, 1.], &[2, 1, 2]),
            vec![DVec3::new(0., 0., 0.), DVec3::new(1., 0., 0.), DVec3::new(1., 1., 0.)]));
        let points = curve.as_polyline(&[(0.25, 0.75)], 1);
        assert_eq!(points, vec![DVec3::new(0.5, 0., 0.), DVec3::new(1., 0., 0.), DVec3::new(1., 0.5, 0.)]);
        assert_eq!(curve.as_polyline(&[(0.75, 0.25)], 1), points.into_iter().rev().collect::<Vec<_>>());
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
