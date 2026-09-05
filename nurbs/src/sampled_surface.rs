use crate::{abstract_surface::AbstractSurface, nd_surface::NDBSplineSurface};
use log::error;
use nalgebra_glm::{dot, DVec2, DVec3};

#[derive(Debug, Clone)]
pub struct SampledSurface<const N: usize> {
    pub surf: NDBSplineSurface<N>,
    samples: Vec<(DVec2, DVec3)>,
    /// Sample indices arranged as an implicit kd-tree over the 3D sample
    /// positions (median at the middle of each range, axis = depth % 3).
    kd: Vec<u32>,
}

/// Squared distance matching `(a - b).norm_squared()` term order, so kd-tree
/// lookups compute bit-identical distances to the previous linear scan.
fn dist2(a: DVec3, b: DVec3) -> f64 {
    let d = a - b;
    d.x * d.x + d.y * d.y + d.z * d.z
}

fn build_kd(samples: &[(DVec2, DVec3)], idx: &mut [u32], depth: usize) {
    if idx.len() <= 1 {
        return;
    }
    let axis = depth % 3;
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        samples[a as usize].1[axis]
            .partial_cmp(&samples[b as usize].1[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (lo, rest) = idx.split_at_mut(mid);
    let (_, hi) = rest.split_at_mut(1);
    build_kd(samples, lo, depth + 1);
    build_kd(samples, hi, depth + 1);
}

/// Exact nearest-neighbor query. Ties on squared distance are broken by the
/// lowest sample index, matching what a linear `min_by_key` scan returns.
fn kd_nearest(
    samples: &[(DVec2, DVec3)],
    idx: &[u32],
    depth: usize,
    p: DVec3,
    best: &mut (f64, u32),
) {
    if idx.is_empty() {
        return;
    }
    let mid = idx.len() / 2;
    let si = idx[mid];
    let pos = samples[si as usize].1;
    let d2 = dist2(pos, p);
    if (d2, si) < *best {
        *best = (d2, si);
    }
    let axis = depth % 3;
    let delta = p[axis] - pos[axis];
    let (near, far) = if delta < 0.0 {
        (&idx[..mid], &idx[mid + 1..])
    } else {
        (&idx[mid + 1..], &idx[..mid])
    };
    kd_nearest(samples, near, depth + 1, p, best);
    // `<=` so an equal-distance, lower-index sample across the splitting
    // plane is still visited (tie-break correctness).
    if delta * delta <= best.0 {
        kd_nearest(samples, far, depth + 1, p, best);
    }
}

impl<const N: usize> SampledSurface<N>
where
    NDBSplineSurface<N>: AbstractSurface,
{
    pub fn new(surf: NDBSplineSurface<N>) -> Self {
        const N: usize = 8;
        let mut samples = Vec::new();
        for i in surf.u_knots.degree()..surf.u_knots.len() - 1 - surf.u_knots.degree() {
            // Skip multiple knots
            if surf.u_knots[i] == surf.u_knots[i + 1] {
                continue;
            }
            for j in surf.v_knots.degree()..surf.v_knots.len() - 1 - surf.v_knots.degree() {
                if surf.v_knots[j] == surf.v_knots[j + 1] {
                    continue;
                }
                // Iterate over a grid within this region
                for u in 0..N {
                    let frac = (u as f64) / (N as f64 - 1.0);
                    let u = surf.u_knots[i] * (1.0 - frac) + surf.u_knots[i + 1] * frac;

                    // Cache the u basis function outside the loop
                    let u_span = surf.u_knots.find_span(u);
                    let u_basis = surf.u_knots.basis_funs_for_span(u_span, u);
                    for v in 0..N {
                        let frac = (v as f64) / (N as f64 - 1.0);
                        let v = surf.v_knots[j] * (1.0 - frac) + surf.v_knots[j + 1] * frac;
                        let uv = DVec2::new(u, v);

                        let v_span = surf.v_knots.find_span(v);
                        let v_basis = surf.v_knots.basis_funs_for_span(v_span, v);
                        let q = surf.point_from_basis(u_span, &u_basis, v_span, &v_basis);
                        samples.push((uv, q));
                    }
                }
            }
        }
        let mut kd: Vec<u32> = (0..samples.len() as u32).collect();
        build_kd(&samples, &mut kd, 0);
        Self { surf, samples, kd }
    }

    // Section 6.1 (start middle page 232)
    pub fn uv_from_point_newtons_method(&self, P: DVec3, uv_0: DVec2) -> Option<DVec2> {
        let out = self.newtons_method_inner(P, uv_0, 256);
        if out.is_none() {
            error!("Could not find UV coordinates");
        }
        out
    }

    fn constrain_uv(&self, uv: DVec2) -> DVec2 {
        // Closure describes geometry, not the domain of the projection solve.
        // Both knot endpoints are eligible constrained minima, also at seams.
        DVec2::new(uv.x.clamp(self.surf.min_u(), self.surf.max_u()),
                   uv.y.clamp(self.surf.min_v(), self.surf.max_v()))
    }

    fn newtons_method_inner(&self, P: DVec3, uv_0: DVec2, max_iter: usize) -> Option<DVec2> {
        // Work in unit-domain coordinates, then normalize each Jacobian
        // column.  Thus neither knot magnitudes nor the physical units of the
        // model determine conditioning or convergence.
        const STATIONARITY_TOL: f64 = 64.0 * f64::EPSILON;
        const DAMPING: f64 = 1e-12;
        const ARMIJO: f64 = 1e-4;
        let mut uv_i = self.constrain_uv(uv_0);
        for _ in 0..max_iter {
            let derivs = self.surf.derivs::<2>(uv_i);
            let S = derivs[0][0];
            let r = S - P;
            let ranges = DVec2::new(
                self.surf.max_u() - self.surf.min_u(),
                self.surf.max_v() - self.surf.min_v(),
            );
            let columns = [derivs[1][0] * ranges.x, derivs[0][1] * ranges.y];
            let norms = DVec2::new(columns[0].norm(), columns[1].norm());
            let unit = [
                if norms.x > 0.0 {
                    columns[0] / norms.x
                } else {
                    DVec3::zeros()
                },
                if norms.y > 0.0 {
                    columns[1] / norms.y
                } else {
                    DVec3::zeros()
                },
            ];
            let gradient = DVec2::new(dot(&r, &unit[0]), dot(&r, &unit[1]));
            let mut projected = gradient;
            let mut free = [true, true];
            let domains = [
                (self.surf.min_u(), self.surf.max_u()),
                (self.surf.min_v(), self.surf.max_v()),
            ];
            for i in 0..2 {
                let (min, max) = domains[i];
                if (uv_i[i] == min && gradient[i] >= 0.0)
                    || (uv_i[i] == max && gradient[i] <= 0.0)
                {
                    projected[i] = 0.0;
                    free[i] = false;
                }
            }
            // Test each parameter direction at its own scale. A large normal
            // offset or a long u direction must not erase a resolvable thin v
            // direction. Position roundoff is likewise componentwise.
            let position_scale = S.abs() + P.abs();
            if (0..2).all(|i| {
                projected[i].abs()
                    <= STATIONARITY_TOL * (norms[i] + dot(&unit[i].abs(), &position_scale))
            }) {
                return Some(uv_i);
            }

            // The squared-distance Hessian includes surface curvature even
            // when the closest point has a nonzero normal residual. Omitting
            // it makes convergence arbitrarily slow near a curvature center.
            let fit_unit = [
                if free[0] { unit[0] } else { DVec3::zeros() },
                if free[1] { unit[1] } else { DVec3::zeros() },
            ];
            let curvature = |i: usize, j: usize, derivative: DVec3| {
                if free[i] && free[j] && norms[i] > 0. && norms[j] > 0. {
                    dot(&r, &(derivative * (ranges[i] / norms[i]) * (ranges[j] / norms[j])))
                } else { 0. }
            };
            let mut a = fit_unit[0].norm_squared() + curvature(0, 0, derivs[2][0]);
            let mut d = fit_unit[1].norm_squared() + curvature(1, 1, derivs[0][2]);
            let mut correlation = dot(&fit_unit[0], &fit_unit[1]) + curvature(0, 1, derivs[1][1]);
            let scale = a.abs().max(d.abs()).max(correlation.abs()).max(1.);
            a /= scale;
            d /= scale;
            correlation /= scale;
            // Shift the spectrum only as needed for a positive definite
            // descent model, including singular and negative-curvature cases.
            let eigen_min = 0.5 * (a + d) - (0.5 * (a - d)).hypot(correlation);
            let shift = (DAMPING - eigen_min).max(0.);
            a += shift;
            d += shift;
            let det = a * d - correlation * correlation;
            let projected = projected / scale;
            let q = DVec2::new(
                (-d * projected.x + correlation * projected.y) / det,
                (correlation * projected.x - a * projected.y) / det,
            );
            let normalized_step = DVec2::new(
                if norms.x > 0.0 { q.x / norms.x } else { 0.0 },
                if norms.y > 0.0 { q.y / norms.y } else { 0.0 },
            );
            let uv_step = DVec2::new(normalized_step.x * ranges.x, normalized_step.y * ranges.y);
            // As with curve projection, only the full Newton step may
            // establish representability convergence, never a backtracked one.
            if uv_i + uv_step == uv_i {
                return Some(uv_i);
            }
            // A shifted indefinite Hessian can request arbitrarily large
            // travel. Start line search within one normalized domain rather
            // than spending its iteration budget shrinking an unbounded step.
            let mut alpha = 1.0 / normalized_step.amax().max(1.0);
            let mut accepted = None;
            for _ in 0..40 {
                let candidate = self.constrain_uv(uv_i + alpha * uv_step);
                let candidate_r = self.surf.point(candidate) - P;
                let mut actual_q = DVec2::zeros();
                for i in 0..2 {
                    actual_q[i] = norms[i] * (candidate[i] - uv_i[i]) / ranges[i];
                }
                let slope = dot(&gradient, &actual_q);
                // Subtract squared distances in factored form. Near a normal
                // projection their difference is below the rounding error of
                // either squared norm; account for position evaluation error
                // rather than stalling before first-order convergence.
                let change = 0.5 * dot(&(candidate_r - r), &(candidate_r + r));
                let roundoff = STATIONARITY_TOL
                    * dot(&(candidate_r.abs() + r.abs()), &position_scale);
                if slope < 0.0 && change <= ARMIJO * slope + roundoff {
                    accepted = Some(candidate);
                    break;
                }
                alpha *= 0.5;
            }
            uv_i = accepted?;
        }
        None
    }

    pub fn uv_from_point(&self, p: DVec3) -> Option<DVec2> {
        assert!(!self.samples.is_empty());
        let mut best = (f64::INFINITY, u32::MAX);
        kd_nearest(&self.samples, &self.kd, 0, p, &mut best);
        let best_idx = if best.1 == u32::MAX {
            0
        } else {
            best.1 as usize
        };
        let best_uv = self.samples[best_idx].0;
        // A closed boundary has two parameter representatives. The nearest
        // geometric sample cannot distinguish them, so solve each bounded
        // representative rather than wrapping iterates across the cut.
        let mut seeds = vec![best_uv];
        for (i, (min, max, open)) in [
            (self.surf.min_u(), self.surf.max_u(), self.surf.u_open),
            (self.surf.min_v(), self.surf.max_v(), self.surf.v_open),
        ].iter().copied().enumerate() {
            if !open && (best_uv[i] == min || best_uv[i] == max) {
                let other = if best_uv[i] == min { max } else { min };
                for j in 0..seeds.len() {
                    let mut alias = seeds[j];
                    alias[i] = other;
                    seeds.push(alias);
                }
            }
        }
        let result = seeds.into_iter()
            .filter_map(|seed| self.newtons_method_inner(p, seed, 256))
            .min_by_key(|&uv| ordered_float::OrderedFloat((self.surf.point(uv) - p).norm_squared()));
        if result.is_none() {
            error!("Could not find UV coordinates");
        }
        result
    }

    // NOTE: do not add warm-start ("hint") seeding from an adjacent contour
    // vertex here. On degenerate patches (e.g. slivers whose whole u-range
    // moves the 3D point by less than the convergence tolerance) a hint seed
    // converges without moving, collapsing distinct contour vertices onto one
    // UV and producing broken CDT input. Every vertex must seed from its own
    // nearest sample so results stay well-defined on such surfaces.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;

    fn plane(
        u_domain: [f64; 2],
        v_domain: [f64; 2],
        origin: DVec3,
        u_edge: DVec3,
        v_edge: DVec3,
    ) -> SampledSurface<3> {
        SampledSurface::new(NDBSplineSurface::new(
            true,
            true,
            KnotVector::from_multiplicities(1, &u_domain, &[2, 2]),
            KnotVector::from_multiplicities(1, &v_domain, &[2, 2]),
            vec![
                vec![origin, origin + v_edge],
                vec![origin + u_edge, origin + u_edge + v_edge],
            ],
        ))
    }

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "{} != {}", a, b);
    }

    #[test]
    fn negative_curvature_projection_bounds_the_trial_step_to_the_domain() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(0., 0.), (0.5, 0.), (1., 10000.)].iter().map(|&(x, y)|
                vec![DVec3::new(x, y, 0.), DVec3::new(x, y, 1.)]).collect()));
        let p = DVec3::new(0.001, 0.0001, 0.5);
        let uv = sampled.uv_from_point_newtons_method(p, DVec2::new(0., 0.5)).unwrap();
        let d = sampled.surf.derivs::<1>(uv);
        let r = d[0][0] - p;
        assert!(uv.x > 0. && uv.x < 0.001);
        assert!(r.norm() < (sampled.surf.point(DVec2::new(0., 0.5)) - p).norm());
        assert!(r.dot(&d[1][0]).abs() / d[1][0].norm() < 1e-12);
    }

    #[test]
    fn projection_stops_at_the_nearest_representable_parameter() {
        let sampled = plane([-41.36699254603, -41.31699254603], [0., 1.],
            DVec3::new(0.5, 5.89, 0.05000000000001),
            DVec3::new(0., 0., 3.552713678801e-15 - 0.05000000000001),
            DVec3::new(1., 0., 0.));
        let p = DVec3::new(0.75, 5.89, 0.05);
        let uv = sampled.uv_from_point(p).unwrap();
        assert!(uv.x > sampled.surf.min_u());
        let error = (sampled.surf.point(uv) - p).norm_squared();
        for bits in [uv.x.to_bits() - 1, uv.x.to_bits() + 1] {
            let neighbor = DVec2::new(f64::from_bits(bits), uv.y);
            assert!(error <= (sampled.surf.point(neighbor) - p).norm_squared());
        }
    }

    #[test]
    fn projection_resolves_thin_directions_despite_large_normal_offset() {
        let sampled = plane(
            [0.0, 1.0],
            [0.0, 1.0],
            DVec3::zeros(),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1e-16, 0.0),
        );
        let uv = sampled
            .uv_from_point(DVec3::new(0.35, 0.73e-16, 2.0))
            .unwrap();
        close(uv.x, 0.35, 1e-12);
        close(uv.y, 0.73, 1e-12);
    }

    #[test]
    fn curved_patch_normal_projection_is_stationary() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(
            true,
            true,
            KnotVector::from_multiplicities(2, &[0.0, 1.0], &[3, 3]),
            KnotVector::from_multiplicities(1, &[0.0, 1.0], &[2, 2]),
            [(0.0, 0.0), (0.5, 0.0), (1.0, 1.0)]
                .iter()
                .map(|&(x, z)| vec![DVec3::new(x, 0.0, z), DVec3::new(x, 1.0, z)])
                .collect(),
        ));
        let expected = DVec2::new(0.37, 0.73);
        let p = sampled.surf.point(expected);
        let normal = DVec3::new(-2.0 * expected.x, 0.0, 1.0).normalize();
        for offset in [0.0, -0.05, 0.05] {
            let uv = sampled.uv_from_point(p + normal * offset).unwrap();
            close(uv.x, expected.x, 1e-10);
            close(uv.y, expected.y, 1e-10);
        }
    }

    #[test]
    fn near_curvature_center_projection_uses_distance_hessian() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(2, &[-1., 1.], &[3, 3]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(-1., 1.), (0., -1.), (1., 1.)].iter().map(|&(x, z)|
                vec![DVec3::new(x, 0., z), DVec3::new(x, 1., z)]).collect()));
        let uv = sampled.uv_from_point_newtons_method(
            DVec3::new(0., 0.37, 0.49999), DVec2::new(0.1, 0.4)).unwrap();
        close(uv.x, 0., 1e-8);
        close(uv.y, 0.37, 1e-12);
    }

    #[test]
    fn closed_surface_projection_retains_a_seam_endpoint_minimum() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(false, true,
            KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 1e-12)].iter()
                .map(|&(x, y)| vec![DVec3::new(x, y, 0.), DVec3::new(x, y, 1.)]).collect()));
        // Geometric closure (with rounded endpoint data) is not an instruction
        // to wrap a local bounded minimization across the two knot endpoints.
        let uv = sampled.uv_from_point(DVec3::new(-1e-12, 0., 0.37)).unwrap();
        assert_eq!(uv.x, 0.);
        close(uv.y, 0.37, 1e-12);
    }

    #[test]
    fn closed_surface_projection_samples_both_sides_of_the_seam() {
        let w = 0.5_f64.sqrt();
        let circle = [
            (1., 0., 1.),
            (1., 1., w),
            (0., 1., 1.),
            (-1., 1., w),
            (-1., 0., 1.),
            (-1., -1., w),
            (0., -1., 1.),
            (1., -1., w),
            (1., 0., 1.),
        ];
        let sampled = SampledSurface::new(NDBSplineSurface::new(
            false,
            true,
            KnotVector::from_multiplicities(2, &[0., 1., 2., 3., 4.], &[3, 2, 2, 2, 3]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            circle
                .iter()
                .map(|&(x, y, w)| {
                    [0., 1.]
                        .iter()
                        .map(|z| nalgebra_glm::DVec4::new(x * w, y * w, z * w, w))
                        .collect()
                })
                .collect(),
        ));
        let expected = DVec2::new(0.01, 0.6);
        let uv = sampled
            .uv_from_point(sampled.surf.point(expected))
            .unwrap();
        close(uv.x, expected.x, 1e-10);
        close(uv.y, expected.y, 1e-10);
        let expected = DVec2::new(3.99, 0.4);
        let uv = sampled.uv_from_point(sampled.surf.point(expected)).unwrap();
        close(uv.x, expected.x, 1e-10);
        close(uv.y, expected.y, 1e-10);
    }

    #[test]
    fn inverse_projection_round_trip_and_normal_offset() {
        let sampled = plane(
            [0.0, 1.0],
            [0.0, 1.0],
            DVec3::new(2.0, -3.0, 4.0),
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(0.0, 5.0, 0.0),
        );
        let expected = DVec2::new(0.371, 0.826);
        let point = sampled.surf.point(expected);
        let round_trip = sampled.uv_from_point(point).unwrap();
        let normal_projection = sampled
            .uv_from_point(point + DVec3::new(0.0, 0.0, 37.0))
            .unwrap();
        close(round_trip.x, expected.x, 1e-10);
        close(round_trip.y, expected.y, 1e-10);
        close(normal_projection.x, expected.x, 1e-10);
        close(normal_projection.y, expected.y, 1e-10);
    }

    #[test]
    fn inverse_projection_obeys_active_boundaries() {
        let sampled = plane(
            [-2.0, 4.0],
            [10.0, 20.0],
            DVec3::zeros(),
            DVec3::new(6.0, 0.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
        );
        let uv = sampled.uv_from_point(DVec3::new(-7.0, 4.0, 3.0)).unwrap();
        close(uv.x, -2.0, 0.0);
        close(uv.y, 14.0, 1e-10);
    }

    #[test]
    fn inverse_projection_is_invariant_to_domain_and_length_scale() {
        for scale in [1e-9, 1e9] {
            let sampled = plane(
                [1e6, 1e6 + 1e-5],
                [-3e-7, 8e-7],
                DVec3::zeros(),
                DVec3::new(scale, 0.0, 0.0),
                DVec3::new(0.0, 2.0 * scale, 0.0),
            );
            let expected = DVec2::new(1e6 + 0.63e-5, -3e-7 + 0.24 * 1.1e-6);
            let uv = sampled.uv_from_point(sampled.surf.point(expected)).unwrap();
            close(uv.x, expected.x, 2e-10);
            close(uv.y, expected.y, 2e-16);
        }
    }

    #[test]
    fn inverse_projection_resolves_thin_patch_direction() {
        let sampled = plane(
            [0.0, 1.0],
            [0.0, 1.0],
            DVec3::zeros(),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1e-12, 0.0),
        );
        let uv = sampled
            .uv_from_point(DVec3::new(0.35, 0.73e-12, 2e-12))
            .unwrap();
        close(uv.x, 0.35, 1e-10);
        close(uv.y, 0.73, 1e-10);
    }

    #[test]
    fn inverse_projection_handles_a_singular_derivative_direction() {
        let sampled = plane(
            [0.0, 1.0],
            [0.0, 1.0],
            DVec3::zeros(),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::zeros(),
        );
        let uv = sampled.uv_from_point(DVec3::new(1.24, 2.0, 0.0)).unwrap();
        close(uv.x, 0.31, 1e-10);
        assert!((0.0..=1.0).contains(&uv.y));
    }

    #[test]
    fn projection_domain_is_bounded_even_for_closed_surfaces() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(
            false,
            true,
            KnotVector::from_multiplicities(1, &[2.0, 5.0], &[2, 2]),
            KnotVector::from_multiplicities(1, &[-7.0, -3.0], &[2, 2]),
            vec![vec![DVec3::zeros(); 2]; 2],
        ));
        for parameter in [-100.0_f64, -2.0, 0.0, 2.0, 3.0, 100.0] {
            assert_eq!(
                sampled.constrain_uv(DVec2::new(parameter, -100.0)),
                DVec2::new(parameter.clamp(2.0, 5.0), -7.0),
            );
        }
        for endpoint in [2.0, 5.0] {
            assert_eq!(
                sampled.constrain_uv(DVec2::new(endpoint, 100.0)),
                DVec2::new(endpoint, -3.0),
            );
        }
    }

    #[test]
    fn samples_only_the_valid_knot_domain() {
        let surface = NDBSplineSurface::new(
            false,
            true,
            KnotVector::from_multiplicities(1, &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[1; 6]),
            KnotVector::from_multiplicities(1, &[0.0, 1.0], &[2, 2]),
            vec![
                vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)],
                vec![DVec3::new(1.0, 0.0, 0.0), DVec3::new(1.0, 1.0, 0.0)],
                vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(2.0, 1.0, 0.0)],
                vec![DVec3::new(3.0, 0.0, 0.0), DVec3::new(3.0, 1.0, 0.0)],
            ],
        );

        let sampled = SampledSurface::new(surface);

        assert!(sampled.samples.iter().all(|(uv, _)| {
            uv.x >= sampled.surf.min_u()
                && uv.x <= sampled.surf.max_u()
                && uv.y >= sampled.surf.min_v()
                && uv.y <= sampled.surf.max_v()
        }));
    }
}
