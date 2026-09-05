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
    /// Control-hull bounds and the contiguous sample range of each knot cell.
    cells: Vec<SurfaceCell>,
}

const PROJECTION_TOL: f64 = 64. * f64::EPSILON;

#[derive(Debug, Clone)]
struct SurfaceCell {
    spans: [usize; 2],
    bounds: [DVec3; 2],
    samples: std::ops::Range<usize>,
}

struct DistanceModel {
    spans: [usize; 2],
    residual: DVec3,
    position_scale: DVec3,
    gradient: DVec2,
    hessian: [f64; 3],
    lo: DVec2,
    hi: DVec2,
    stationary: [bool; 2],
    converged: bool,
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

/// Compare the rectangle's stationary candidates after mapping each step to
/// representable parameters. Axis candidates keep a resolvable direction free
/// when rounding prevents the other component of a coupled step from moving.
fn quadratic_step(g: DVec2, h: [f64; 3], lo: DVec2, hi: DVec2,
                  represent: impl Fn(DVec2) -> DVec2) -> DVec2 {
    let [a, b, d] = h;
    let mut best = DVec2::zeros();
    let mut consider = |q: DVec2| {
        let q = represent(q);
        // Compare in factored form so a resolved thin direction is not
        // erased by the common contribution of a much larger direction.
        let sum = q + best;
        let change = dot(&(q - best), &(g + 0.5 * DVec2::new(
            a * sum.x + b * sum.y, b * sum.x + d * sum.y)));
        if change < 0. {
            best = q;
        }
    };
    for x in [lo.x, hi.x] {
        for y in [lo.y, hi.y] {
            consider(DVec2::new(x, y));
        }
        if d > 0. { consider(DVec2::new(x, (-(g.y + b * x) / d).clamp(lo.y, hi.y))); }
    }
    for y in [lo.y, hi.y] {
        if a > 0. { consider(DVec2::new((-(g.x + b * y) / a).clamp(lo.x, hi.x), y)); }
    }
    if a > 0. { consider(DVec2::new((-g.x / a).clamp(lo.x, hi.x), 0.)); }
    if d > 0. { consider(DVec2::new(0., (-g.y / d).clamp(lo.y, hi.y))); }
    let det = a.mul_add(d, -b * b);
    if a > 0. && det > 0. {
        let q = DVec2::new(b.mul_add(g.y, -d * g.x), b.mul_add(g.x, -a * g.y)) / det;
        if (0..2).all(|i| q[i] >= lo[i] && q[i] <= hi[i]) { consider(q); }
    }
    best
}

impl<const N: usize> SampledSurface<N>
where
    NDBSplineSurface<N>: AbstractSurface,
{
    pub fn new(surf: NDBSplineSurface<N>) -> Self {
        const N: usize = 8;
        let mut samples = Vec::new();
        let mut cells = Vec::new();
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
                let start = samples.len();
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
                let mut bounds = surf.control_bounds([i, j]);
                for axis in 0..3 {
                    let roundoff = PROJECTION_TOL * bounds[0][axis].abs().max(bounds[1][axis].abs());
                    bounds[0][axis] -= roundoff;
                    bounds[1][axis] += roundoff;
                }
                cells.push(SurfaceCell { spans: [i, j], bounds, samples: start..samples.len() });
            }
        }
        let mut kd: Vec<u32> = (0..samples.len() as u32).collect();
        build_kd(&samples, &mut kd, 0);
        Self { surf, samples, kd, cells }
    }

    // Section 6.1 (start middle page 232)
    pub fn uv_from_point_newtons_method(&self, P: DVec3, uv_0: DVec2) -> Option<DVec2> {
        let domain = [&self.surf.u_knots, &self.surf.v_knots].map(|k| 0..k.len());
        let out = self.newtons_method_inner(P, uv_0, 256, domain);
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

    fn stepped_uv(&self, uv: DVec2, ranges: DVec2, spans: [usize; 2], step: DVec2) -> DVec2 {
        let mut candidate = uv + step.component_mul(&ranges);
        for (i, knots) in [&self.surf.u_knots, &self.surf.v_knots].iter().enumerate() {
            let (min, max) = (knots[spans[i]], knots[spans[i] + 1]);
            candidate[i] = if step[i] == (min - uv[i]) / ranges[i] { min }
                else if step[i] == (max - uv[i]) / ranges[i] { max }
                else { candidate[i].clamp(min, max) };
        }
        candidate
    }

    fn distance_model(&self, P: DVec3, uv: DVec2, ranges: DVec2, spans: [usize; 2]) -> DistanceModel {
        // Fixed unit-domain coordinates preserve the surface differential's
        // rank and scale continuously, including at collapsed boundaries.
        let derivs = self.surf.derivs_in_span::<2>(uv, spans, P);
        let r = derivs[0][0];
        let columns = [derivs[1][0] * ranges.x, derivs[0][1] * ranges.y];
        let norms = DVec2::new(columns[0].norm(), columns[1].norm());
        let unit = [0, 1].map(|i| if norms[i] > 0. { columns[i] / norms[i] } else { DVec3::zeros() });
        let gradient = DVec2::new(dot(&r, &unit[0]), dot(&r, &unit[1]));
        let mut projected = gradient;
        let domains = [
            (self.surf.u_knots[spans[0]], self.surf.u_knots[spans[0] + 1]),
            (self.surf.v_knots[spans[1]], self.surf.v_knots[spans[1] + 1]),
        ];
        for i in 0..2 {
            let (min, max) = domains[i];
            if (uv[i] == min && gradient[i] >= 0.) || (uv[i] == max && gradient[i] <= 0.) {
                projected[i] = 0.;
            }
        }
        // Test each parameter direction at its own scale. A large normal
        // offset or a long u direction must not erase a resolvable thin v
        // direction. Position roundoff is likewise componentwise.
        let position_scale = (P + r).abs() + P.abs();
        let stationary = [0, 1].map(|i| {
            projected[i].abs() <= PROJECTION_TOL * (norms[i] + dot(&unit[i].abs(), &position_scale))
        });
        let g = DVec2::new(dot(&r, &columns[0]), dot(&r, &columns[1]));
        let h = [
            columns[0].norm_squared() + dot(&r, &derivs[2][0]) * ranges.x * ranges.x,
            dot(&columns[0], &columns[1]) + dot(&r, &derivs[1][1]) * ranges.x * ranges.y,
            columns[1].norm_squared() + dot(&r, &derivs[0][2]) * ranges.y * ranges.y,
        ];
        let lo = DVec2::new(domains[0].0 - uv.x, domains[1].0 - uv.y).component_div(&ranges);
        let hi = DVec2::new(domains[0].1 - uv.x, domains[1].1 - uv.y).component_div(&ranges);
        let step = quadratic_step(g, h, lo, hi,
            |q| (self.stepped_uv(uv, ranges, spans, q) - uv).component_div(&ranges));
        let candidate = self.stepped_uv(uv, ranges, spans, step);
        // Only the full-cell model step may establish representability
        // convergence, never a step shortened by the trust region.
        let converged = (0..2).all(|i| stationary[i] || candidate[i] == uv[i]);
        DistanceModel { spans, residual: r, position_scale, gradient: g, hessian: h, lo, hi, stationary, converged }
    }

    fn newtons_method_inner(&self, P: DVec3, uv_0: DVec2, max_iter: usize,
        domain: [std::ops::Range<usize>; 2]) -> Option<DVec2> {
        let ranges = DVec2::new(self.surf.max_u() - self.surf.min_u(), self.surf.max_v() - self.surf.min_v());
        let mut uv_i = self.constrain_uv(uv_0);
        let mut radius: f64 = 1.;
        for _ in 0..max_iter {
            // Smooth interiors and knot junctions use the same bounded model.
            // At a junction, every incident cell must satisfy its one-sided
            // conditions before the point is a constrained minimum.
            let mut models = smallvec::SmallVec::<[DistanceModel; 4]>::new();
            for u in self.surf.u_knots.spans_at(uv_i.x).filter(|u| domain[0].contains(u)) {
                for v in self.surf.v_knots.spans_at(uv_i.y).filter(|v| domain[1].contains(v)) {
                    models.push(self.distance_model(P, uv_i, ranges, [u, v]));
                }
            }
            if models.iter().all(|m| m.converged) { return Some(uv_i); }
            let mut accepted: Option<(DVec2, DVec3, f64)> = None;
            for _ in 0..40 {
                for m in models.iter().filter(|m| !m.converged) {
                    let step = quadratic_step(m.gradient, m.hessian,
                        m.lo.sup(&DVec2::repeat(-radius)), m.hi.inf(&DVec2::repeat(radius)),
                        |q| (self.stepped_uv(uv_i, ranges, m.spans, q) - uv_i).component_div(&ranges));
                    let mut candidate = self.stepped_uv(uv_i, ranges, m.spans, step);
                    let mut candidate_r = self.surf.derivs_in_span::<0>(candidate, m.spans, P)[0][0];
                    let prediction = |point: DVec2| {
                        let q = (point - uv_i).component_div(&ranges);
                        let h = m.hessian;
                        dot(&q, &(m.gradient + 0.5 * DVec2::new(h[0] * q.x + h[1] * q.y,
                            h[1] * q.x + h[2] * q.y)))
                    };
                    // A coupled step may displace an already stationary
                    // active bound through quotient roundoff. Retain that
                    // bound unless leaving it improves actual distance; do
                    // not reset unconverged or interior coordinates to knots.
                    for i in 0..2 {
                        if !m.stationary[i] || (m.lo[i] != 0. && m.hi[i] != 0.) { continue; }
                        let mut bound = candidate;
                        bound[i] = uv_i[i];
                        // Keep the original descent trial when a knot variant
                        // has no predicted gain; it cannot prove convergence.
                        if prediction(bound) >= 0. { continue; }
                        let bound_r = self.surf.derivs_in_span::<0>(bound, m.spans, P)[0][0];
                        if dot(&(bound_r - candidate_r), &(bound_r + candidate_r)) <= 0. {
                            candidate = bound;
                            candidate_r = bound_r;
                        }
                    }
                    let predicted = prediction(candidate);
                    // Compare residual distances in factored form to retain
                    // small improvements near a nonzero normal offset.
                    let change = 0.5 * dot(&(candidate_r - m.residual), &(candidate_r + m.residual));
                    let roundoff = PROJECTION_TOL
                        * dot(&(candidate_r.abs() + m.residual.abs()), &m.position_scale);
                    if predicted < 0. && change <= 0.1 * predicted + roundoff
                        && accepted.as_ref().map_or(true, |(_, best, _)|
                            dot(&(candidate_r - best), &(candidate_r + best)) < 0.) {
                        // Roundoff may permit a step without a resolved gain.
                        // Such acceptance is not evidence that the quadratic
                        // fits: reduce its radius when it overpredicts descent.
                        let scale = if change > 0.25 * predicted { 0.25 }
                            else if change <= 0.75 * predicted { 2. } else { 1. };
                        accepted = Some((candidate, candidate_r, scale));
                    }
                }
                if let Some((_, _, scale)) = accepted {
                    radius = (scale * radius).min(1.);
                    break;
                }
                radius *= 0.25;
            }
            uv_i = accepted?.0;
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
        let distance = |uv| self.surf.derivs_relative_to::<0>(uv, p)[0][0].norm_squared();
        let domain = [&self.surf.u_knots, &self.surf.v_knots].map(|k| 0..k.len());
        let mut result = seeds.iter().copied()
            .filter_map(|seed| self.newtons_method_inner(p, seed, 256, domain.clone()))
            .min_by_key(|&uv| ordered_float::OrderedFloat(distance(uv)));
        let mut error = result.map_or(f64::INFINITY, distance);
        // A nearby sample need not lie in the basin of the nearby surface
        // sheet. Consider every knot cell whose control hull could improve
        // the current projection. Keep each search within its cell: a bound
        // on surface position is not a bound on an unrestrained Newton basin.
        for cell in &self.cells {
            let bounds = cell.bounds;
            let lower_bound: f64 = (0..3).map(|i|
                (bounds[0][i] - p[i]).max(p[i] - bounds[1][i]).max(0.).powi(2)).sum();
            if lower_bound >= error { continue; }
            let seed = self.samples[cell.samples.clone()].iter()
                .min_by_key(|(_, q)| ordered_float::OrderedFloat(dist2(*q, p))).unwrap().0;
            let domain = cell.spans.map(|span| span..span + 1);
            if let Some(uv) = self.newtons_method_inner(p, seed, 256, domain) {
                let candidate_error = distance(uv);
                if candidate_error < error {
                    result = Some(uv);
                    error = candidate_error;
                }
            }
        }
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
    fn projection_tests_both_sides_of_interior_knots() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(1, &[0., 0.5, 1.], &[2, 1, 2]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(0., 0.5), (0.5, 0.), (1., 0.5)].iter().map(|&(x, z)|
                vec![DVec3::new(x, 0., z), DVec3::new(x, 1., z)]).collect()));
        for (p, expected) in [(DVec3::new(0.5, 0.37, -0.1), DVec2::new(0.5, 0.37)),
            (DVec3::new(0.7, 0.37, 0.2), DVec2::new(0.7, 0.37))] {
            for start in [0.25, 0.5, 0.75] {
                let uv = sampled.uv_from_point_newtons_method(p, DVec2::new(start, 0.5)).unwrap();
                close(uv.x, expected.x, 1e-12);
                close(uv.y, expected.y, 1e-12);
            }
        }
    }

    #[test]
    fn bounded_quadratic_preserves_thin_directions_and_handles_indefinite_curvature() {
        let lo = DVec2::repeat(-1.);
        let hi = DVec2::repeat(1.);
        let q = quadratic_step(DVec2::new(-0.3, -0.7e-32), [1., 0., 1e-32], lo, hi, |q| q);
        close(q.x, 0.3, 1e-15);
        close(q.y, 0.7, 1e-15);
        let q = quadratic_step(DVec2::new(-0.1, 0.), [1., 2., 1.], lo, hi, |q| q);
        assert_eq!(q, DVec2::new(1., -1.));
    }

    #[test]
    fn rounded_coupled_steps_keep_the_representable_direction_free() {
        let origin = DVec2::new(0.5, 1.);
        let g = DVec2::new(1e-44, -1e-17);
        let h = [1e-42, -1e-26, 1.];
        let lo = DVec2::new(-0.5, -0.5);
        let hi = DVec2::new(0.5, 1.);
        let represent = |q: DVec2| (origin + q) - origin;
        let continuous = represent(quadratic_step(g, h, lo, hi, |q| q));
        assert!(continuous.x > 0. && continuous.y == 0.);
        let discrete = quadratic_step(g, h, lo, hi, represent);
        assert!(discrete.x < 0. && discrete.y == 0.);
        assert!(g.x * discrete.x + 0.5 * h[0] * discrete.x * discrete.x < 0.);
    }

    #[test]
    fn rational_extrusion_boundary_remains_a_straight_parameter_line() {
        let points = [(-1.767765, 3.557235, 1.),
            (1.789502361076, 3.557235, 0.83152615303), (3.151478602946, 0.2710256648, 1.)];
        let sampled = SampledSurface::new(NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(2, &[0., 1.17789368835], &[3, 3]),
            KnotVector::from_multiplicities(1, &[7.012443731511, 7.2], &[2, 2]),
            points.iter().map(|&(x, y, w)| [15.487556268489, 15.3].iter()
                .map(|&z| nalgebra_glm::DVec4::new(x, y, z, 1.) * w).collect()).collect()));
        for z in [15.487538034315, 15.46, 15.39, 15.3] {
            let uv = sampled.uv_from_point(DVec3::new(-1.767765, 3.557235, z)).unwrap();
            assert_eq!(uv.x, 0.);
            close(sampled.surf.point(uv).z, z, 4. * f64::EPSILON * z);
        }
    }

    #[test]
    fn projection_near_a_pole_preserves_the_vanishing_tangent() {
        // S(u,v) = (u, u*v, 0.001*u*v*(1-v)). At u=0 the v
        // tangent vanishes, but an interior normal projection still exists.
        let sampled = SampledSurface::new(NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            vec![vec![DVec3::zeros(); 3], vec![DVec3::new(1., 0., 0.),
                DVec3::new(1., 0.5, 0.0005), DVec3::new(1., 1., 0.)]]));
        let expected = DVec2::new(1e-10, 0.5);
        let jet = sampled.surf.derivs::<1>(expected);
        let p = jet[0][0] + jet[1][0].cross(&jet[0][1]).normalize() * 1e-7;
        let uv = sampled.uv_from_point_newtons_method(p, DVec2::zeros()).unwrap();
        close(uv.x, expected.x, 1e-14);
        close(uv.y, expected.y, 1e-5);
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
    fn trust_radius_shrinks_when_roundoff_accepts_a_poor_model() {
        // A 120-degree circular arc extruded in z. At either arc endpoint,
        // the quadratic predicts descent to the equally distant other end.
        // The large constrained z offset masks that poor fit in acceptance,
        // but must not prevent the trust radius from shrinking.
        let y = 0.75_f64.sqrt();
        let surface = NDBSplineSurface::new(true, true,
            KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(0.5, -y, 1.), (2., 0., 0.5), (0.5, y, 1.)].iter().map(|&(x, y, w)|
                [1., 2.].iter().map(|&z|
                    nalgebra_glm::DVec4::new(x * 1e-16, y * 1e-16, z, 1.) * w)
                    .collect()).collect());
        let sampled = SampledSurface::new(surface);
        for u in [0., 1.] {
            let uv = sampled.uv_from_point_newtons_method(
                DVec3::new(2e-16, 0., 3.), DVec2::new(u, 1.)).unwrap();
            close(uv.x, 0.5, 1e-12);
            assert_eq!(uv.y, 1.);
        }
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
    fn inverse_projection_finds_the_nearby_sheet_not_the_nearby_sample() {
        // Two close parallel strips joined above the target. The nearest
        // grid sample belongs to the left strip, but the target is on the
        // right strip. Newton on the left stops at an off-surface minimum.
        let controls: Vec<Vec<DVec3>> = [(0., 0.1), (0., 0.8), (0.001, 1.), (0.001, 0.)]
            .iter().map(|&(x, y)| [0., 1.].iter()
                .map(|&z| DVec3::new(x, y, z)).collect()).collect();
        let u = KnotVector::from_multiplicities(1, &[0., 1., 2., 3.], &[2, 1, 1, 2]);
        let v = KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let target = DVec3::new(0.001, 0.5, 0.33);
        let polynomial = SampledSurface::new(NDBSplineSurface::new(
            true, true, u.clone(), v.clone(), controls.clone(),
        ));
        let uv = polynomial.uv_from_point(target).unwrap();
        assert!((polynomial.surf.point(uv) - target).norm() < 1e-12);
        let rational = SampledSurface::new(NDBSplineSurface::new(
            true, true, u, v, controls.iter().zip([0.25, 0.5, 2., 1.].iter())
                .map(|(row, &w)| row.iter().map(|p|
                    nalgebra_glm::DVec4::new(p.x * w, p.y * w, p.z * w, w)
                ).collect()).collect(),
        ));
        let uv = rational.uv_from_point(target).unwrap();
        assert!((rational.surf.point(uv) - target).norm() < 1e-12);
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
