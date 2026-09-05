use nalgebra_glm::{dot, length, length2, DMat2x2, DVec2, DVec3};
use crate::{abstract_surface::AbstractSurface, nd_surface::NDBSplineSurface};
use log::error;

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
    where NDBSplineSurface<N>: AbstractSurface
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
                        let q = surf.point_from_basis(
                            u_span, &u_basis, v_span, &v_basis);
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

    fn constrain_uv(&self, mut uv: DVec2) -> DVec2 {
        let domains = [
            (self.surf.min_u(), self.surf.max_u(), self.surf.u_open),
            (self.surf.min_v(), self.surf.max_v(), self.surf.v_open),
        ];
        for (value, (min, max, open)) in uv.iter_mut().zip(domains) {
            if open {
                *value = value.clamp(min, max);
            } else if *value < min || *value > max {
                // Newton steps can cross arbitrarily many periods. Keep
                // in-domain endpoints unchanged to preserve their knot side.
                *value = min + (*value - min).rem_euclid(max - min);
            }
        }
        uv
    }

    fn newtons_method_inner(&self, P: DVec3, uv_0: DVec2, max_iter: usize) -> Option<DVec2> {
        let eps1 = 0.01; // a Euclidean distance error bound
        let eps2 = 0.01; // a cosine error bound

        let mut uv_i = self.constrain_uv(uv_0);
        for _ in 0..max_iter {
            // The surface and its derivatives at uv_i
            let derivs = self.surf.derivs::<2>(uv_i);
            let S = derivs[0][0];
            let S_u = derivs[1][0];
            let S_v = derivs[0][1];
            let S_uu = derivs[2][0];
            let S_uv = derivs[1][1]; // S_vu is the same
            let S_vv = derivs[0][2];
            let r = S - P;

            // If |S(uv_i) - P| < \epsilon_1  and
            //    |S_u(uv_i) dot (S(uv_i) - P)| / |S_u(uv_i)| / |S(uv_i) - P| < \epsilon_2  and
            //    |S_v(uv_i) dot (S(uv_i) - P)| / |S_v(uv_i)| / |S(uv_i) - P| < \epsilon_2
            // then we are done
            let r_len = length(&r);
            if r_len < eps1 {
                let su_len = length(&S_u);
                let sv_len = length(&S_v);
                // Skip cosine check when the derivative or residual is
                // degenerate (near-zero) to avoid 0/0 = NaN failures.
                // Use a tight threshold so we only bypass for truly
                // degenerate surfaces (collapsed control point rows).
                let cos_u_ok = su_len < 1e-10 || r_len < 1e-10
                    || dot(&r, &S_u).abs() / su_len / r_len < eps2;
                let cos_v_ok = sv_len < 1e-10 || r_len < 1e-10
                    || dot(&r, &S_v).abs() / sv_len / r_len < eps2;
                if cos_u_ok && cos_v_ok {
                    return Some(uv_i);
                }
            }

            // Otherwise, compute uv_{i+1} by computing:
            // let r(u, v) = S(u, v) - P
            // let f(u, v) = r(u, v) dot S_u(u, v)
            // let g(u, v) = r(u, v) dot S_v(u, v)
            // let K_i = -(f(uv_{i}), g(uv_{i}))
            // let J_i = [[df/du, df/dv], [dg/du, dg/dv]]
            //           = [[|S_u|^2 + r dot S_uu, S_u dot S_v + r dot S_uv],
            //              [S_u dot S_v + r dot S_vu, |S_v|^2 + r dot S_vv]]
            // let delta_i = (J_i)^{-1} * K_i
            // let uv_{i+1} = delta_i + uv_i
            let f = dot(&r, &S_u);
            let g = dot(&r, &S_v);
            let K_i = -DVec2::new(f, g);
            let J_i = symmetric2x2(
                length2(&S_u) + dot(&r, &S_uu),
                dot(&S_u, &S_v) + dot(&r, &S_uv),
                length2(&S_v) + dot(&r, &S_vv),
            );
            let delta_i = match J_i.try_inverse() {
                None => {
                    // Singular Jacobian (e.g. degenerate surface edge where
                    // a whole row of control points collapses to one point).
                    // If we're already reasonably close, accept the result.
                    if r_len < eps1 * 10.0 {
                        return Some(uv_i);
                    }
                    return None;
                },
                Some(m) => m * K_i,
            };
            let uv_ip1 = self.constrain_uv(uv_i + delta_i);

            // If the values didn't change much, we can stop iterating
            // if |(u_{i+1} - u_i) * S_u(u_i, v_i) + (v_{i+1} - v_i) * S_v(u_i, v_i) | < \epsilon_1

            let delta_i = uv_ip1 - uv_i;
            if length(&(delta_i.x * S_u + delta_i.y * S_v)) < eps1 {
                return Some(uv_ip1);
            }

            // otherwise, iterate again
            uv_i = uv_ip1;
        }
        None
    }

    pub fn uv_from_point(&self, p: DVec3) -> Option<DVec2> {
        assert!(!self.samples.is_empty());
        let mut best = (f64::INFINITY, u32::MAX);
        kd_nearest(&self.samples, &self.kd, 0, p, &mut best);
        let best_idx = if best.1 == u32::MAX { 0 } else { best.1 as usize };
        let best_uv = self.samples[best_idx].0;
        self.uv_from_point_newtons_method(p, best_uv)
    }

    // NOTE: do not add warm-start ("hint") seeding from an adjacent contour
    // vertex here. On degenerate patches (e.g. slivers whose whole u-range
    // moves the 3D point by less than the convergence tolerance) a hint seed
    // converges without moving, collapsing distinct contour vertices onto one
    // UV and producing broken CDT input. Every vertex must seed from its own
    // nearest sample so results stay well-defined on such surfaces.
}

/// Builds the symmetric matrix [[a, b], [b, d]]
fn symmetric2x2(a: f64, b: f64, d: f64) -> DMat2x2 {
    // In column major order; because it's symmetric, it doesn't matter
    let mut mat = DMat2x2::identity();
    mat.set_column(0, &DVec2::new(a, b));
    mat.set_column(1, &DVec2::new(b, d));
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;

    #[test]
    fn constrains_multiple_periods_and_preserves_domain_endpoints() {
        let sampled = SampledSurface::new(NDBSplineSurface::new(
            false,
            true,
            KnotVector::from_multiplicities(1, &[2.0, 5.0], &[2, 2]),
            KnotVector::from_multiplicities(1, &[-7.0, -3.0], &[2, 2]),
            vec![vec![DVec3::zeros(); 2]; 2],
        ));
        for period in [-100.0, -2.0, 0.0, 2.0, 100.0] {
            assert_eq!(
                sampled.constrain_uv(DVec2::new(3.0 + 3.0 * period, -100.0)),
                DVec2::new(3.0, -7.0),
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
