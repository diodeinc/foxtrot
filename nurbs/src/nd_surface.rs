use std::cmp::min;
use nalgebra_glm::{DVec2, DVec3, TVec};
use crate::{KnotVector, VecF};

#[derive(Debug, Clone)]
pub struct NDBSplineSurface<const D: usize> {
    pub u_open: bool,
    pub v_open: bool,
    pub u_knots: KnotVector,
    pub v_knots: KnotVector,
    control_points: Vec<Vec<TVec<f64, D>>>,
}

impl NDBSplineSurface<4> {
    /// A convex, constant-weight bilinear patch is a regular plane exactly
    /// when its four controls are coplanar. No geometric tolerance is used.
    pub fn bilinear_plane_normal(&self) -> Option<DVec3> {
        if self.u_knots.degree() != 1 || self.v_knots.degree() != 1
            || self.control_points.len() != 2
            || self.control_points.iter().any(|row| row.len() != 2) {
            return None;
        }
        let controls = [&self.control_points[0][0], &self.control_points[1][0],
            &self.control_points[1][1], &self.control_points[0][1]];
        let weight = controls[0].w;
        if weight == 0.0 || !weight.is_finite() || controls.iter().any(|p| p.w != weight) {
            return None;
        }
        // Equal weights allow predicates on the homogeneous numerators,
        // avoiding division roundoff. Respect exact-predicate exponent bounds.
        let points = controls.map(|p| DVec3::new(p.x, p.y, p.z));
        if points.iter().flat_map(|p| p.iter()).any(|&v|
            !v.is_finite() || (v != 0.0 && (v.abs() < 2.0_f64.powi(-142) || v.abs() > 2.0_f64.powi(201)))) {
            return None;
        }
        let xyz = points.map(|p| robust::Coord3D { x: p.x, y: p.y, z: p.z });
        if robust::orient3d(xyz[0], xyz[1], xyz[2], xyz[3]) != 0.0 {
            return None;
        }
        let normal = (points[1] - points[0]).cross(&(points[3] - points[0]));
        let dropped = normal.iamax();
        if normal[dropped] == 0.0 { return None; }
        let coordinates = [(dropped + 1) % 3, (dropped + 2) % 3];
        let xy = points.map(|p| robust::Coord { x: p[coordinates[0]], y: p[coordinates[1]] });
        // Strict convexity excludes collapsed or folded parameterizations.
        if (0..4).any(|i| robust::orient2d(xy[i], xy[(i + 1) % 4], xy[(i + 2) % 4])
            * normal[dropped].signum() <= 0.0) {
            return None;
        }
        Some(normal)
    }
}

/// Non-rational b-spline surface with 3D control points
impl<const D: usize> NDBSplineSurface<D> {
    pub fn new(
        u_open: bool,
        v_open: bool,
        u_knots: KnotVector,
        v_knots: KnotVector,
        control_points: Vec<Vec<TVec<f64, D>>>,
    ) -> Self {
        Self {
            u_open,
            v_open,
            u_knots,
            v_knots,
            control_points,
        }
    }

    pub fn min_u(&self) -> f64 {
        self.u_knots.min_t()
    }
    pub fn max_u(&self) -> f64 {
        self.u_knots.max_t()
    }
    pub fn min_v(&self) -> f64 {
        self.v_knots.min_t()
    }
    pub fn max_v(&self) -> f64 {
        self.v_knots.max_t()
    }

    /// Tests whether a rational boundary lies within `uncertainty` of its
    /// first Cartesian control. Positive weights give a convex-hull bound
    /// on the entire iso-curve. Zero uncertainty requires exact coincidence.
    ///
    /// `parameter` is 0 for a fixed u and 1 for a fixed v.  Evaluating the
    /// fixed direction's basis (rather than selecting an end control row)
    /// also handles non-clamped knot vectors.
    pub fn rational_boundary_is_point(&self, parameter: usize, value: f64, uncertainty: f64) -> bool {
        let controls = self.boundary_controls(parameter, value);
        let Some(reference) = controls.first() else { return false; };
        if D < 2 || reference[D - 1] <= 0.0 { return false; }
        controls.iter().all(|point| {
            point[D - 1] > 0.0 && (0..D - 1).fold(0.0_f64, |distance, i| {
                // Use the evaluator's homogeneous translation before division:
                // multiplying a shared Cartesian point by different weights
                // need not produce equal rounded Cartesian quotients.
                distance.hypot((point[i] - (reference[i] / reference[D - 1]) * point[D - 1]) / point[D - 1])
            }) <= uncertainty
        })
    }

    /// Sufficient control-hull test for matching endpoint iso-curves. Equal
    /// normalized positive weights give both curves the same rational basis,
    /// so control distances bound their pointwise separation everywhere.
    pub fn rational_boundaries_coincide(&self, parameter: usize, uncertainty: f64) -> bool {
        let (min, max) = match parameter {
            0 => (self.min_u(), self.max_u()),
            1 => (self.min_v(), self.max_v()),
            _ => return false,
        };
        if D < 2 { return false; }
        let a = self.boundary_controls(parameter, min);
        let b = self.boundary_controls(parameter, max);
        let (Some(first_a), Some(first_b)) = (a.first(), b.first()) else { return false; };
        let w = D - 1;
        if first_a[w] <= 0.0 || first_b[w] <= 0.0 { return false; }
        a.iter().zip(&b).all(|(a, b)| {
            a[w] > 0.0 && b[w] > 0.0 && a[w] / first_a[w] == b[w] / first_b[w]
                && (0..w).fold(0.0_f64, |distance, i|
                    distance.hypot(a[i] / a[w] - b[i] / b[w])) <= uncertainty
        })
    }

    fn boundary_controls(&self, parameter: usize, value: f64) -> Vec<TVec<f64, D>> {
        match parameter {
            0 => {
                let span = self.u_knots.find_span(value);
                let basis = self.u_knots.basis_funs_for_span(span, value);
                let first = span - self.u_knots.degree();
                (0..self.control_points[0].len()).map(|v| {
                    basis.iter().enumerate().fold(TVec::zeros(), |sum, (i, b)| {
                        sum + *b * self.control_points[first + i][v]
                    })
                }).collect()
            },
            1 => {
                let span = self.v_knots.find_span(value);
                let basis = self.v_knots.basis_funs_for_span(span, value);
                let first = span - self.v_knots.degree();
                self.control_points.iter().map(|row| {
                    basis.iter().enumerate().fold(TVec::zeros(), |sum, (i, b)| {
                        sum + *b * row[first + i]
                    })
                }).collect()
            },
            _ => Vec::new(),
        }
    }

    /// Converts a point at position uv onto the 3D mesh, using basis functions
    /// of order `p + 1` and `q + 1` respectively.
    ///
    /// ALGORITHM A3.5
    pub fn surface_point(&self, uv: DVec2) -> TVec<f64, D> {
        let uspan = self.u_knots.find_span(uv.x);
        let Nu = self.u_knots.basis_funs_for_span(uspan, uv.x);

        let vspan = self.v_knots.find_span(uv.y);
        let Nv = self.v_knots.basis_funs_for_span(vspan, uv.y);

        self.surface_point_from_basis(uspan, &Nu, vspan, &Nv)
    }

    pub fn surface_point_from_basis(&self,
        uspan: usize, Nu: &VecF,
        vspan: usize, Nv: &VecF) -> TVec<f64, D>
    {
        let (origin, point) = self.surface_point_relative(uspan, Nu, vspan, Nv, |p, origin| p - origin);
        point + origin
    }

    pub(crate) fn surface_point_relative(&self,
        uspan: usize, Nu: &VecF, vspan: usize, Nv: &VecF,
        difference: impl Fn(TVec<f64, D>, TVec<f64, D>) -> TVec<f64, D>,
    ) -> (TVec<f64, D>, TVec<f64, D>) {
        let (origin, jet) = self.tensor_product::<0>([uspan, vspan],
            std::slice::from_ref(Nu), std::slice::from_ref(Nv), difference);
        (origin, jet[0][0])
    }

    /// Returns all derivatives of the surface.  If `D = surface_derivs()`,
    /// `D[k][l]` is the derivative of the surface `k` times in the `u`
    /// direction and `l` times in the `v` direction.
    ///
    /// We compute derivatives up to and including the `d`'th order derivatives.
    ///
    /// ALGORITHM A3.6
    pub fn surface_derivs<const E: usize>(&self, uv: DVec2) -> Vec<Vec<TVec<f64, D>>> {
        let spans = [self.u_knots.find_span(uv.x), self.v_knots.find_span(uv.y)];
        let (origin, mut derivatives) = self.surface_derivs_relative::<E>(uv, spans, |p, origin| p - origin);
        derivatives[0][0] += origin;
        derivatives
    }

    pub(crate) fn surface_derivs_relative<const E: usize>(&self, uv: DVec2, spans: [usize; 2],
        difference: impl Fn(TVec<f64, D>, TVec<f64, D>) -> TVec<f64, D>,
    ) -> (TVec<f64, D>, Vec<Vec<TVec<f64, D>>>) {
        let Nu = self.u_knots.basis_funs_derivs_for_span(spans[0], uv.x, min(E, self.u_knots.degree()));
        let Nv = self.v_knots.basis_funs_derivs_for_span(spans[1], uv.y, min(E, self.v_knots.degree()));
        self.tensor_product::<E>(spans, &Nu, &Nv, difference)
    }

    fn tensor_product<const E: usize>(&self, spans: [usize; 2],
        Nu: &[impl AsRef<[f64]>], Nv: &[impl AsRef<[f64]>],
        difference: impl Fn(TVec<f64, D>, TVec<f64, D>) -> TVec<f64, D>,
    ) -> (TVec<f64, D>, Vec<Vec<TVec<f64, D>>>) {
        let p = self.u_knots.degree();
        let q = self.v_knots.degree();
        // The output matrix goes all the way to order d, even if some of the
        // surfaces are lower order (those values will be locked at 0)
        let mut SKL = vec![vec![TVec::zeros(); E + 1]; E + 1];

        let [uspan, vspan] = spans;
        // The largest tensor basis selects a nearby coordinate origin.
        let uanchor = Nu[0].as_ref().iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let vanchor = Nv[0].as_ref().iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let origin = self.control_points[uspan - p + uanchor][vspan - q + vanchor];
        let mut temp = vec![TVec::zeros(); q + 1];
        for (k, Nu) in Nu.iter().map(AsRef::as_ref).enumerate() {
            for s in 0..=q {
                // Apply partition of unity separately on each axis. A
                // coordinate independent of u must not acquire u roundoff,
                // including in derivatives of a coordinate varying with v.
                let anchor = difference(self.control_points[uspan - p + uanchor][vspan - q + s], origin);
                temp[s] = TVec::zeros();
                for r in 0..=p {
                    temp[s] += Nu[r] * (difference(self.control_points[uspan - p + r][vspan - q + s], origin) - anchor);
                }
                if k == 0 { temp[s] += anchor; }
            }
            let dd = min(E - k, Nv.len() - 1);
            for l in 0..=dd {
                for s in 0..=q {
                    SKL[k][l] += Nv[l].as_ref()[s] * (temp[s] - temp[vanchor]);
                }
                if l == 0 { SKL[k][l] += temp[vanchor]; }
            }
        }
        (origin, SKL)
    }

    // Computes the relative scale of U and V, based on average distance between
    // control points in 3D space
    pub fn aspect_ratio(&self) -> f64 {
        let mut u_sum = 0.0;
        let mut v_sum = 0.0;
        // Helper function to find 3-distance even if this is 4D
        let distance = |a, b| {
            let delta: TVec<f64, D> = a - b;
            DVec3::new(delta[0], delta[1], delta[2]).norm()
        };
        for i in 0..self.control_points.len() {
            for j in 0..self.control_points[i].len() {
                if i > 0 {
                    v_sum += distance(self.control_points[i - 1][j],
                                      self.control_points[i][j]);
                }
                if j > 0 {
                    u_sum += distance(self.control_points[i][j - 1],
                                      self.control_points[i][j]);
                }
            }
        }
        let u_mean = u_sum / self.control_points.len() as f64;
        let v_mean = v_sum / self.control_points[0].len() as f64;

        u_mean / v_mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_glm::DVec4;

    #[test]
    fn extrusion_coordinates_and_derivatives_are_independent_of_the_other_axis() {
        use crate::AbstractSurface;
        let profile = || KnotVector::from_multiplicities(3, &[0., 0.41, 1.], &[4, 1, 4]);
        let height = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let controls: Vec<Vec<_>> = (0..5).map(|i| [0.00988, 0.00989].iter().map(|&z|
            DVec3::new(i as f64, (i * i) as f64, z)).collect()).collect();
        for transposed in [false, true] {
            let (u, v, points) = if transposed {
                (height(), profile(), (0..2).map(|j| controls.iter().map(|row| row[j]).collect()).collect())
            } else { (profile(), height(), controls.clone()) };
            let a = NDBSplineSurface::new(true, true, u.clone(), v.clone(), points.clone());
            let b = NDBSplineSurface::new(true, true, u, v, points.iter().map(|row|
                row.iter().map(|p| DVec4::new(p.x, p.y, p.z, 1.)).collect()).collect());
            let check = |jets: Vec<Vec<DVec3>>, expected_z: f64| {
                assert_eq!(jets[0][0].z, expected_z);
                let (profile, height) = if transposed { (jets[0][1], jets[1][0]) }
                    else { (jets[1][0], jets[0][1]) };
                assert_eq!(profile.z, 0.);
                assert_eq!((height.x, height.y), (0., 0.));
                assert_eq!(jets[1][1], DVec3::zeros());
            };
            let uv = |t| if transposed { DVec2::new(1e-12, t) } else { DVec2::new(t, 1e-12) };
            let reference = DVec3::new(0., 0., 0.00988);
            let expected = a.derivs_relative_to::<2>(uv(0.), reference)[0][0].z;
            for i in 0..=64 {
                check(a.derivs_relative_to::<2>(uv(i as f64 / 64.), reference), expected);
                check(b.derivs_relative_to::<2>(uv(i as f64 / 64.), reference), expected);
            }
        }
    }

    #[test]
    fn clamped_corner_preserves_small_coordinates() {
        let knots = || KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]);
        let end = DVec4::new(1e-30, 2e-30, 3e-30, 1.);
        let mut controls = vec![vec![DVec4::repeat(1.); 3]; 3];
        controls[2][2] = end;
        let surface = NDBSplineSurface::new(true, true, knots(), knots(), controls);
        let uv = DVec2::repeat(1.);
        assert_eq!(surface.surface_point(uv), end);
        assert_eq!(surface.surface_derivs::<2>(uv)[0][0], end);
    }

    #[test]
    fn constant_coordinates_have_exactly_zero_derivatives() {
        let knots = || KnotVector::from_multiplicities(2, &[0., 0.01], &[3, 3]);
        let surface = NDBSplineSurface::new(true, true, knots(), knots(),
            (0..3).map(|i| (0..3).map(|j|
                DVec4::new(8.58999999999999, i as f64, (j * j) as f64, 1.)).collect()).collect());
        for i in 0..=16 {
            let uv = DVec2::new(0.01 * i as f64 / 16., 0.0037);
            assert_eq!(surface.surface_point(uv).x, 8.58999999999999);
            let derivatives = surface.surface_derivs::<2>(uv);
            assert_eq!(derivatives[0][0].x, 8.58999999999999);
            for k in 0..=2 { for l in 0..=2-k {
                if k + l > 0 { assert_eq!(derivatives[k][l].x, 0.); }
            } }
        }
    }

    #[test]
    fn bilinear_plane_recognition_is_exact_and_rejects_folds() {
        let knots = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let controls = vec![
            vec![DVec4::new(1., 2., 3., 1.), DVec4::new(2., 2., 6., 1.)],
            vec![DVec4::new(3., 3., 3., 1.), DVec4::new(4., 3., 6., 1.)],
        ];
        let mut surface = NDBSplineSurface::new(true, true, knots(), knots(), controls);
        let normal = DVec3::new(2., 1., 0.).cross(&DVec3::new(1., 0., 3.));
        assert_eq!(surface.bilinear_plane_normal(), Some(normal));
        for row in &mut surface.control_points { for point in row { *point *= 8.; } }
        assert_eq!(surface.bilinear_plane_normal(), Some(normal * 64.));
        let saved = surface.control_points[1][1];
        surface.control_points[1][1].z = f64::from_bits(saved.z.to_bits() + 1);
        assert!(surface.bilinear_plane_normal().is_none(), "one-ulp warp is not a plane");
        surface.control_points[1][1] = surface.control_points[0][0];
        assert!(surface.bilinear_plane_normal().is_none(), "folded patch is not a regular plane");
        surface.control_points[1][1] = saved;
        surface.control_points[1][1].w *= 2.;
        assert!(surface.bilinear_plane_normal().is_none(), "variable weights need the rational chart");
    }

    #[test]
    fn collapsed_boundary_uses_actual_nonclamped_endpoint_basis() {
        let controls = vec![
            vec![DVec4::new(3., 3., 4., 1.), DVec4::new(1., 3., 4., 1.), DVec4::new(5., 5., 4., 1.)],
            vec![DVec4::new(4., 3., 4., 1.), DVec4::new(0., 3., 4., 1.), DVec4::new(8., 5., 4., 1.)],
        ];
        let clamped = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let nonclamped = || KnotVector::from_multiplicities(2, &[-1., 0., 1., 2., 3., 4.], &[1; 6]);
        let surface = NDBSplineSurface::new(true, true, clamped(), nonclamped(), controls.clone());
        assert!(surface.rational_boundary_is_point(1, surface.min_v(), 0.));
        assert!(!surface.rational_boundary_is_point(1, surface.max_v(), 0.));
        let transposed = (0..3).map(|v| controls.iter().map(|row| row[v]).collect()).collect();
        let surface = NDBSplineSurface::new(true, true, nonclamped(), clamped(), transposed);
        assert!(surface.rational_boundary_is_point(0, surface.min_u(), 0.));
        assert!(!surface.rational_boundary_is_point(0, surface.max_u(), 0.));
    }

    #[test]
    fn collapsed_boundary_preserves_weighted_cartesian_constants() {
        let knots = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let pole = DVec4::new(-13.91, 4.325, -15.57, 1.);
        let weighted = pole * 0.707106781187;
        assert_ne!(weighted.z / weighted.w, pole.z);
        let mut surface = NDBSplineSurface::new(true, true, knots(), knots(), vec![
            vec![pole, pole + DVec4::new(1., 0., 0., 0.)],
            vec![weighted, weighted + DVec4::new(0., 1., 0., 0.)],
        ]);
        assert!(surface.rational_boundary_is_point(1, 0., 0.));
        surface.control_points[1][0].z = f64::from_bits(weighted.z.to_bits() + 1);
        assert!(!surface.rational_boundary_is_point(1, 0., 0.), "a resolved weighted displacement is not a pole");
    }

    #[test]
    fn collapsed_boundary_respects_declared_uncertainty_at_every_scale() {
        for scale in [1e-200, 1e-7, 1., 1e200] {
            let knots = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
            let surface = NDBSplineSurface::new(true, true, knots(), knots(), vec![
                vec![DVec4::new(0., 0., 0., 1.), DVec4::new(0., 0., scale, 1.)],
                vec![DVec4::new(2. * scale, 0., 0., 2.), DVec4::new(2. * scale, 0., 2. * scale, 2.)],
            ]);
            assert!(!surface.rational_boundary_is_point(1, 0., 0.));
            assert!(!surface.rational_boundary_is_point(1, 0., scale * 0.5));
            assert!(surface.rational_boundary_is_point(1, 0., scale));
        }
    }

    #[test]
    fn coincident_boundaries_require_matching_rational_basis_and_resolved_gap() {
        let knots = || KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]);
        let row = vec![DVec4::new(0., 0., 0., 1.), DVec4::new(0., 0.5, 0., 0.5), DVec4::new(0., 2., 0., 1.)];
        let mut controls = vec![row.clone(), row.clone(), row.clone()];
        for p in &mut controls[1] { p.x = p.w; }
        for p in &mut controls[2] { p.x = 1e-8 * p.w; }
        let mut surface = NDBSplineSurface::new(true, true, knots(), knots(), controls);
        assert!(!surface.rational_boundaries_coincide(0, 0.));
        assert!(!surface.rational_boundaries_coincide(0, 0.5e-8));
        assert!(surface.rational_boundaries_coincide(0, 1e-8));
        for p in &mut surface.control_points[2] { *p *= 2.; }
        assert!(surface.rational_boundaries_coincide(0, 1e-8));
        surface.control_points[2][1] *= 2.;
        assert!(!surface.rational_boundaries_coincide(0, 1e-8));
    }
}
