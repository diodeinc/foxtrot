use nalgebra_glm::{DVec2, DVec3};
use crate::{abstract_surface::AbstractSurface, nd_surface::NDBSplineSurface, VecF};

pub type NURBSSurface = NDBSplineSurface<4>;

impl AbstractSurface for NURBSSurface {
    fn point(&self, uv: DVec2) -> DVec3 {
        let uspan = self.u_knots.find_span(uv.x);
        let Nu = self.u_knots.basis_funs_for_span(uspan, uv.x);
        let vspan = self.v_knots.find_span(uv.y);
        let Nv = self.v_knots.basis_funs_for_span(vspan, uv.y);
        self.point_from_basis(uspan, &Nu, vspan, &Nv)
    }
    fn point_from_basis(&self, uspan: usize, Nu: &VecF,
                               vspan: usize, Nv: &VecF) -> DVec3
    {
        let (origin, p) = self.surface_point_relative(uspan, Nu, vspan, Nv, crate::rational_difference);
        p.xyz() / (p.w + origin.w) + origin.xyz() / origin.w
    }

    fn control_bounds(&self, spans: [usize; 2]) -> [DVec3; 2] {
        self.span_control_bounds(spans, |p| p.xyz() / p.w)
    }

    fn derivs_relative_to<const E: usize>(&self, uv: DVec2, reference: DVec3) -> Vec<Vec<DVec3>> {
        self.derivs_in_span::<E>(uv, [self.u_knots.find_span(uv.x), self.v_knots.find_span(uv.y)], reference)
    }

    fn derivs_in_span<const E: usize>(&self, uv: DVec2, spans: [usize; 2], reference: DVec3) -> Vec<Vec<DVec3>> {
        let shift = |p: nalgebra_glm::DVec4| nalgebra_glm::DVec4::new(
            (-reference.x).mul_add(p.w, p.x),
            (-reference.y).mul_add(p.w, p.y),
            (-reference.z).mul_add(p.w, p.z), p.w);
        let (origin, mut derivs) = self.surface_derivs_relative::<E>(uv, spans,
            |p, origin| crate::rational_difference(shift(p), shift(origin)));
        let origin = shift(origin);
        derivs[0][0].w += origin.w;
        let mut SKL = vec![vec![DVec3::zeros(); E + 1]; E + 1];
        let bin = |a, b| num_integer::binomial(a, b) as f64;
        for k in 0..=E {
            for l in 0..=(E - k) {
                let mut v = derivs[k][l].xyz();
                for j in 1..=l {
                    v -= bin(l, j) * derivs[0][j].w * SKL[k][l - j];
                }
                for i in 1..=k {
                    v -= bin(k, i) * derivs[i][0].w * SKL[k - i][l];
                    let mut v2 = DVec3::zeros();
                    for j in 1..=l {
                        v2 += bin(l, j) * derivs[i][j].w * SKL[k - i][l - j];
                    }
                    v -= bin(k, i) * v2;
                }
                SKL[k][l] = v / derivs[0][0].w;
            }
        }
        SKL[0][0] += origin.xyz() / origin.w;
        SKL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotVector;
    use nalgebra_glm::DVec4;

    #[test]
    fn knot_cell_derivatives_preserve_both_sides_of_a_crease() {
        let surface = NURBSSurface::new(true, true,
            KnotVector::from_multiplicities(1, &[0., 0.5, 1.], &[2, 1, 2]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(0., 0.5), (0.5, 0.), (1., 0.5)].iter().map(|&(x, z)|
                vec![DVec4::new(x, 0., z, 1.), DVec4::new(x, 1., z, 1.)]).collect());
        let uv = DVec2::new(0.5, 0.3);
        let left = surface.derivs_in_span::<1>(uv, [1, 1], DVec3::zeros());
        let right = surface.derivs_in_span::<1>(uv, [2, 1], DVec3::zeros());
        assert_eq!(left[0][0], right[0][0]);
        assert_eq!(left[1][0], DVec3::new(1., 0., -1.));
        assert_eq!(right[1][0], DVec3::new(1., 0., 1.));
        assert_eq!(surface.derivs::<1>(uv), right);
    }

    #[test]
    fn relative_jets_retain_displacements_below_world_coordinate_precision() {
        fn check(surface: impl AbstractSurface) {
            let uv = DVec2::new(0.25 + 1e-10, 0.3);
            let reference = DVec3::new(1e9 + 0.25, 0.3, 0.);
            assert_eq!(surface.point(uv), reference);
            let jet = surface.derivs_relative_to::<1>(uv, reference);
            assert!((jet[0][0].x - (uv.x - 0.25)).abs() < 1e-16);
            assert_eq!(jet[1][0], DVec3::x());
            assert_eq!(jet[0][1], DVec3::y());
        }
        let knots = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let controls: Vec<Vec<_>> = [0., 1.].iter().map(|&u|
            [0., 1.].iter().map(|&v| DVec3::new(1e9 + u, v, 0.)).collect()).collect();
        check(crate::BSplineSurface::new(true, true, knots(), knots(), controls.clone()));
        check(NURBSSurface::new(true, true, knots(), knots(), controls.iter().map(|row|
            row.iter().map(|p| DVec4::new(p.x, p.y, p.z, 1.)).collect()).collect()));
    }

    #[test]
    fn rational_constant_coordinates_have_zero_derivatives() {
        let z = 1.849999999971;
        let knots = || KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]);
        let weights = [1., 0.5, 1.];
        let surface = NURBSSurface::new(true, true, knots(), knots(),
            (0..3).map(|i| (0..3).map(|j|
                DVec4::new(i as f64 * 1e-16, j as f64, z, 1.) * (weights[i] * weights[j]))
                .collect()).collect());
        for i in 0..=32 {
            let uv = DVec2::new(i as f64 / 32., 0.37);
            assert_eq!(surface.point(uv).z, z);
            let d = surface.derivs::<3>(uv);
            assert_eq!(d[0][0].z, z);
            for k in 0..=3 { for l in 0..=3-k {
                if k + l > 0 { assert_eq!(d[k][l].z, 0.); }
            } }
        }
    }
}
