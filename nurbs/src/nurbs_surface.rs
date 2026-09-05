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

    fn derivs<const E: usize>(&self, uv: DVec2) -> Vec<Vec<DVec3>> {
        let (origin, mut derivs) = self.surface_derivs_relative::<E>(uv, crate::rational_difference);
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
