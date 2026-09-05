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

    /// Tests whether the represented Cartesian controls of a rational
    /// boundary iso-curve coincide. The last component is the weight.
    ///
    /// `parameter` is 0 for a fixed u and 1 for a fixed v.  Evaluating the
    /// fixed direction's basis (rather than selecting an end control row)
    /// also handles non-clamped knot vectors.
    pub fn rational_boundary_is_point(&self, parameter: usize, value: f64) -> bool {
        let controls: Vec<TVec<f64, D>> = match parameter {
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
            _ => return false,
        };
        let Some(reference) = controls.first() else { return false; };
        if D < 2 || reference[D - 1] == 0.0 { return false; }
        controls.iter().all(|point| {
            point[D - 1] != 0.0 && (0..D - 1).all(|i| {
                // Compare represented Cartesian controls, without a geometric
                // tolerance. Homogeneous controls can have different weights.
                point[i] / point[D - 1] == reference[i] / reference[D - 1]
            })
        })
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
        let p = self.u_knots.degree();
        let q = self.v_knots.degree();

        let uind = uspan - p;
        let mut S = TVec::zeros();
        for l in 0..=q {
            let mut temp = TVec::zeros();
            let vind = vspan - q + l;
            for k in 0..=p {
                temp += Nu[k] * self.control_points[uind + k][vind];
            }
            S += Nv[l] * temp;
        }
        S
    }

    /// Returns all derivatives of the surface.  If `D = surface_derivs()`,
    /// `D[k][l]` is the derivative of the surface `k` times in the `u`
    /// direction and `l` times in the `v` direction.
    ///
    /// We compute derivatives up to and including the `d`'th order derivatives.
    ///
    /// ALGORITHM A3.6
    pub fn surface_derivs<const E: usize>(&self, uv: DVec2) -> Vec<Vec<TVec<f64, D>>> {
        let p = self.u_knots.degree();
        let q = self.v_knots.degree();

        // Simple initialization of du
        let du = min(E, p);
        let dv = min(E, q);

        // The output matrix goes all the way to order d, even if some of the
        // surfaces are lower order (those values will be locked at 0)
        let mut SKL = vec![vec![TVec::zeros(); E + 1]; E + 1];

        let uspan = self.u_knots.find_span(uv.x);
        let Nu_deriv = self.u_knots.basis_funs_derivs_for_span(uspan, uv.x, du);

        let vspan = self.v_knots.find_span(uv.y);
        let Nv_deriv = self.v_knots.basis_funs_derivs_for_span(vspan, uv.y, dv);

        let mut temp = vec![TVec::zeros(); q + 1];
        for k in 0..=du {
            for s in 0..=q {
                temp[s] = TVec::zeros();
                for r in 0..=p {
                    temp[s] += Nu_deriv[k][r] * self.control_points[uspan - p + r][vspan - q + s];
                }
            }
            let dd = min(E - k, dv);
            for l in 0..=dd {
                for s in 0..=q {
                    SKL[k][l] += Nv_deriv[l][s] * temp[s];
                }
            }
        }
        SKL
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
    fn collapsed_boundary_uses_actual_nonclamped_endpoint_basis() {
        let controls = vec![
            vec![DVec4::new(3., 3., 4., 1.), DVec4::new(1., 3., 4., 1.), DVec4::new(5., 5., 4., 1.)],
            vec![DVec4::new(4., 3., 4., 1.), DVec4::new(0., 3., 4., 1.), DVec4::new(8., 5., 4., 1.)],
        ];
        let clamped = || KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
        let nonclamped = || KnotVector::from_multiplicities(2, &[-1., 0., 1., 2., 3., 4.], &[1; 6]);
        let surface = NDBSplineSurface::new(true, true, clamped(), nonclamped(), controls.clone());
        assert!(surface.rational_boundary_is_point(1, surface.min_v()));
        assert!(!surface.rational_boundary_is_point(1, surface.max_v()));
        let transposed = (0..3).map(|v| controls.iter().map(|row| row[v]).collect()).collect();
        let surface = NDBSplineSurface::new(true, true, nonclamped(), clamped(), transposed);
        assert!(surface.rational_boundary_is_point(0, surface.min_u()));
        assert!(!surface.rational_boundary_is_point(0, surface.max_u()));
    }
}
