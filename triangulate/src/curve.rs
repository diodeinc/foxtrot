use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4};

use crate::Error;
use nurbs::{AbstractCurve, NDBSplineCurve, SampledCurve};
use crate::surface::Surface;

const BSPLINE_POINTS_PER_KNOT: usize = 8;
const ELLIPSE_SAMPLES_PER_REV: usize = 32;
const CONIC_MAX_DEPTH: usize = 20;

#[derive(Debug)]
pub enum Curve {
    // TODO: move this to a standalone struct?
    Ellipse {
        eplane_from_world: DMat4,
        world_from_eplane: DMat4,
        closed: bool,
        dir: bool
    },
    OpenConic {
        plane_from_world: DMat4,
        world_from_plane: DMat4,
        hyperbola: bool,
    },
    Line,
    BSplineCurveWithKnots {
        curve: SampledCurve<3>,
        dir: bool,
    },
    NURBSCurve {
        curve: SampledCurve<4>,
        dir: bool,
    },
}

impl Curve {
    pub fn new_ellipse(location: DVec3, axis: DVec3, ref_direction: DVec3,
                       radius1: f64, radius2: f64, closed: bool, dir: bool)
        -> Result<Self, Error>
    {
        // Build a rotation matrix to go from flat (XY) to 3D space
        let world_from_eplane = Surface::make_affine_transform(axis,
            radius1 * ref_direction,
            radius2 * axis.cross(&ref_direction),
            location);
        let eplane_from_world = world_from_eplane
            .try_inverse()
            .ok_or(Error::SingularTransform("ellipse transform"))?;
        Ok(Self::Ellipse {
            world_from_eplane,
            eplane_from_world,
            closed, dir
        })
    }

    pub fn new_circle(location: DVec3, axis: DVec3, ref_direction: DVec3,
                      radius: f64, closed: bool, dir: bool) -> Result<Self, Error> {
        Self::new_ellipse(location, axis, ref_direction,
                          radius, radius, closed, dir)
    }

    fn new_open_conic(location: DVec3, axis: DVec3, ref_direction: DVec3,
                      x_scale: f64, y_scale: f64, hyperbola: bool)
        -> Result<Self, Error>
    {
        let world_from_plane = Surface::make_affine_transform(
            axis, x_scale * ref_direction, y_scale * axis.cross(&ref_direction), location);
        let plane_from_world = world_from_plane
            .try_inverse()
            .ok_or(Error::SingularTransform("open conic transform"))?;
        Ok(Self::OpenConic { plane_from_world, world_from_plane, hyperbola })
    }

    pub fn new_hyperbola(location: DVec3, axis: DVec3, ref_direction: DVec3,
                         semi_axis: f64, semi_imag_axis: f64) -> Result<Self, Error> {
        Self::new_open_conic(location, axis, ref_direction,
                             semi_axis, semi_imag_axis, true)
    }

    pub fn new_parabola(location: DVec3, axis: DVec3, ref_direction: DVec3,
                        focal_dist: f64) -> Result<Self, Error> {
        if focal_dist == 0.0 {
            return Err(Error::InvalidGeometry("parabola focal distance is zero"));
        }
        Self::new_open_conic(location, axis, ref_direction,
                             focal_dist, 2.0 * focal_dist, false)
    }

    pub fn new_line() -> Self {
        Self::Line
    }

    fn curve_points<const N: usize>(u: DVec3, v: DVec3, curve: &SampledCurve<N>,
                                     is_loop: bool, dir: bool) -> Result<Vec<DVec3>, Error>
        where NDBSplineCurve<N>: AbstractCurve
    {
        let (t_start, t_end) = if is_loop {
            // Full-loop edge: sample the entire parameter range.  For closed
            // spline curves, the start and end vertices are identical, so the
            // direction cannot be recovered from their parameters; use the
            // oriented edge / same_sense direction instead.
            if dir {
                (curve.min_u(), curve.max_u())
            } else {
                (curve.max_u(), curve.min_u())
            }
        } else {
            (curve.u_from_point(u), curve.u_from_point(v))
        };
        let mut c = curve.as_polyline(t_start, t_end, BSPLINE_POINTS_PER_KNOT);
        if c.is_empty() {
            return Err(Error::InvalidGeometry("curve polyline is empty"));
        }
        c[0] = u;
        if let Some(last) = c.last_mut() {
            *last = v;
        }
        Ok(c)
    }

    pub fn build(&self, u: DVec3, v: DVec3, is_loop: bool) -> Result<Vec<DVec3>, Error> {
        match self {
            Self::Line => Ok(vec![u, v]),
            Self::BSplineCurveWithKnots { curve, dir } => Self::curve_points(u, v, curve, is_loop, *dir),
            Self::NURBSCurve { curve, dir } => Self::curve_points(u, v, curve, is_loop, *dir),
            Self::OpenConic { plane_from_world, world_from_plane, hyperbola } => {
                let local = |p: DVec3| plane_from_world * DVec4::new(p.x, p.y, p.z, 1.0);
                let a = local(u);
                let b = local(v);
                if *hyperbola && (a.x < -1e-9 || b.x < -1e-9) {
                    return Err(Error::InvalidGeometry("hyperbola endpoint is on negative branch"));
                }
                let t0 = if *hyperbola { a.y.asinh() } else { a.y };
                let t1 = if *hyperbola { b.y.asinh() } else { b.y };
                if !t0.is_finite() || !t1.is_finite() {
                    return Err(Error::InvalidGeometry("open conic parameter is not finite"));
                }

                let eval = |t: f64| {
                    let p = if *hyperbola {
                        DVec4::new(t.cosh(), t.sinh(), 0.0, 1.0)
                    } else {
                        DVec4::new(t * t, t, 0.0, 1.0)
                    };
                    glm::vec4_to_vec3(&(world_from_plane * p))
                };
                let tangent = |t: f64| {
                    let p = if *hyperbola {
                        DVec4::new(t.sinh(), t.cosh(), 0.0, 0.0)
                    } else {
                        DVec4::new(2.0 * t, 1.0, 0.0, 0.0)
                    };
                    glm::normalize(&glm::vec4_to_vec3(&(world_from_plane * p)))
                };
                let max_angle = 2.0 * std::f64::consts::PI / ELLIPSE_SAMPLES_PER_REV as f64;
                let mut parameters = vec![t0];
                fn subdivide<F: Fn(f64) -> DVec3>(
                    out: &mut Vec<f64>, tangent: &F, a: f64, b: f64,
                    max_angle: f64, depth: usize,
                ) {
                    let angle = tangent(a).dot(&tangent(b)).clamp(-1.0, 1.0).acos();
                    if angle > max_angle && depth < CONIC_MAX_DEPTH {
                        let m = (a + b) * 0.5;
                        subdivide(out, tangent, a, m, max_angle, depth + 1);
                        subdivide(out, tangent, m, b, max_angle, depth + 1);
                    } else {
                        out.push(b);
                    }
                }
                subdivide(&mut parameters, &tangent, t0, t1, max_angle, 0);
                let mut out: Vec<_> = parameters.into_iter().map(eval).collect();
                out[0] = u;
                *out.last_mut().unwrap() = v;
                Ok(out)
            },
            Self::Ellipse {
                eplane_from_world, world_from_eplane, closed, dir
            } => {
                // Project from 3D into the "ellipse plane".  In the "eplane",
                // the ellipse lies on the unit circle.
                let u_eplane = eplane_from_world *
                               DVec4::new(u.x, u.y, u.z, 1.0);
                let v_eplane = eplane_from_world *
                               DVec4::new(v.x, v.y, v.z, 1.0);

                // Pick the starting angle in the circle's flat plane
                let u_ang = u_eplane.y.atan2(u_eplane.x);
                let mut v_ang = v_eplane.y.atan2(v_eplane.x);
                const PI2: f64 = 2.0 * std::f64::consts::PI;
                if *closed {
                    if *dir {
                        v_ang = u_ang + PI2;
                    } else {
                        v_ang = u_ang - PI2;
                    }
                } else if *dir && v_ang <= u_ang {
                    v_ang += PI2;
                } else if !*dir && v_ang >= u_ang {
                    v_ang -= PI2;
                }

                let count = 4.max(
                    (ELLIPSE_SAMPLES_PER_REV as f64 * (u_ang - v_ang).abs() /
                    (2.0 * std::f64::consts::PI)).round() as usize);

                let mut out_world = vec![u];
                // Walk around the circle, using the true positions for start
                // and end points to improve numerical accuracy.
                for i in 1..(count - 1) {
                    let frac = (i as f64) / ((count - 1) as f64);
                    let ang = u_ang * (1.0 - frac) + v_ang * frac;
                    let pos_eplane = DVec4::new(ang.cos(), ang.sin(), 0.0, 1.0);

                    // Project back into 3D
                    let p = world_from_eplane * pos_eplane;
                    out_world.push(glm::vec4_to_vec3(&p));
                }
                out_world.push(v);
                Ok(out_world)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(a: DVec3, b: DVec3) {
        assert!((a - b).norm() < 1e-10, "{:?} != {:?}", a, b);
    }

    #[test]
    fn hyperbola_uses_positive_branch_and_endpoint_parameter_direction() {
        let curve = Curve::new_hyperbola(
            DVec3::new(3.0, 4.0, 5.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            2.0, 0.5,
        ).unwrap();
        let point = |t: f64| DVec3::new(3.0 + 0.5 * t.sinh(), 4.0, 5.0 + 2.0 * t.cosh());

        let forward = curve.build(point(-1.0), point(1.5), false).unwrap();
        assert!(forward.len() > 2);
        assert_near(forward[0], point(-1.0));
        assert_near(*forward.last().unwrap(), point(1.5));

        let reverse = curve.build(point(1.5), point(-1.0), false).unwrap();
        assert_eq!(forward.len(), reverse.len());
        for (a, b) in forward.iter().zip(reverse.iter().rev()) {
            assert_near(*a, *b);
        }

        assert!(curve.build(DVec3::new(3.0, 4.0, 3.0), point(1.0), false).is_err());
    }

    #[test]
    fn parabola_supports_negative_focal_distance_and_both_directions() {
        let curve = Curve::new_parabola(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 1.0, 0.0),
            -2.0,
        ).unwrap();
        let point = |t: f64| DVec3::new(1.0 + 4.0 * t, 2.0 - 2.0 * t * t, 3.0);

        let forward = curve.build(point(-2.0), point(1.0), false).unwrap();
        assert!(forward.len() > 2);
        let reverse = curve.build(point(1.0), point(-2.0), false).unwrap();
        assert_eq!(forward.len(), reverse.len());
        for (a, b) in forward.iter().zip(reverse.iter().rev()) {
            assert_near(*a, *b);
        }
        assert_eq!(Curve::new_parabola(
            DVec3::zeros(), DVec3::z(), DVec3::x(), 0.0,
        ).unwrap_err(), Error::InvalidGeometry("parabola focal distance is zero"));
    }
}
