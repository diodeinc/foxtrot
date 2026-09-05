use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4};

use crate::Error;
use nurbs::{AbstractCurve, NDBSplineCurve, SampledCurve};
use crate::surface::Surface;

const BSPLINE_POINTS_PER_KNOT: usize = 8;
const ELLIPSE_SAMPLES_PER_REV: usize = 32;
const CONIC_MAX_DEPTH: usize = 20;

// Remove only vertices whose deletion leaves the represented 3D polyline
// exactly unchanged. In particular, preserve corners and collinear reversals.
fn simplify_polyline(points: &mut Vec<DVec3>) {
    let between = |a: DVec3, b: DVec3, c: DVec3| {
        if (0..3).any(|i| b[i] < a[i].min(c[i]) || b[i] > a[i].max(c[i])) {
            return false;
        }
        // Stay within the exact orientation predicates' exponent envelope.
        // Outside it, retain samples rather than risk deleting real curvature.
        if [a, b, c].iter().flat_map(|p| p.iter()).any(|&v|
            !v.is_finite() || (v != 0.0 && (v.abs() < 2.0_f64.powi(-142) || v.abs() > 2.0_f64.powi(201)))) {
            return false;
        }
        (0..3).all(|i| {
            let project = |p: DVec3| robust::Coord { x: p[i], y: p[(i + 1) % 3] };
            robust::orient2d(project(a), project(b), project(c)) == 0.0
        })
    };
    let mut kept = 0;
    for i in 0..points.len() {
        let p = points[i];
        while kept >= 2 && between(points[kept - 2], points[kept - 1], p) {
            kept -= 1;
        }
        points[kept] = p;
        kept += 1;
    }
    points.truncate(kept);
}

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

    fn curve_points<const N: usize>(u: DVec3, v: DVec3, curve: &SampledCurve<N>,
                                     is_loop: bool, dir: bool) -> Result<Vec<DVec3>, Error>
        where NDBSplineCurve<N>: AbstractCurve
    {
        let t_start = curve.u_from_point(u)
            .ok_or(Error::InvalidGeometry("curve start projection did not converge"))?;
        let t_end = if is_loop { t_start } else {
            curve.u_from_point(v)
                .ok_or(Error::InvalidGeometry("curve end projection did not converge"))?
        };
        // A closed curve has two arcs between its endpoints. EDGE_CURVE's
        // same_sense selects the directed arc, including traversal of the cut.
        // Full loops start at the actual vertex, not the first knot.
        let wraps = is_loop || (curve.is_closed()
            && if dir { t_end < t_start } else { t_end > t_start });
        let mut ranges = if wraps {
            let (exit, entry) = if dir { (curve.max_u(), curve.min_u()) }
                                else { (curve.min_u(), curve.max_u()) };
            vec![(t_start, exit), (entry, t_end)]
        } else {
            vec![(t_start, t_end)]
        };
        if wraps {
            ranges.retain(|&(a, b)| a != b);
        }
        let mut c = curve.as_polyline(&ranges, BSPLINE_POINTS_PER_KNOT);
        if c.is_empty() {
            return Err(Error::InvalidGeometry("curve polyline is empty"));
        }
        c[0] = u;
        if let Some(last) = c.last_mut() {
            *last = v;
        }
        // Shared STEP vertices may differ from the curve within source
        // tolerance. Their replacement introduces bends which must participate
        // in reduction, even when the underlying spline is exactly straight.
        simplify_polyline(&mut c);
        Ok(c)
    }

    pub fn build(&self, u: DVec3, v: DVec3, is_loop: bool) -> Result<Vec<DVec3>, Error> {
        match self {
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

    #[test]
    fn closed_spline_trims_follow_edge_sense_and_start_at_the_vertex() {
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            nurbs::KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            vec![DVec3::zeros(), DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), DVec3::zeros()]));
        let a = DVec3::new(0.5, 0., 0.);
        let b = DVec3::new(0., 0.5, 0.);
        let forward = Curve::curve_points(a, b, &curve, false, true).unwrap();
        let reverse = Curve::curve_points(a, b, &curve, false, false).unwrap();
        assert_eq!(forward, vec![a, DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), b]);
        assert_eq!(reverse, vec![a, DVec3::zeros(), b]);
        assert_eq!(Curve::curve_points(b, a, &curve, false, true).unwrap(),
            reverse.into_iter().rev().collect::<Vec<_>>());
        let full = Curve::curve_points(a, a, &curve, true, true).unwrap();
        assert_eq!(full, vec![a, DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), DVec3::zeros(), a]);
        assert_eq!(Curve::curve_points(a, a, &curve, true, false).unwrap(),
            full.into_iter().rev().collect::<Vec<_>>());
    }

    #[test]
    fn polyline_reduction_preserves_bends_reversals_and_small_curvature() {
        let a = DVec3::zeros();
        let b = DVec3::new(1., 2., 3.);
        let c = b * 2.;
        let mut line = vec![a, b, c];
        simplify_polyline(&mut line);
        assert_eq!(line, vec![a, c]);
        for points in [vec![a, b, a], vec![a, b, DVec3::new(2., 4., 6. + 1e-14)],
                       vec![a, b * 1e-200, DVec3::new(2e-200, 4e-200, 7e-200)]] {
            let mut reduced = points.clone();
            simplify_polyline(&mut reduced);
            assert_eq!(reduced, points);
        }
    }

    #[test]
    fn straight_high_degree_trims_do_not_create_redundant_vertices() {
        let curve = NDBSplineCurve::new(true,
            nurbs::KnotVector::from_multiplicities(3, &[0., 1.], &[4, 4]),
            (0..4).map(|i| DVec3::new(i as f64, -0.107370820668693, 0.4)).collect());
        let endpoints = [curve.point(0.4), curve.point(0.40000001)];
        let points = Curve::curve_points(endpoints[0], endpoints[1],
            &SampledCurve::new(curve), false, true).unwrap();
        assert_eq!(points, endpoints);
    }

    #[test]
    fn reduction_preserves_bends_at_topological_endpoints() {
        for degree in [1, 3] {
            let curve = SampledCurve::new(NDBSplineCurve::new(true,
                nurbs::KnotVector::from_multiplicities(degree, &[0., 1.], &[degree + 1, degree + 1]),
                (0..=degree).map(|i| DVec3::new(3. * i as f64 / degree as f64, 0., 0.)).collect()));
            let a = DVec3::new(0., 0.01, 0.);
            let b = DVec3::new(3., 0.01, 0.);
            let points = Curve::curve_points(a, b, &curve, false, true).unwrap();
            assert_eq!(points, vec![a, DVec3::new(0.375, 0., 0.), DVec3::new(2.625, 0., 0.), b]);
            assert_eq!(Curve::curve_points(b, a, &curve, false, true).unwrap(),
                points.into_iter().rev().collect::<Vec<_>>());
        }
    }

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
