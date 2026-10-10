use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4};

use crate::Error;
use nurbs::{AbstractCurve, NDBSplineCurve, SampledCurve};
use crate::surface::Surface;

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
    Line {
        origin: DVec3,
        direction: DVec3,
    },
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
                                     is_loop: bool, dir: bool, tolerance: f64) -> Result<Vec<DVec3>, Error>
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
        // Full loops start at the actual vertex, not the first knot. An open
        // curve cannot run against the sense, so if its ends meet within the
        // chord budget, the sense selects the arc across them too.
        let reversed = if dir { t_end < t_start } else { t_end > t_start };
        let wraps = is_loop || (reversed && (curve.is_closed()
            || (curve.point(curve.min_u()) - curve.point(curve.max_u())).norm() <= tolerance));
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
        let mut c = curve.polyline_with_tolerance(&ranges, tolerance)
            .ok_or(Error::InvalidGeometry("nonpositive rational curve weight"))?;
        // Both vertices project to one curve point when at least one is off
        // the curve. Join them with a chord, like any other vertex offset.
        if c.is_empty() {
            c.push(u);
        }
        Ok(Self::attach_endpoints(c, u, v))
    }

    fn attach_endpoints(mut c: Vec<DVec3>, u: DVec3, v: DVec3) -> Vec<DVec3> {
        // Keep resolved curve/vertex offsets: distinct edges can share the
        // same topological endpoints (for example the two sides of a sliver).
        let displaced = |a: DVec3, b: DVec3|
            (a-b).norm() > 64. * f64::EPSILON * (a.norm()+b.norm());
        if displaced(c[0], u) { c.insert(0,u); } else { c[0] = u; }
        if displaced(*c.last().unwrap(), v) { c.push(v); } else { *c.last_mut().unwrap() = v; }
        // Shared STEP vertices may differ from the curve within source
        // tolerance. Their replacement introduces bends which must participate
        // in reduction, even when the underlying spline is exactly straight.
        simplify_polyline(&mut c);
        c
    }

    pub fn build(&self, u: DVec3, v: DVec3, is_loop: bool, tolerance: f64) -> Result<Vec<DVec3>, Error> {
        match self {
            // A line does not close: an edge from a vertex back to itself
            // has zero length.
            Self::Line { .. } if is_loop => Ok(vec![u]),
            Self::Line { origin, direction } => {
                let project = |p: DVec3| {
                    let offset = p - origin;
                    p - (offset - direction * (offset.dot(direction) / direction.norm_squared()))
                };
                Ok(Self::attach_endpoints(vec![project(u), project(v)], u, v))
            }
            Self::BSplineCurveWithKnots { curve, dir } => Self::curve_points(u, v, curve, is_loop, *dir, tolerance),
            Self::NURBSCurve { curve, dir } => Self::curve_points(u, v, curve, is_loop, *dir, tolerance),
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
                let second_derivative = |t: f64| {
                    let p = if *hyperbola {
                        DVec4::new(t.cosh(), t.sinh(), 0.0, 0.0)
                    } else {
                        DVec4::new(2.0, 0.0, 0.0, 0.0)
                    };
                    glm::vec4_to_vec3(&(world_from_plane * p)).norm()
                };
                let mut parameters = vec![t0];
                let middle = (t0+t1)*0.5;
                let mut pending = vec![(middle,t1),(t0,middle)];
                while let Some((a,b)) = pending.pop() {
                    let bound = second_derivative(a).max(second_derivative(b)) * (b-a).powi(2) / 8.;
                    if !bound.is_finite() { return Err(Error::InvalidGeometry("nonfinite conic bound")); }
                    if bound > tolerance {
                        let m = (a+b)*0.5;
                        if m == a || m == b { return Err(Error::InvalidGeometry("conic resolution exhausted")); }
                        pending.push((m,b)); pending.push((a,m));
                    } else {
                        parameters.push(b);
                    }
                }
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

                // Linear interpolation error <= max|C''| * delta² / 8.
                // For this ellipse max|C''| is its larger semiaxis.
                let radius = world_from_eplane.column(0).xyz().norm()
                    .max(world_from_eplane.column(1).xyz().norm());
                let max_angle = (8. * tolerance / radius).sqrt().min(std::f64::consts::FRAC_PI_2);
                let count = (((u_ang-v_ang).abs()/max_angle).ceil() as usize + 1).max(3);

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
    fn chord_budget_scales_with_radius_and_does_not_alias_spline_inflections() {
        let tolerance = 0.01;
        for radius in [0.1, 2.5, 7.5, 100.] {
            let circle = Curve::new_circle(DVec3::zeros(), DVec3::z(), DVec3::x(), radius, true, true).unwrap();
            let points = circle.build(DVec3::x()*radius, DVec3::x()*radius, true, tolerance).unwrap();
            for edge in points.windows(2) {
                assert!(radius - ((edge[0]+edge[1])*0.5).norm() <= tolerance);
            }
        }
        let curve = NDBSplineCurve::new(true,
            nurbs::KnotVector::from_multiplicities(3, &[0.,1.], &[4,4]),
            vec![DVec3::zeros(), DVec3::new(1.,3.,0.), DVec3::new(2.,-3.,0.), DVec3::new(3.,0.,0.)]);
        let points = curve.polyline_with_tolerance(&[(0.,1.)], tolerance).unwrap();
        for i in 0..=1000 {
            let p = curve.point(i as f64/1000.);
            let distance = points.windows(2).map(|e| {
                let d = e[1]-e[0];
                (p-e[0]-d*((p-e[0]).dot(&d)/d.norm_squared()).clamp(0.,1.)).norm()
            }).fold(f64::INFINITY, f64::min);
            assert!(distance <= tolerance);
        }
    }

    #[test]
    fn closed_spline_trims_follow_edge_sense_and_start_at_the_vertex() {
        let curve = SampledCurve::new(NDBSplineCurve::new(false,
            nurbs::KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            vec![DVec3::zeros(), DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), DVec3::zeros()]));
        let a = DVec3::new(0.5, 0., 0.);
        let b = DVec3::new(0., 0.5, 0.);
        let forward = Curve::curve_points(a, b, &curve, false, true, 0.01).unwrap();
        let reverse = Curve::curve_points(a, b, &curve, false, false, 0.01).unwrap();
        assert_eq!(forward, vec![a, DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), b]);
        assert_eq!(reverse, vec![a, DVec3::zeros(), b]);
        assert_eq!(Curve::curve_points(b, a, &curve, false, true, 0.01).unwrap(),
            reverse.into_iter().rev().collect::<Vec<_>>());
        let full = Curve::curve_points(a, a, &curve, true, true, 0.01).unwrap();
        assert_eq!(full, vec![a, DVec3::x(), DVec3::new(1., 1., 0.), DVec3::y(), DVec3::zeros(), a]);
        assert_eq!(Curve::curve_points(a, a, &curve, true, false, 0.01).unwrap(),
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
            &SampledCurve::new(curve), false, true, 0.01).unwrap();
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
            let points = Curve::curve_points(a, b, &curve, false, true, 0.01).unwrap();
            assert_eq!(points.first(), Some(&a));
            assert_eq!(points.last(), Some(&b));
            assert_eq!(Curve::curve_points(b, a, &curve, false, true, 0.01).unwrap(),
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

        let forward = curve.build(point(-1.0), point(1.5), false, 0.01).unwrap();
        assert!(forward.len() > 2);
        assert_near(forward[0], point(-1.0));
        assert_near(*forward.last().unwrap(), point(1.5));

        let reverse = curve.build(point(1.5), point(-1.0), false, 0.01).unwrap();
        assert_eq!(forward.len(), reverse.len());
        for (a, b) in forward.iter().zip(reverse.iter().rev()) {
            assert_near(*a, *b);
        }

        assert!(curve.build(DVec3::new(3.0, 4.0, 3.0), point(1.0), false, 0.01).is_err());
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

        let forward = curve.build(point(-2.0), point(1.0), false, 0.01).unwrap();
        assert!(forward.len() > 2);
        let reverse = curve.build(point(1.0), point(-2.0), false, 0.01).unwrap();
        assert_eq!(forward.len(), reverse.len());
        for (a, b) in forward.iter().zip(reverse.iter().rev()) {
            assert_near(*a, *b);
        }
        assert_eq!(Curve::new_parabola(
            DVec3::zeros(), DVec3::z(), DVec3::x(), 0.0,
        ).unwrap_err(), Error::InvalidGeometry("parabola focal distance is zero"));
    }
}
