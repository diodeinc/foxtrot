use std::f64::{EPSILON, consts::PI};

use nalgebra_glm as glm;
use glm::{DVec2, DVec3, DVec4, DMat4};

use nurbs::{AbstractSurface, SampledSurface};
use crate::{Error, mesh::Vertex};

#[derive(Debug, Clone)]
pub enum SplineChart {
    Cartesian { v_scale: f64 },
    Polar {
        periodic: usize,
        periodic_min: f64,
        period: f64,
        radial_origin: f64,
        radial_scale: f64,
        radial_min: f64,
        radial_max: f64,
    },
}

impl SplineChart {
    fn lower(&self, raw: DVec2) -> DVec2 {
        match *self {
            Self::Cartesian { v_scale } => DVec2::new(raw.x, raw.y * v_scale),
            Self::Polar { periodic, periodic_min, period, radial_origin, radial_scale, .. } => {
                let other = 1 - periodic;
                let radius = (raw[other] - radial_origin) / radial_scale;
                let angle = 2.0 * PI * (raw[periodic] - periodic_min).rem_euclid(period) / period;
                let mut mapped = DVec2::zeros();
                mapped[periodic] = radius * angle.cos();
                mapped[other] = -radial_scale.signum() * radius * angle.sin();
                mapped
            },
        }
    }

    fn raw(&self, mapped: DVec2) -> Option<DVec2> {
        match *self {
            Self::Cartesian { v_scale } => Some(DVec2::new(mapped.x, mapped.y / v_scale)),
            Self::Polar { periodic, periodic_min, period, radial_origin, radial_scale,
                          radial_min, radial_max } => {
                let other = 1 - periodic;
                let radius = mapped.norm();
                let radial = radial_origin + radial_scale * radius;
                // Account only for arithmetic error in a chart roundtrip.
                // Sampling outside the represented radial domain is rejected.
                let roundoff = 8.0 * EPSILON
                    * (radial_origin.abs() + (radial_scale * radius).abs());
                if radial < radial_min - roundoff || radial > radial_max + roundoff {
                    return None;
                }
                let sin = -radial_scale.signum() * mapped[other];
                let angle = sin.atan2(mapped[periodic]).rem_euclid(2.0 * PI);
                let mut raw = DVec2::zeros();
                raw[periodic] = periodic_min + period * angle / (2.0 * PI);
                raw[other] = radial.clamp(radial_min, radial_max);
                Some(raw)
            },
        }
    }

}

// Represents a surface in 3D space, with a function to project a 3D point
// on the surface down to a 2D space.
#[derive(Debug, Clone)]
pub enum Surface {
    Cylinder {
        location: DVec3,
        axis: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    Plane {
        normal: DVec3,
        coordinates: [usize; 2],
        orientation: f64,
    },
    Cone {
        mat: DMat4,
        mat_i: DMat4,
        angle: f64,
    },
    NURBS { surf: SampledSurface<4>, chart: SplineChart, pole: Option<(DVec3, f64)> },
    Sphere {
        location: DVec3,
        mat: DMat4,     // uv to world
        mat_i: DMat4,   // world to uv
        radius: f64,
    },
    Torus {
        axis: DVec3,
        location: DVec3,
        mat: DMat4,
        mat_i: DMat4,
        major_radius: f64,
        minor_radius: f64,
        polar_major: bool,
        radial_start: f64,
    },
}

impl Surface {
    pub fn new_nurbs(surf: SampledSurface<4>, uncertainty: f64) -> Self {
        if let Some(normal) = surf.surf.bilinear_plane_normal() {
            return Self::new_plane(normal, DVec3::zeros(), DVec3::zeros())
                .expect("regular planar patch has a nonzero finite normal");
        }
        let u_periodic = !surf.surf.u_open;
        let v_periodic = !surf.surf.v_open;
        let chart = if u_periodic ^ v_periodic {
            let periodic = if u_periodic { 0 } else { 1 };
            let radial = 1 - periodic;
            let (radial_min, radial_max) = if radial == 0 {
                (surf.surf.min_u(), surf.surf.max_u())
            } else {
                (surf.surf.min_v(), surf.surf.max_v())
            };
            let min_point = surf.surf.rational_boundary_is_point(radial, radial_min, uncertainty);
            let max_point = surf.surf.rational_boundary_is_point(radial, radial_max, uncertainty);
            if min_point && max_point {
                SplineChart::Cartesian { v_scale: surf.surf.aspect_ratio() }
            } else {
                let span = radial_max - radial_min;
                let (radial_origin, radial_scale) = if min_point {
                    (radial_min, span)
                } else if max_point {
                    (radial_max, -span)
                } else {
                    (radial_min - span, span)
                };
                let (periodic_min, periodic_max) = if periodic == 0 {
                    (surf.surf.min_u(), surf.surf.max_u())
                } else {
                    (surf.surf.min_v(), surf.surf.max_v())
                };
                SplineChart::Polar { periodic, periodic_min,
                    period: periodic_max - periodic_min, radial_origin, radial_scale,
                    radial_min, radial_max }
            }
        } else {
            SplineChart::Cartesian { v_scale: surf.surf.aspect_ratio() }
        };
        let pole = match chart {
            SplineChart::Polar { periodic, periodic_min, radial_origin, radial_min, radial_max, .. }
                if radial_origin == radial_min || radial_origin == radial_max => {
                    let mut raw = DVec2::zeros();
                    raw[periodic] = periodic_min;
                    raw[1 - periodic] = radial_origin;
                    Some((surf.surf.point(raw), uncertainty))
                },
            _ => None,
        };
        Surface::NURBS { surf, chart, pole }
    }

    fn fallback_perpendicular(axis: DVec3) -> DVec3 {
        let candidate = if axis.x.abs() < 0.9 {
            DVec3::new(1.0, 0.0, 0.0)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        };
        (candidate - axis * candidate.dot(&axis)).normalize()
    }

    pub fn new_sphere(location: DVec3, radius: f64) -> Result<Self, Error> {
        Ok(Surface::Sphere {
            // mat and mat_i are built in prepare()
            mat: DMat4::identity(),
            mat_i: DMat4::identity(),
            location, radius,
        })
    }
    pub fn new_cylinder(axis: DVec3, ref_direction: DVec3, location: DVec3, radius: f64)
        -> Result<Self, Error>
    {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        let mat_i = mat.try_inverse()
            .ok_or(Error::SingularTransform("cylinder transform"))?;
        Ok(Surface::Cylinder {
            mat,
            mat_i,
            axis, radius, location,
            z_min: 0.0,
            z_max: 0.0,
        })
    }

    pub fn new_torus(location: DVec3, axis: DVec3,
                     major_radius: f64, minor_radius: f64) -> Result<Self, Error>
    {
        let ref_direction = Self::fallback_perpendicular(axis);
        Self::new_torus_with_ref_direction(
            location, axis, ref_direction, major_radius, minor_radius)
    }

    pub fn new_torus_with_ref_direction(location: DVec3, axis: DVec3,
                     ref_direction: DVec3,
                     major_radius: f64, minor_radius: f64) -> Result<Self, Error>
    {
        // Torus parameterization uses local X as the revolution axis and
        // local Z as the zero-angle radial direction.
        let mat = Self::make_rigid_transform(ref_direction, axis, location);
        let mat_i = mat.try_inverse()
            .ok_or(Error::SingularTransform("torus transform"))?;
        Ok(Surface::Torus {
            mat, mat_i, location, axis, major_radius, minor_radius,
            polar_major: true, radial_start: 0.0,
        })
    }

    pub fn new_plane(axis: DVec3, _ref_direction: DVec3, _location: DVec3) -> Result<Self, Error> {
        let dropped = axis.iamax();
        let scale = axis[dropped].abs();
        if scale == 0.0 || !axis.iter().all(|x| x.is_finite()) {
            return Err(Error::SingularTransform("plane normal"));
        }
        // Project onto the best-conditioned coordinate plane. Copying two
        // coordinates preserves their exact predicates, unlike a rounded
        // inverse affine transform. Metric distortion is at most sqrt(3).
        Ok(Surface::Plane {
            normal: (axis / scale).normalize(),
            coordinates: [(dropped + 1) % 3, (dropped + 2) % 3],
            orientation: axis[dropped].signum(),
        })
    }

    pub fn new_cone(axis: DVec3, ref_direction: DVec3, location: DVec3, angle: f64)
        -> Result<Self, Error>
    {
        let mat = Self::make_rigid_transform(axis, ref_direction, location);
        let mat_i = mat.try_inverse()
            .ok_or(Error::SingularTransform("cone transform"))?;
        Ok(Surface::Cone {
            mat,
            mat_i,
            angle,
        })
    }

    pub fn make_affine_transform(z_world: DVec3, x_world: DVec3, y_world: DVec3, origin_world: DVec3) -> DMat4 {
        let mut mat = DMat4::identity();
        mat.set_column(0, &glm::vec3_to_vec4(&x_world));
        mat.set_column(1, &glm::vec3_to_vec4(&y_world));
        mat.set_column(2, &glm::vec3_to_vec4(&z_world));
        mat.set_column(3, &glm::vec3_to_vec4(&origin_world));
        mat[(3, 3)] = 1.0;
        mat
    }

    fn make_rigid_transform(z_world: DVec3, x_world: DVec3, origin_world: DVec3) -> DMat4 {
        // STEP inputs are not always perfectly orthogonal, and some models
        // include degenerate ref-directions. Build a stable orthonormal basis
        // so downstream lowering/inversion does not fail on slightly bad input.
        let z = if z_world.norm_squared() > EPSILON {
            z_world.normalize()
        } else {
            DVec3::new(0.0, 0.0, 1.0)
        };
        let x_proj = x_world - z * x_world.dot(&z);
        let x = if x_proj.norm_squared() > EPSILON {
            x_proj.normalize()
        } else {
            Self::fallback_perpendicular(z)
        };
        let y = z.cross(&x).normalize();
        let x = y.cross(&z).normalize();
        Self::make_affine_transform(z, x, y, origin_world)
    }

    fn surf_lower(p: DVec3, surf: &SampledSurface<4>) -> Result<DVec2, Error> {
        surf.uv_from_point(p).ok_or(Error::CouldNotLower)
    }

    fn spline_raw(surf: &SampledSurface<4>, chart: &SplineChart,
                  mapped: DVec2) -> Option<DVec2> {
        chart.raw(mapped).map(|raw| DVec2::new(
            Self::spline_parameter(raw.x, surf.surf.min_u(), surf.surf.max_u(), surf.surf.u_open),
            Self::spline_parameter(raw.y, surf.surf.min_v(), surf.surf.max_v(), surf.surf.v_open),
        ))
    }

    /// Lowers a 3D point on a specific surface into a 2D space defined by
    /// the surface type.  This should only be called from `lower_verts`,
    /// to ensure that `prepare` is called first.
    fn lower(&self, p: DVec3) -> Result<DVec2, Error> {
        let p_ = DVec4::new(p.x, p.y, p.z, 1.0);
        match self {
            Surface::Plane { coordinates, orientation, .. } => {
                Ok(DVec2::new(p[coordinates[0]], p[coordinates[1]] * orientation))
            },
            Surface::Cone { mat_i, .. } => {
                let xy = glm::vec4_to_vec2(&(mat_i * p_));
                Ok(DVec2::new(-xy.x, xy.y))
            },

            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                let p = mat_i * p_;
                // We convert the Z coordinates to either add or subtract from
                // the radius, so that we maintain the right topology (instead
                // of doing something like theta-z coordinates, which wrap
                // around awkwardly).

                // Scale from radius=1 to radius=0.5 based on Z
                let dz = z_max - z_min;
                if dz.abs() < EPSILON {
                    return Err(Error::InvalidGeometry("cylinder has zero height"));
                }
                let z = (p.z - z_min) / dz;
                let scale = 1.0 / (1.0 + z);
                Ok(DVec2::new(p.x * scale, p.y * scale))
            },
            Surface::Torus { mat_i, major_radius, minor_radius,
                             polar_major, radial_start, .. } => {
                if major_radius.abs() < EPSILON || minor_radius.abs() < EPSILON {
                    return Err(Error::InvalidGeometry("torus has a zero radius"));
                }
                let p = mat_i * p_;
                /*
                         ^ Y
                         |
                    /---------\
                   /     |     \
                   |   -----   |
                   |   | O |- -|- - >Z
                   |   -----   |
                   \           /
                    \---------/

                    (X axis points into the screen)
                */
                let major_angle = p.y.atan2(p.z);

                // Rotate the point so that it's got Y = 0, so we can calculate
                // the minor angle
                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let new_mat_i = new_mat.try_inverse()
                    .ok_or(Error::SingularTransform("torus lowering transform"))?;
                let new_p = new_mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                let minor_angle = new_p.x.atan2(new_p.z);

                // Keep the boundary's wider periodic direction as the polar
                // coordinate, so full rings stay closed without an artificial
                // seam. Unroll the narrower direction radially.
                let (polar_angle, radial_angle, base_radius, radial_scale) =
                    if *polar_major {
                        (major_angle, minor_angle,
                         major_radius.abs(), minor_radius.abs())
                    } else {
                        (minor_angle, major_angle,
                         minor_radius.abs(), major_radius.abs())
                    };
                let radial_angle = Self::unwrap_from_start(
                    radial_angle, *radial_start);
                let radius = base_radius +
                    (radial_angle - *radial_start) * radial_scale;
                // Exchanging major/minor parameter roles reverses the chart
                // orientation. Reflect one coordinate to keep CCW outward.
                let sin = polar_angle.sin() * if *polar_major { 1.0 } else { -1.0 };
                Ok(DVec2::new(
                    radius * polar_angle.cos(),
                    radius * sin,
                ))
            },
            Surface::NURBS { surf, chart, pole } => {
                // A collapsed iso-boundary has no unique angular inverse.
                // Use the same declared equivalence that selected this chart,
                // rather than asking Newton to resolve its arbitrary angle.
                if let Some((point, uncertainty)) = pole {
                    let d = p - point;
                    if d.x.hypot(d.y).hypot(d.z) <= *uncertainty {
                        return Ok(DVec2::zeros());
                    }
                }
                Ok(chart.lower(Self::surf_lower(p, surf)?))
            },
            Surface::Sphere { mat_i, radius, .. } => {
                // mat_i is constructed in prepare to be a reasonable basis
                let p = (mat_i * p_).xyz() / *radius;
                let r = p.yz().norm();

                // Angle from 0 to PI
                let angle = r.atan2(p.x);
                let yz = p.yz();
                Ok(if yz.norm() < EPSILON {
                    yz
                } else {
                    yz * angle / yz.norm()
                })
            },
        }
    }

    fn angular_distance(a: DVec3, b: DVec3) -> f64 {
        a.cross(&b).norm().atan2(a.dot(&b))
    }

    fn point_minor_arc_distance(p: DVec3, a: DVec3, b: DVec3) -> Result<f64, Error> {
        let cross = a.cross(&b);
        let cross_norm = cross.norm();
        if cross_norm <= 32.0 * EPSILON {
            if a.dot(&b) < 0.0 {
                return Err(Error::InvalidGeometry("ambiguous antipodal spherical edge"));
            }
            return Ok(Self::angular_distance(p, a));
        }
        let n = cross / cross_norm;
        let projected = p - n * p.dot(&n);
        let projected_norm = projected.norm();
        let endpoint_distance = Self::angular_distance(p, a)
            .min(Self::angular_distance(p, b));
        if projected_norm <= 32.0 * EPSILON {
            return Ok(endpoint_distance);
        }
        let projected = projected / projected_norm;
        let arc_angle = cross_norm.atan2(a.dot(&b));
        let on_arc = |x: DVec3| {
            Self::angular_distance(a, x) <= arc_angle
                && Self::angular_distance(x, b) <= arc_angle
        };
        let mut distance = endpoint_distance;
        if on_arc(projected) {
            distance = distance.min(Self::angular_distance(p, projected));
        }
        if on_arc(-projected) {
            distance = distance.min(Self::angular_distance(p, -projected));
        }
        Ok(distance)
    }

    fn spherical_winding_sum(q: DVec3, points: &[DVec3],
                             edges: &[(usize, usize)]) -> Result<(f64, f64), Error> {
        let mut sum = 0.0;
        let mut magnitude = 0.0;
        for &(i, j) in edges {
            let (a, b) = (*points.get(i).ok_or(Error::InvalidGeometry("boundary edge index"))?,
                          *points.get(j).ok_or(Error::InvalidGeometry("boundary edge index"))?);
            let numerator = q.dot(&a.cross(&b));
            let denominator = 1.0 + q.dot(&a) + a.dot(&b) + b.dot(&q);
            let term = 2.0 * numerator.atan2(denominator);
            sum += term;
            magnitude += term.abs();
        }
        // A forward error allowance proportional to the work and accumulated
        // angle.  Candidates whose sign is not resolved are rejected.
        let error = 64.0 * EPSILON * (magnitude + edges.len() as f64 * PI);
        Ok((sum, error))
    }

    fn sphere_chart(points: &[DVec3], edges: &[(usize, usize)]) -> Result<DVec3, Error> {
        let mut candidates = Vec::with_capacity(edges.len());
        for (edge_index, &(i, j)) in edges.iter().enumerate() {
            let a = *points.get(i).ok_or(Error::InvalidGeometry("boundary edge index"))?;
            let b = *points.get(j).ok_or(Error::InvalidGeometry("boundary edge index"))?;
            let sum_norm = (a + b).norm();
            let cross_norm = a.cross(&b).norm();
            if sum_norm <= 32.0 * EPSILON {
                return Err(Error::InvalidGeometry("ambiguous antipodal spherical edge"));
            }
            if cross_norm > 32.0 * EPSILON {
                candidates.push((cross_norm.atan2(a.dot(&b)), edge_index));
            }
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        for (_, selected) in candidates {
            let (i, j) = edges[selected];
            let a = points[i];
            let b = points[j];
            let m = (a + b).normalize();
            let n = a.cross(&b).normalize();
            let mut clearance = Self::angular_distance(m, a)
                .min(Self::angular_distance(m, b));
            for (edge_index, &(u, v)) in edges.iter().enumerate() {
                if edge_index != selected {
                    clearance = clearance.min(Self::point_minor_arc_distance(
                        m, points[u], points[v])?);
                }
            }
            let clearance_error = 64.0 * EPSILON * (edges.len() as f64 + 1.0);
            if !clearance.is_finite() || clearance <= clearance_error {
                continue;
            }
            let delta = clearance * 0.5;
            let exterior_pole = delta.cos() * m - delta.sin() * n;
            let q = -exterior_pole.normalize();
            let (winding, winding_error) = Self::spherical_winding_sum(q, points, edges)?;
            if winding > winding_error {
                return Ok(q);
            }
        }
        Err(Error::CouldNotLower)
    }

    fn prepare(&mut self, verts: &[Vertex], boundary_edges: &[(usize, usize)],
               same_sense: bool) -> Result<(), Error> {
        if verts.is_empty() {
            return Err(Error::InvalidGeometry("surface has no vertices"));
        }
        match self {
            Surface::Cylinder { mat_i, z_min, z_max, .. } => {
                *z_min = std::f64::INFINITY;
                *z_max = -std::f64::INFINITY;
                for v in verts {
                    let p = (*mat_i) * DVec4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                    if p.z < *z_min {
                        *z_min = p.z;
                    }
                    if p.z > *z_max {
                        *z_max = p.z;
                    }
                }
            },
            Surface::Sphere { mat, mat_i, location, .. } => {
                let points: Vec<_> = verts.iter().map(|v| {
                    let offset = v.pos - *location;
                    if offset.norm_squared() == 0.0 {
                        Err(Error::InvalidGeometry("sphere boundary at center"))
                    } else {
                        Ok(offset.normalize())
                    }
                }).collect::<Result<_, _>>()?;
                let oriented_edges: Vec<_> = boundary_edges.iter().map(|&(a, b)| {
                    if same_sense { (a, b) } else { (b, a) }
                }).collect();
                let chart_center = Self::sphere_chart(&points, &oriented_edges)?;
                let axis = Self::fallback_perpendicular(chart_center);
                *mat = Self::make_rigid_transform(axis, chart_center, *location);
                *mat_i = mat
                    .try_inverse()
                    .ok_or(Error::SingularTransform("sphere transform"))?;
            },
            Surface::Torus { mat_i, major_radius,
                             polar_major, radial_start, .. } => {
                let mut major_angles = Vec::with_capacity(verts.len());
                let mut minor_angles = Vec::with_capacity(verts.len());
                for vertex in verts {
                    let (major, minor) = Self::torus_angles(
                        *mat_i, vertex.pos, *major_radius)?;
                    major_angles.push(major);
                    minor_angles.push(minor);
                }
                let (major_start, major_span) =
                    Self::smallest_circular_arc(&mut major_angles);
                let (minor_start, minor_span) =
                    Self::smallest_circular_arc(&mut minor_angles);
                *polar_major = major_span >= minor_span;
                *radial_start = if *polar_major {
                    minor_start
                } else {
                    major_start
                };
            },
            _ => (),
        }
        Ok(())
    }

    fn type_name(&self) -> &'static str {
        match self {
            Surface::Cylinder { .. } => "lower:Cylinder",
            Surface::Plane { .. } => "lower:Plane",
            Surface::Cone { .. } => "lower:Cone",
            Surface::NURBS { .. } => "lower:NURBS",
            Surface::Sphere { .. } => "lower:Sphere",
            Surface::Torus { .. } => "lower:Torus",
        }
    }

    pub fn lower_verts(&mut self, verts: &mut [Vertex], boundary_edges: &[(usize, usize)],
                       same_sense: bool)
        -> Result<Vec<(f64, f64)>, Error>
    {
        let name = self.type_name();
        crate::timing::time(name, || self.lower_verts_inner(verts, boundary_edges, same_sense))
    }

    fn lower_verts_inner(&mut self, verts: &mut [Vertex], boundary_edges: &[(usize, usize)],
                         same_sense: bool)
        -> Result<Vec<(f64, f64)>, Error>
    {
        self.prepare(verts, boundary_edges, same_sense)?;
        let mut pts = Vec::with_capacity(verts.len());
        for v in verts {
            // Project to the 2D subspace for triangulation
            let proj = self.lower(v.pos)?;
            // Update the surface normal
            v.norm = self.normal(v.pos, proj);
            pts.push((proj.x, proj.y));
        }
        Ok(pts)
    }

    fn periodic_uv_periods(&self) -> (Option<f64>, Option<f64>) {
        match self {
            Surface::NURBS { surf, .. } => {
                let u_period = if surf.surf.u_open {
                    None
                } else {
                    Some(surf.surf.max_u() - surf.surf.min_u())
                };
                let v_period = if surf.surf.v_open {
                    None
                } else {
                    Some((surf.surf.max_v() - surf.surf.min_v()) * surf.surf.aspect_ratio())
                };
                (u_period, v_period)
            },
            _ => (None, None),
        }
    }

    fn torus_angles(mat_i: DMat4, point: DVec3,
                    major_radius: f64) -> Result<(f64, f64), Error> {
        let p = (mat_i * DVec4::new(
            point.x, point.y, point.z, 1.0)).xyz();
        let major_angle = p.y.atan2(p.z);
        let radial = p.y.hypot(p.z);
        let minor_angle = p.x.atan2(radial - major_radius);
        if major_angle.is_finite() && minor_angle.is_finite() {
            Ok((major_angle, minor_angle))
        } else {
            Err(Error::InvalidGeometry("non-finite torus angle"))
        }
    }

    fn smallest_circular_arc(angles: &mut [f64]) -> (f64, f64) {
        if angles.is_empty() {
            return (0.0, 0.0);
        }
        let period = 2.0 * PI;
        for angle in angles.iter_mut() {
            *angle = angle.rem_euclid(period);
        }
        angles.sort_by(|a, b| a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal));

        let mut largest_gap = -1.0;
        let mut start = angles[0];
        for i in 0..angles.len() {
            let next = if i + 1 < angles.len() {
                angles[i + 1]
            } else {
                angles[0] + period
            };
            let gap = next - angles[i];
            if gap > largest_gap {
                largest_gap = gap;
                // Keep the same floating representative used for the data.
                // rem_euclid can round a tiny negative angle to `period`;
                // normalizing it again would move this endpoint to zero.
                start = angles[(i + 1) % angles.len()];
            }
        }
        (start, (period - largest_gap).max(0.0))
    }

    fn unwrap_from_start(angle: f64, start: f64) -> f64 {
        let period = 2.0 * PI;
        let angle = angle.rem_euclid(period);
        if angle + 1e-12 < start {
            angle + period
        } else {
            angle
        }
    }

    fn unwrap_near(value: f64, reference: f64, period: f64) -> f64 {
        if period.abs() <= EPSILON || !period.is_finite() {
            value
        } else {
            value + ((reference - value) / period).round() * period
        }
    }

    fn uv_coord(p: (f64, f64), coord: usize) -> f64 {
        if coord == 0 { p.0 } else { p.1 }
    }

    fn set_uv_coord(p: &mut (f64, f64), coord: usize, value: f64) {
        if coord == 0 {
            p.0 = value;
        } else {
            p.1 = value;
        }
    }

    fn unwrap_periodic_coord(pts: &mut [(f64, f64)],
                             edges: &[(usize, usize)],
                             start_edge: usize,
                             end_edge: usize,
                             coord: usize,
                             period: f64,
                             skip_large_closing_jump: bool) -> bool {
        if period.abs() <= EPSILON || !period.is_finite() {
            return false;
        }
        let n = end_edge - start_edge;
        if n < 2 {
            return false;
        }
        let vertices: Vec<_> = edges[start_edge..end_edge]
            .iter()
            .map(|edge| edge.0)
            .collect();
        if vertices.iter().any(|&idx| idx >= pts.len()) {
            return false;
        }

        let raw: Vec<_> = vertices.iter()
            .map(|&idx| Self::uv_coord(pts[idx], coord))
            .collect();
        let mut best = raw.clone();
        let mut best_max_jump = f64::INFINITY;
        let mut best_closing_jump = -1.0;
        let mut best_sum_jump = f64::INFINITY;

        // Try every vertex as the loop anchor.  Score only the traversed edges:
        // a contour which crosses a periodic seam must retain one full-period
        // jump on the closing edge to form a non-degenerate polygon in UV.
        for anchor in 0..n {
            let mut candidate = raw.clone();
            let mut prev = anchor;
            let mut max_jump: f64 = 0.0;
            let mut sum_jump = 0.0;
            for step in 1..n {
                let cur = (anchor + step) % n;
                candidate[cur] = Self::unwrap_near(raw[cur], candidate[prev], period);
                let d = (candidate[cur] - candidate[prev]).abs();
                max_jump = max_jump.max(d);
                sum_jump += d * d;
                prev = cur;
            }
            let closing_jump = (candidate[anchor] - candidate[prev]).abs();

            if max_jump < best_max_jump - 1e-9
                || ((max_jump - best_max_jump).abs() <= 1e-9
                    && (closing_jump > best_closing_jump + 1e-9
                        || ((closing_jump - best_closing_jump).abs() <= 1e-9
                            && sum_jump < best_sum_jump)))
            {
                best = candidate;
                best_max_jump = max_jump;
                best_closing_jump = closing_jump;
                best_sum_jump = sum_jump;
            }
        }

        if skip_large_closing_jump && best_closing_jump > period.abs() * 0.5 {
            // A single closed edge that winds all the way around the periodic
            // dimension cannot be represented as a closed contour in one
            // unwrapped plane without leaving one full-period constrained edge.
            // Leave these iso-periodic loops in their lowered coordinates;
            // otherwise the artificial cut can create CDT regressions.
            return false;
        }

        for (&idx, value) in vertices.iter().zip(best.into_iter()) {
            Self::set_uv_coord(&mut pts[idx], coord, value);
        }
        true
    }

    /// Unwrap periodic UV coordinates along each boundary loop.
    ///
    /// STEP files often describe periodic NURBS surfaces (for example,
    /// OpenCASCADE cylinders) using only 3D edge curves.  A point on the seam
    /// has two valid UV coordinates; lowering each 3D vertex independently can
    /// put adjacent seam vertices on opposite sides of the period, producing
    /// crossing or zero-length constrained edges.  Walk each selected contour
    /// in order and shift periodic coordinates by whole periods so neighboring
    /// vertices stay close in the triangulation domain.
    pub fn unwrap_periodic(&self,
                           pts: &mut [(f64, f64)],
                           edges: &[(usize, usize)],
                           ranges: &[(usize, usize, bool)]) {
        if ranges.is_empty() {
            return;
        }
        if matches!(self, Surface::NURBS { chart: SplineChart::Polar { .. }, .. }) {
            return;
        }
        let (u_period, v_period) = self.periodic_uv_periods();
        if u_period.is_none() && v_period.is_none() {
            return;
        }

        for &(start_edge, end_edge, single_edge_bound) in ranges {
            if start_edge >= end_edge || end_edge > edges.len() {
                continue;
            }
            for (coord, period) in [u_period, v_period].iter().enumerate() {
                if let Some(period) = period {
                    Self::unwrap_periodic_coord(
                        pts, edges, start_edge, end_edge, coord, *period, single_edge_bound);
                }
            }
        }
    }

    pub fn raise(&self, uv: DVec2) -> Option<DVec3> {
        match self {
            Surface::Sphere { mat, radius, .. } => {
                let angle = uv.norm();
                if angle > PI {
                    return None;
                }
                let x = angle.cos();

                // Calculate pre-transformed position
                let pos = (*radius) * if uv.norm() < EPSILON {
                    DVec3::new(x, 0.0, 0.0)
                } else {
                    let yz = uv.normalize() * angle.sin();
                    DVec3::new(x, yz.x, yz.y)
                };
                // Transform into world space
                let pos = (mat * DVec4::new(pos.x, pos.y, pos.z, 1.0))
                    .xyz();
                Some(pos)
            },
            Surface::NURBS { surf, chart, .. } =>
                Self::spline_raw(surf, chart, uv).map(|raw| surf.surf.point(raw)),
            Surface::Torus { mat, minor_radius, major_radius,
                             polar_major, radial_start, .. } => {
                if major_radius.abs() < EPSILON || minor_radius.abs() < EPSILON {
                    return None;
                }
                let sin = if *polar_major { uv.y } else { -uv.y };
                let polar_angle = sin.atan2(uv.x);
                let (major_angle, minor_angle) = if *polar_major {
                    (polar_angle,
                     *radial_start +
                         (uv.norm() - major_radius.abs()) / minor_radius.abs())
                } else {
                    (*radial_start +
                         (uv.norm() - minor_radius.abs()) / major_radius.abs(),
                     polar_angle)
                };
                let new_p = DVec3::new(minor_angle.sin(), 0.0, minor_angle.cos()) * *minor_radius;

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos());
                let new_mat = Self::make_rigid_transform(
                    z, DVec3::new(1.0, 0.0, 0.0), z * *major_radius);
                let p = new_mat * DVec4::new(new_p.x, new_p.y, new_p.z, 1.0);

                Some((mat * p).xyz())
            },
            _ => None,
        }
    }

    fn bbox(pts: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let (mut xmin, mut xmax) = (std::f64::INFINITY, -std::f64::INFINITY);
        let (mut ymin, mut ymax) = (std::f64::INFINITY, -std::f64::INFINITY);
        for (px, py) in pts {
            xmin = px.min(xmin);
            ymin = py.min(ymin);
            xmax = px.max(xmax);
            ymax = py.max(ymax);
        }
        (xmin, xmax, ymin, ymax)
    }

    fn spline_parameter(value: f64, min: f64, max: f64, open: bool) -> f64 {
        if open {
            value.clamp(min, max)
        } else {
            min + (value - min).rem_euclid(max - min)
        }
    }

    fn add_torus_steiner_points(&self, pts: &mut Vec<(f64, f64)>,
                                verts: &mut Vec<Vertex>, radial_scale: f64)
    {
        const ANGULAR_SAMPLES: usize = 32;

        let mut radii = Vec::with_capacity(pts.len());
        let mut angles = Vec::with_capacity(pts.len());
        for &(x, y) in pts.iter() {
            let radius = x.hypot(y);
            let angle = y.atan2(x);
            if radius.is_finite() && angle.is_finite() {
                radii.push(radius);
                angles.push(angle);
            }
        }
        if radii.is_empty() || angles.is_empty() {
            return;
        }

        let radial_min = radii.iter().copied()
            .fold(f64::INFINITY, f64::min);
        let radial_max = radii.iter().copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !radial_min.is_finite() || !radial_max.is_finite() ||
           radial_max - radial_min <= EPSILON
        {
            return;
        }

        let (angular_start, measured_span) =
            Self::smallest_circular_arc(&mut angles);
        let full_revolution = measured_span > 1.5 * PI;
        let angular_span = if full_revolution { 2.0 * PI } else { measured_span };
        if angular_span <= EPSILON {
            return;
        }

        // Radius in the polar chart encodes the other intrinsic angle.
        // Resolve both angles at the same rate, independent of length units
        // and which torus parameter is represented radially.
        let radial_angle = (radial_max - radial_min) / radial_scale;
        let radial_segments = (radial_angle * ANGULAR_SAMPLES as f64 / (2.0 * PI)).ceil() as usize;
        for radial_index in 1..radial_segments {
            let radial_fraction = radial_index as f64 / radial_segments as f64;
            let radius = radial_min * (1.0 - radial_fraction) +
                         radial_max * radial_fraction;
            for angular_index in 0..ANGULAR_SAMPLES {
                let angular_fraction = if full_revolution {
                    (angular_index as f64 + 0.5) / ANGULAR_SAMPLES as f64
                } else {
                    (angular_index as f64 + 1.0) / (ANGULAR_SAMPLES + 1) as f64
                };
                let angle = angular_start + angular_span * angular_fraction;
                let uv = DVec2::new(radius * angle.cos(), radius * angle.sin());
                if let Some(pos) = self.raise(uv) {
                    pts.push((uv.x, uv.y));
                    verts.push(Vertex {
                        pos,
                        norm: self.normal(pos, uv),
                        color: DVec3::new(0.0, 0.0, 0.0),
                    });
                }
            }
        }
    }

    fn add_spline_steiner_points(
        &self,
        pts: &mut Vec<(f64, f64)>,
        verts: &mut Vec<Vertex>,
        surf: &SampledSurface<4>,
        chart: &SplineChart,
    ) {
        const SAMPLES: usize = 16;

        let (xmin, xmax, ymin, ymax) = Self::bbox(pts);
        for x in 0..SAMPLES {
            let x_frac = (x as f64 + 1.0) / (SAMPLES as f64 + 1.0);
            let projected_u = x_frac * xmax + (1.0 - x_frac) * xmin;
            for y in 0..SAMPLES {
                let y_frac = (y as f64 + 1.0) / (SAMPLES as f64 + 1.0);
                let projected_v = y_frac * ymax + (1.0 - y_frac) * ymin;
                let projected = DVec2::new(projected_u, projected_v);
                let Some(raw_uv) = Self::spline_raw(surf, chart, projected) else { continue; };
                let pos = surf.surf.point(raw_uv);
                pts.push((projected_u, projected_v));
                verts.push(Vertex {
                    pos,
                    norm: Self::surf_normal(raw_uv, surf),
                    color: DVec3::new(0.0, 0.0, 0.0),
                });
            }
        }
    }

    pub fn add_steiner_points(&self, pts: &mut Vec<(f64, f64)>,
                                     verts: &mut Vec<Vertex>)
    {
        if let Surface::Torus { polar_major, major_radius, minor_radius, .. } = self {
            let radial_scale = if *polar_major { minor_radius.abs() } else { major_radius.abs() };
            self.add_torus_steiner_points(pts, verts, radial_scale);
            return;
        }

        match self {
            Surface::NURBS { surf, chart, .. } => {
                self.add_spline_steiner_points(pts, verts, surf, chart);
                return;
            },
            _ => (),
        }

        let (xmin, xmax, ymin, ymax) = Self::bbox(&pts);
        let num_pts = match self {
            Surface::Sphere { .. }   => 6,
            _ => 0,
        };

        for x in 0..num_pts {
            let x_frac = (x as f64 + 1.0) / (num_pts as f64 + 1.0);
            let u = x_frac * xmax + (1.0 - x_frac) * xmin;
            for y in 0..num_pts {
                let y_frac = (y as f64 + 1.0) / (num_pts as f64 + 1.0);
                let v = y_frac * ymax + (1.0 - y_frac) * ymin;

                let uv = DVec2::new(u, v);
                if let Some(pos) = self.raise(uv) {
                    pts.push((u, v));
                    verts.push(Vertex {
                        pos,
                        norm: self.normal(pos, uv),
                        color: DVec3::new(0.0, 0.0, 0.0),
                    });
                }
            }
        }
    }

    fn surf_normal(uv: DVec2, surf: &SampledSurface<4>) -> DVec3 {
        let derivs = surf.surf.derivs::<1>(uv);
        let n = derivs[1][0].cross(&derivs[0][1]);
        if n.norm_squared() > 1e-20 {
            return n.normalize();
        }

        // At a collapsed spline pole one derivative is zero exactly at the
        // boundary, but the surface still has a well-defined limiting normal.
        // Evaluate just inside each parameter boundary before giving up.
        let u_step = (surf.surf.max_u() - surf.surf.min_u()) * 1e-3;
        let v_step = (surf.surf.max_v() - surf.surf.min_v()) * 1e-3;
        for candidate in [
            DVec2::new(uv.x - u_step, uv.y),
            DVec2::new(uv.x + u_step, uv.y),
            DVec2::new(uv.x, uv.y - v_step),
            DVec2::new(uv.x, uv.y + v_step),
        ] {
            let candidate = DVec2::new(
                Self::spline_parameter(
                    candidate.x, surf.surf.min_u(), surf.surf.max_u(), surf.surf.u_open,
                ),
                Self::spline_parameter(
                    candidate.y, surf.surf.min_v(), surf.surf.max_v(), surf.surf.v_open,
                ),
            );
            let derivs = surf.surf.derivs::<1>(candidate);
            let n = derivs[1][0].cross(&derivs[0][1]);
            if n.norm_squared() > 1e-20 {
                return n.normalize();
            }
        }
        DVec3::zeros()
    }

    // Calculate the surface normal, using either the 3D or 2D position
    pub fn normal(&self, p: DVec3, uv: DVec2) -> DVec3 {
        match self {
            Surface::Plane { normal, .. } => *normal,
            Surface::Cone { mat, mat_i, angle, .. } => {
                // Project into CONE SPACE
                let pos = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);
                let xy = if pos.xy().norm() > std::f64::EPSILON {
                    pos.xy().normalize()
                } else {
                    return DVec3::zeros();
                };
                let normal = DVec4::new(xy.x * angle.cos(),
                                        xy.y * angle.cos(), -angle.sin(), 0.0);
                // Deproject back into world space
                (mat * normal).xyz()
            }
            Surface::Sphere { location, .. } => (p - location).normalize(),
            Surface::Cylinder { mat, mat_i, .. } => {
                // Project the point onto the axis
                let proj = mat_i * DVec4::new(p.x, p.y, p.z, 1.0);

                // Then the normal is just pointing along that direction
                // (same hack as below)
                let norm = DVec3::new(proj.x, proj.y, 0.0).normalize();
                (mat * norm.to_homogeneous()).xyz()
            },
            Surface::NURBS { surf, chart, .. } => Self::spline_raw(surf, chart, uv)
                .map(|raw| Self::surf_normal(raw, surf)).unwrap_or_else(DVec3::zeros),
            Surface::Torus { mat, mat_i, major_radius, .. } => {
                let p = (*mat_i * DVec4::new(p.x, p.y, p.z, 1.0)).xyz();
                let major_angle = p.y.atan2(p.z);

                let z = DVec3::new(0.0, major_angle.sin(), major_angle.cos()) * *major_radius;
                let norm = (p - z).normalize();

                (mat * norm.to_homogeneous()).xyz()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nurbs::{KnotVector, NURBSSurface};

    #[test]
    fn a_declared_pole_maps_to_one_chart_point_without_inverting_its_angle() {
        let knots = || KnotVector::from_multiplicities(2, &[0., 1.], &[3, 3]);
        let controls = vec![
            vec![DVec4::new(1e-16, 0., 1., 1.), DVec4::new(1., 0., 1., 1.), DVec4::new(1., 0., 0., 1.)],
            vec![DVec4::new(-1e-16, 1e-16, 1., 1.), DVec4::new(0., 1., 1., 1.), DVec4::new(0., 1., 0., 1.)],
            vec![DVec4::new(1e-16, 0., 1., 1.), DVec4::new(1., 0., 1., 1.), DVec4::new(1., 0., 0., 1.)],
        ];
        let surface = Surface::new_nurbs(SampledSurface::new(NURBSSurface::new(
            false, true, knots(), knots(), controls)), 1e-9);
        assert_eq!(surface.lower(DVec3::new(1e-11, 0., 1. + 1e-11)).unwrap(), DVec2::zeros());
        let away = surface.raise(DVec2::new(0.1, 0.1)).unwrap();
        assert!(surface.lower(away).unwrap().norm() > 0.1);
    }

    #[test]
    fn circular_arc_keeps_the_selected_endpoint_representative() {
        for input in [[-1e-16, 0., PI / 2., PI], [0.13, 0.7, 0.4, 0.2]] {
            let mut angles = input;
            let (start, span) = Surface::smallest_circular_arc(&mut angles);
            for angle in input {
                let offset = Surface::unwrap_from_start(angle, start) - start;
                assert!(offset >= -1e-14 && offset <= span + 1e-14,
                    "{} lies outside [{}, {}]", angle, start, start + span);
            }
        }
    }

    #[test]
    fn planar_projection_copies_coordinates_and_preserves_orientation() {
        for dropped in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut axis = DVec3::new(0.2, 0.3, 0.4);
                axis[dropped] = sign;
                let surface = Surface::new_plane(axis, DVec3::zeros(), DVec3::zeros()).unwrap();
                let a = DVec3::new(3.13, -0.45, 1.75);
                let mut b = a;
                let mut c = a;
                let x = (dropped + 1) % 3;
                let y = (dropped + 2) % 3;
                b[x] += 1.0;
                b[dropped] -= axis[x] / axis[dropped];
                c[y] += 1.0;
                c[dropped] -= axis[y] / axis[dropped];
                let pa = surface.lower(a).unwrap();
                assert_eq!(pa, DVec2::new(a[x], a[y] * sign));
                let pb = surface.lower(b).unwrap() - pa;
                let pc = surface.lower(c).unwrap() - pa;
                assert!((pb.x * pc.y - pb.y * pc.x) * (b - a).cross(&(c - a)).dot(&axis) > 0.);
            }
        }
        let tiny = Surface::new_plane(DVec3::new(0., 1e-100, 0.), DVec3::zeros(), DVec3::zeros()).unwrap();
        assert_eq!(tiny.normal(DVec3::zeros(), DVec2::zeros()), DVec3::new(0., 1., 0.));
    }

    fn latitude_loop(latitude: f64, segments: usize, reverse: bool,
                     vertices: &mut Vec<Vertex>, edges: &mut Vec<(usize, usize)>) {
        let start = vertices.len();
        for i in 0..segments {
            let angle = 2.0 * PI * i as f64 / segments as f64;
            vertices.push(Vertex {
                pos: DVec3::new(latitude.cos() * angle.cos(),
                                latitude.cos() * angle.sin(), latitude.sin()),
                norm: DVec3::zeros(),
                color: DVec3::zeros(),
            });
        }
        for i in 0..segments {
            let edge = (start + i, start + (i + 1) % segments);
            edges.push(if reverse { (edge.1, edge.0) } else { edge });
        }
    }

    fn lower_sphere(mut vertices: Vec<Vertex>, edges: &[(usize, usize)], same_sense: bool)
        -> (Surface, Vec<Vertex>, Vec<(f64, f64)>, DVec3)
    {
        let mut surface = Surface::new_sphere(DVec3::zeros(), 1.0).unwrap();
        let points = surface.lower_verts(&mut vertices, edges, same_sense).unwrap();
        let chart_center = match &surface {
            Surface::Sphere { mat, .. } => mat.column(0).xyz(),
            _ => unreachable!(),
        };
        for (vertex, &(u, v)) in vertices.iter().zip(&points) {
            assert!(u.is_finite() && v.is_finite());
            assert!((surface.raise(DVec2::new(u, v)).unwrap() - vertex.pos).norm() < 1e-12);
            assert!(vertex.norm.dot(&vertex.pos) > 1.0 - 1e-12);
        }
        (surface, vertices, points, chart_center)
    }

    #[test]
    fn spherical_point_to_minor_arc_distance_selects_interior_and_endpoint() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let interior = DVec3::new(1.0, 1.0, 0.2).normalize();
        let distance = Surface::point_minor_arc_distance(interior, a, b).unwrap();
        assert!((distance - 0.2_f64.atan2(2.0_f64.sqrt())).abs() < 1e-14);

        let beyond = DVec3::new(-0.01, 1.0, 0.001).normalize();
        let endpoint = Surface::angular_distance(beyond, b);
        assert!((Surface::point_minor_arc_distance(beyond, a, b).unwrap() - endpoint).abs()
            < 1e-14);
    }

    #[test]
    fn oriented_hemisphere_has_exterior_antipode_and_nonzero_mesh() {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        latitude_loop(0.0, 32, false, &mut vertices, &mut edges);
        let (_surface, _vertices, points, q) = lower_sphere(vertices, &edges, true);
        assert!((-q).z < 0.0, "chart antipode must be outside the north hemisphere");
        let area2: f64 = edges.iter().map(|&(i, j)|
            points[i].0 * points[j].1 - points[j].0 * points[i].1).sum();
        assert!(area2.abs() > 1.0);
        let mut triangulation = cdt::Triangulation::new_with_edges(&points, &edges).unwrap();
        triangulation.run().unwrap();
        assert!(triangulation.triangles().next().is_some());
    }

    #[test]
    fn spherical_chart_does_not_depend_on_length_units() {
        for radius in [1e-9, 1.0, 1e9] {
            let mut vertices = Vec::new();
            let mut edges = Vec::new();
            latitude_loop(0.0, 32, false, &mut vertices, &mut edges);
            for vertex in &mut vertices { vertex.pos *= radius; }
            let mut surface = Surface::new_sphere(DVec3::zeros(), radius).unwrap();
            let uv = surface.lower_verts(&mut vertices, &edges, true).unwrap();
            for (vertex, &(u, v)) in vertices.iter().zip(&uv) {
                let raised = surface.raise(DVec2::new(u, v)).unwrap();
                assert!((raised - vertex.pos).norm() / radius < 1e-12);
                assert!(vertex.norm.dot(&(vertex.pos / radius)) > 1.0 - 1e-12);
            }
        }
    }

    #[test]
    fn same_sense_reversal_selects_equivalent_spherical_chart() {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        latitude_loop(-0.35, 24, false, &mut vertices, &mut edges);
        let reversed: Vec<_> = edges.iter().map(|&(a, b)| (b, a)).collect();
        let (_, _, _, q1) = lower_sphere(vertices.clone(), &edges, true);
        let (_, _, _, q2) = lower_sphere(vertices, &reversed, false);
        assert!(q1.dot(&q2) > 1.0 - 1e-14);
        assert!((-q1).z < -0.35_f64.sin());
    }

    #[test]
    fn spherical_band_and_multiple_holes_put_antipode_in_known_exterior() {
        let mut band_vertices = Vec::new();
        let mut band_edges = Vec::new();
        latitude_loop(-0.4, 32, false, &mut band_vertices, &mut band_edges);
        latitude_loop(0.4, 32, true, &mut band_vertices, &mut band_edges);
        let (_, _, _, band_q) = lower_sphere(band_vertices, &band_edges, true);
        assert!((-band_q).z.abs() > 0.4_f64.sin());

        // A north cap with two clockwise holes.  The holes are represented by
        // small geodesic diamonds, making this also a non-convex trim region.
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        latitude_loop(-0.2, 40, false, &mut vertices, &mut edges);
        for center_x in [-0.45_f64, 0.45] {
            let start = vertices.len();
            for p in [
                DVec3::new(center_x - 0.12, 0.0, 0.8),
                DVec3::new(center_x, 0.12, 0.8),
                DVec3::new(center_x + 0.12, 0.0, 0.8),
                DVec3::new(center_x, -0.12, 0.8),
            ] { vertices.push(Vertex { pos: p.normalize(), norm: DVec3::zeros(), color: DVec3::zeros() }); }
            for i in 0..4 { edges.push((start + i, start + (i + 1) % 4)); }
            let points: Vec<_> = vertices.iter().map(|v| v.pos).collect();
            let (area, error) = Surface::spherical_winding_sum(
                DVec3::new(0.0, 0.0, 1.0), &points, &edges[edges.len() - 4..],
            ).unwrap();
            assert!(area < -error, "holes must have clockwise signed area");
        }
        let (_, _, _, q) = lower_sphere(vertices, &edges, true);
        let pole = -q;
        let in_hole = pole.z > 0.0 && [-0.45_f64, 0.45].iter().any(|center_x| {
            // Central projection back onto the diamonds' construction plane.
            (0.8 * pole.x / pole.z - center_x).abs()
                + (0.8 * pole.y / pole.z).abs() < 0.12
        });
        assert!(pole.z < -0.2_f64.sin() || in_hole);
    }

    #[test]
    fn periodic_unwrapping_preserves_nonuniform_boundary_parameters() {
        let surface = Surface::new_nurbs(SampledSurface::new(NURBSSurface::new(
            false, true,
            KnotVector::from_multiplicities(1, &[0., 1., 2., 3., 4.], &[2, 1, 1, 1, 2]),
            KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]),
            [(1., 0.), (0., 1.), (-1., 0.), (0., -1.), (1., 0.)].iter()
                .map(|&(x, y)| vec![DVec4::new(x, y, 0., 1.), DVec4::new(x, y, 1., 1.)])
                .collect(),
        )), 0.);
        let raw = vec![(0., 1.), (0.1, 1.), (0.6, 1.), (1., 1.),
                       (1., 0.), (0.6, 0.), (0.1, 0.), (0., 0.)];
        let (surf, chart) = match &surface {
            Surface::NURBS { surf, chart, .. } => (surf, chart),
            _ => unreachable!(),
        };
        let mut points: Vec<_> = raw.iter().map(|&(u, v)| {
            let p = chart.lower(DVec2::new(u, v));
            (p.x, p.y)
        }).collect();
        let original = points.clone();
        let edges: Vec<_> = (0..points.len()).map(|i| (i, (i + 1) % points.len())).collect();
        surface.unwrap_periodic(&mut points, &edges, &[(0, edges.len(), false)]);
        assert_eq!(points, original, "polar charts bypass seam unwrapping");
        for (&raw, &after) in raw.iter().zip(&points) {
            let a = surf.surf.point(DVec2::new(raw.0, raw.1));
            let b = surface.raise(DVec2::new(after.0, after.1)).unwrap();
            assert!((a - b).norm() < 1e-14, "unwrapping must not relocate boundary geometry");
        }
    }

    #[test]
    fn polar_charts_roundtrip_with_positive_orientation() {
        for periodic in [0, 1] {
            for scale in [-3.0, 3.0] {
                let chart = SplineChart::Polar {
                    periodic, periodic_min: 2.0, period: 5.0,
                    radial_origin: if scale > 0.0 { -1.0 } else { 2.0 },
                    radial_scale: scale, radial_min: -1.0, radial_max: 2.0,
                };
                let other = 1 - periodic;
                let mut raw = DVec2::zeros();
                raw[periodic] = 3.1;
                raw[other] = 0.4;
                let mapped = chart.lower(raw);
                let back = chart.raw(mapped).unwrap();
                assert!((back - raw).norm() < 1e-12);

                let mut seam = raw;
                seam[periodic] = 2.0;
                let first = chart.lower(seam);
                for turn in [-2.0, -1.0, 1.0, 2.0] {
                    seam[periodic] = 2.0 + turn * 5.0;
                    assert_eq!(chart.lower(seam), first,
                        "periodic seam copies must have identical chart coordinates");
                }

                let h = 1e-6;
                let mut du = raw;
                let mut dv = raw;
                du.x += h;
                dv.y += h;
                let a = (chart.lower(du) - mapped) / h;
                let b = (chart.lower(dv) - mapped) / h;
                assert!(a.x * b.y - a.y * b.x > 0.0);
            }
        }
    }

    #[test]
    fn full_periodic_spline_disks_and_bands_preserve_area() {
        for periodic in [0, 1] {
            for pole in [None, Some(false), Some(true)] {
                let w = 0.5_f64.sqrt();
                let circle = [(1., 0., 1.), (1., 1., w), (0., 1., 1.), (-1., 1., w),
                    (-1., 0., 1.), (-1., -1., w), (0., -1., 1.), (1., -1., w), (1., 0., 1.)];
                let mut controls: Vec<Vec<DVec4>> = circle.iter().map(|&(x, y, w)| {
                    [0., 1.].iter().map(|&v| {
                        let radius = match pole { None => 1., Some(false) => v, Some(true) => 1. - v };
                        let z = if pole.is_some() { radius } else { v };
                        DVec4::new(x * radius * w, y * radius * w, z * w, w)
                    }).collect()
                }).collect();
                let angular = KnotVector::from_multiplicities(2, &[0., 1., 2., 3., 4.], &[3, 2, 2, 2, 3]);
                let radial = KnotVector::from_multiplicities(1, &[0., 1.], &[2, 2]);
                let (u_knots, v_knots) = if periodic == 0 { (angular, radial) } else {
                    controls = (0..2).map(|v| controls.iter().map(|row| row[v]).collect()).collect();
                    (radial, angular)
                };
                let mut surface = Surface::new_nurbs(SampledSurface::new(NURBSSurface::new(
                    periodic != 0, periodic != 1, u_knots, v_knots, controls,
                )), 0.);
                assert!(matches!(surface, Surface::NURBS { chart: SplineChart::Polar { .. }, .. }));
                let mut vertices = Vec::new();
                let mut edges = Vec::new();
                let radii = if pole.is_some() { vec![1.] } else { vec![2., 1.] };
                for (ring, radius) in radii.into_iter().enumerate() {
                    let start = vertices.len();
                    for i in 0..64 {
                        let angle = (if ring == 0 { 1. } else { -1. }) * 2. * PI * i as f64 / 64.;
                        let uv = DVec2::new(radius * angle.cos(), radius * angle.sin());
                        vertices.push(Vertex { pos: surface.raise(uv).unwrap(),
                            norm: DVec3::zeros(), color: DVec3::zeros() });
                        edges.push((start + i, start + (i + 1) % 64));
                    }
                }
                if pole.is_some() {
                    vertices.push(Vertex { pos: surface.raise(DVec2::zeros()).unwrap(),
                        norm: DVec3::zeros(), color: DVec3::zeros() });
                }
                let mut points = surface.lower_verts(&mut vertices, &edges, true).unwrap();
                surface.add_steiner_points(&mut points, &mut vertices);
                let mut t = cdt::Triangulation::new_with_edges(&points, &edges).unwrap();
                t.run().unwrap();
                let mut area = 0.;
                for (a, b, c) in t.triangles() {
                    let cross = (vertices[b].pos - vertices[a].pos)
                        .cross(&(vertices[c].pos - vertices[a].pos));
                    assert!(cross.norm() > 0.0);
                    area += 0.5 * cross.norm();
                }
                let expected = if pole.is_some() { PI * 2.0_f64.sqrt() } else { 2. * PI };
                assert!((area / expected - 1.).abs() < 0.01, "area {} != {}", area, expected);
            }
        }
    }

    #[test]
    fn periodic_seam_bound_retains_a_full_width_uv_polygon() {
        let period = 2.0 * PI;
        let mut points = vec![
            (0.0, 1.0),
            (period, 0.5),
            (0.0, 0.0),
            (period * 0.25, 0.0),
            (period * 0.5, 0.0),
            (period * 0.75, 0.0),
            (0.0, 0.0),
            (period, 0.5),
        ];
        let edges = (0..points.len())
            .map(|i| (i, (i + 1) % points.len()))
            .collect::<Vec<_>>();

        assert!(Surface::unwrap_periodic_coord(
            &mut points, &edges, 0, edges.len(), 0, period, false,
        ));

        let u_min = points.iter().map(|p| p.0)
            .fold(f64::INFINITY, f64::min);
        let u_max = points.iter().map(|p| p.0)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((u_max - u_min - period).abs() < 1e-9);

        let mut triangulation =
            cdt::Triangulation::new_with_edges(&points, &edges).unwrap();
        triangulation.run().unwrap();
        assert!(triangulation.triangles().next().is_some());
    }

    #[test]
    fn bspline_surface_gets_interior_steiner_points() {
        let knots = || KnotVector::from_multiplicities(2, &[0.0, 1.0], &[3, 3]);
        let control_points = (0..3).map(|u| {
            (0..3).map(|v| {
                DVec4::new(u as f64, v as f64, (u * v) as f64 * 0.25, 1.0)
            }).collect()
        }).collect();
        let surface = Surface::new_nurbs(SampledSurface::new(NURBSSurface::new(
            true, true, knots(), knots(), control_points,
        )), 0.);
        let mut points = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut vertices = Vec::new();

        surface.add_steiner_points(&mut points, &mut vertices);

        assert_eq!(points.len(), 4 + 16 * 16);
        assert_eq!(vertices.len(), 16 * 16);
        assert!(vertices.iter().all(|vertex| vertex.norm.norm() > 0.99));
    }

    #[test]
    fn torus_sampling_resolves_both_intrinsic_angles() {
        for scale in [1e-9, 1., 1e9] {
            for polar_major in [true, false] {
                let mut surface = Surface::new_torus_with_ref_direction(
                    DVec3::zeros(), DVec3::z(), DVec3::x(), 4. * scale, scale).unwrap();
                let Surface::Torus { polar_major: polar, radial_start, .. } = &mut surface else { unreachable!() };
                *polar = polar_major;
                *radial_start = 0.;
                let (base, radial_scale) = if polar_major { (4. * scale, scale) } else { (scale, 4. * scale) };
                let mut points = Vec::new();
                for radius in [base, base + PI * radial_scale] {
                    for i in 0..32 {
                        let angle = 2. * PI * i as f64 / 32.;
                        points.push((radius * angle.cos(), radius * angle.sin()));
                    }
                }
                surface.add_steiner_points(&mut points, &mut Vec::new());
                let mut angles: Vec<_> = points.iter().map(|&(x, y)| (x.hypot(y) - base) / radial_scale).collect();
                angles.sort_by(f64::total_cmp);
                let gap = angles.windows(2).map(|v| v[1] - v[0]).fold(0.0_f64, f64::max);
                assert!(gap <= 2. * PI / 32. + 1e-12, "unresolved radial angle {}", gap);
            }
        }
    }

    #[test]
    fn planar_bilinear_splines_use_world_coordinate_predicates() {
        let surface = Surface::new_nurbs(SampledSurface::new(NURBSSurface::new(
            true, true,
            KnotVector::from_multiplicities(1, &[-1., 7.5], &[2, 2]),
            KnotVector::from_multiplicities(1, &[-6.17, 1.], &[2, 2]),
            vec![vec![DVec4::new(18., 4.665, -3.75, 1.), DVec4::new(18., 4.665, 3.42, 1.)],
                 vec![DVec4::new(18., -3.835, -3.75, 1.), DVec4::new(18., -3.835, 3.42, 1.)]],
        )), 0.);
        assert!(matches!(surface, Surface::Plane { .. }));
        let mut vertices: Vec<_> = [(3.665, -3.75), (4.665, -3.75), (4.665, 3.42), (3.665, 3.42)]
            .iter().map(|&(y, z)| Vertex { pos: DVec3::new(18., y, z), norm: DVec3::zeros(), color: DVec3::zeros() }).collect();
        let mut points: Vec<_> = vertices.iter().map(|v| {
            let uv = surface.lower(v.pos).unwrap(); (uv.x, uv.y)
        }).collect();
        surface.add_steiner_points(&mut points, &mut vertices);
        assert_eq!(vertices.len(), 4, "flat patches require no curvature samples");
        let t = cdt::Triangulation::build_with_edges(&points, &[(0, 1), (1, 2), (2, 3), (3, 0)]).unwrap();
        let area: f64 = t.triangles().map(|(a, b, c)| {
            let area = (vertices[b].pos - vertices[a].pos).cross(&(vertices[c].pos - vertices[a].pos)).norm() * 0.5;
            assert!(area > 0.); area
        }).sum();
        assert!((area - 7.17).abs() < 1e-12);
    }

    #[test]
    fn bspline_pole_uses_the_limiting_surface_normal() {
        let knots = || KnotVector::from_multiplicities(2, &[0.0, 1.0], &[3, 3]);
        let control_points = (0..3).map(|u| {
            let x = u as f64;
            vec![
                DVec4::new(x, 0.0, 0.0, 1.0),
                DVec4::new(x, 0.0, 1.0, 1.0),
                DVec4::new(1.0, 0.0, 1.0, 1.0),
            ]
        }).collect();
        let sampled = SampledSurface::new(NURBSSurface::new(
            true, true, knots(), knots(), control_points,
        ));

        let normal = Surface::surf_normal(DVec2::new(0.5, 1.1), &sampled);

        assert!(normal.norm() > 0.99);
        assert!(normal.y.abs() > 0.99);
    }

    fn torus_point(major_angle: f64, minor_angle: f64) -> DVec3 {
        let major_radius = 4.9;
        let minor_radius = 0.1;
        let ring_radius = major_radius + minor_radius * minor_angle.cos();
        DVec3::new(
            minor_radius * minor_angle.sin(),
            ring_radius * major_angle.sin(),
            ring_radius * major_angle.cos(),
        )
    }

    fn append_ring(vertices: &mut Vec<Vertex>, edges: &mut Vec<(usize, usize)>,
                   fixed_angle: f64, vary_major: bool, reverse: bool) {
        const SEGMENTS: usize = 32;
        let start = vertices.len();
        for i in 0..SEGMENTS {
            let direction = if reverse { -1.0 } else { 1.0 };
            let varying_angle = direction * 2.0 * PI * i as f64 / SEGMENTS as f64;
            let (major_angle, minor_angle) = if vary_major {
                (varying_angle, fixed_angle)
            } else {
                (fixed_angle, varying_angle)
            };
            vertices.push(Vertex {
                pos: torus_point(major_angle, minor_angle),
                norm: DVec3::zeros(),
                color: DVec3::zeros(),
            });
            edges.push((start + i, start + (i + 1) % SEGMENTS));
        }
    }

    fn assert_band_tessellates(mut surface: Surface, mut vertices: Vec<Vertex>,
                               edges: Vec<(usize, usize)>) {
        let mut points = surface.lower_verts(&mut vertices, &edges, true).unwrap();
        for (vertex, &(u, v)) in vertices.iter().zip(&points) {
            let raised = surface.raise(DVec2::new(u, v)).unwrap();
            assert!((raised - vertex.pos).norm() < 1e-9);
            let du = surface.raise(DVec2::new(u + 1e-6, v)).unwrap() - raised;
            let dv = surface.raise(DVec2::new(u, v + 1e-6)).unwrap() - raised;
            assert!(du.cross(&dv).dot(&vertex.norm) > 0.0,
                "both toroidal charts must preserve outward surface orientation");
        }
        let boundary_len = points.len();
        let radial_min = points.iter().map(|(u, v)| u.hypot(*v))
            .fold(f64::INFINITY, f64::min);
        let radial_max = points.iter().map(|(u, v)| u.hypot(*v))
            .fold(f64::NEG_INFINITY, f64::max);
        surface.add_steiner_points(&mut points, &mut vertices);
        assert!(points.len() > boundary_len);
        assert_eq!(points.len(), vertices.len());
        assert!(points[boundary_len..].iter().all(|(u, v)| {
            let radius = u.hypot(*v);
            radius > radial_min && radius < radial_max
        }));
        let mut triangulation =
            cdt::Triangulation::new_with_edges(&points, &edges).unwrap();
        triangulation.run().unwrap();
        assert!(triangulation.triangles().next().is_some());
    }

    fn test_torus() -> Surface {
        Surface::new_torus_with_ref_direction(
            DVec3::zeros(),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            4.9,
            0.1,
        ).unwrap()
    }

    #[test]
    fn full_major_toroidal_band_has_non_crossing_annular_contours() {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        append_ring(&mut vertices, &mut edges, 1.2, true, false);
        append_ring(&mut vertices, &mut edges, 0.2, true, true);
        assert_band_tessellates(test_torus(), vertices, edges);
    }

    #[test]
    fn full_minor_toroidal_band_uses_the_other_annular_chart() {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        append_ring(&mut vertices, &mut edges, 1.2, false, false);
        append_ring(&mut vertices, &mut edges, 0.2, false, true);
        assert_band_tessellates(test_torus(), vertices, edges);
    }
}
