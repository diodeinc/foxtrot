use glm::DVec3;
use nalgebra_glm as glm;
use nurbs::KnotVector;
use step::ap214::*;
use step::step_file::StepFile;
use triangulate::surface::Surface;
use triangulate::triangulate::*;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let target_surf: usize = args[2].parse().unwrap();

    let data = std::fs::read(path).expect("Could not open file");
    let flat = StepFile::strip_flatten(&data);
    let step = StepFile::parse(&flat);

    // Find the face that references this surface
    for (i, entity) in step.0.iter().enumerate() {
        if let Entity::AdvancedFace(face) = entity {
            if face.face_geometry.0 == target_surf {
                eprintln!("Found face #{} for surface #{}", i, target_surf);
                eprintln!(
                    "same_sense: {}, bounds: {}",
                    face.same_sense,
                    face.bounds.len()
                );

                // Build the surface
                let surf = get_surface_for_debug(&step, face.face_geometry);
                if surf.is_none() {
                    eprintln!("Could not build surface");
                    return;
                }
                let mut surf = surf.unwrap();

                // Collect contour points
                let mut verts: Vec<triangulate::mesh::Vertex> = Vec::new();
                let mut edges: Vec<(usize, usize)> = Vec::new();
                let mut num_pts = 0usize;

                for b in &face.bounds {
                    let bound_contours = face_bound_3d(&step, *b);
                    if bound_contours.is_empty() {
                        continue;
                    }
                    if bound_contours.len() == 1 {
                        verts.push(triangulate::mesh::Vertex {
                            pos: bound_contours[0],
                            norm: DVec3::zeros(),
                            color: DVec3::zeros(),
                        });
                        num_pts += 1;
                    } else {
                        let start = num_pts;
                        for pt in bound_contours {
                            edges.push((num_pts, num_pts + 1));
                            verts.push(triangulate::mesh::Vertex {
                                pos: pt,
                                norm: DVec3::zeros(),
                                color: DVec3::zeros(),
                            });
                            num_pts += 1;
                        }
                        num_pts -= 1;
                        verts.pop();
                        edges.pop();
                        edges.last_mut().unwrap().1 = start;
                    }
                }

                // Lower to 2D
                let pts = surf.lower_verts(&mut verts).expect("Could not lower");

                eprintln!("\n=== 2D Points ({}) ===", pts.len());
                for (i, (x, y)) in pts.iter().enumerate() {
                    eprintln!("  pt[{}] = ({:.10}, {:.10})", i, x, y);
                }
                eprintln!("\n=== Edges ({}) ===", edges.len());
                for (i, (a, b)) in edges.iter().enumerate() {
                    eprintln!("  edge[{}] = ({}, {})", i, a, b);
                }

                // Try CDT
                eprintln!("\n=== CDT Attempt ===");
                match cdt::Triangulation::build_with_edges(&pts, &edges) {
                    Ok(t) => {
                        let tris: Vec<_> = t.triangles().collect();
                        eprintln!("Success: {} triangles", tris.len());
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }

                return;
            }
        }
    }
    eprintln!("Face for surface {} not found", target_surf);
}

fn get_surface_for_debug(s: &StepFile, surf: step::ap214::Surface) -> Option<Surface> {
    match &s[surf] {
        Entity::Plane(p) => {
            let (location, axis, ref_direction) = axis2_placement_3d_debug(s, p.position);
            Some(Surface::new_plane(axis, ref_direction, location))
        }
        Entity::CylindricalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d_debug(s, c.position);
            Some(Surface::new_cylinder(
                axis,
                ref_direction,
                location,
                c.radius.0 .0 .0,
            ))
        }
        Entity::BSplineSurfaceWithKnots(b) => {
            let u_knots: Vec<f64> = b.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = b
                .u_multiplicities
                .iter()
                .map(|&k| k.try_into().unwrap())
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                b.u_degree.try_into().unwrap(),
                &u_knots,
                &u_multiplicities,
            );
            let v_knots: Vec<f64> = b.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = b
                .v_multiplicities
                .iter()
                .map(|&k| k.try_into().unwrap())
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                b.v_degree.try_into().unwrap(),
                &v_knots,
                &v_multiplicities,
            );
            let control_points: Vec<Vec<DVec3>> = b
                .control_points_list
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|cp| {
                            if let Entity::CartesianPoint(p) = &s.0[cp.0] {
                                DVec3::new(
                                    p.coordinates[0].0,
                                    p.coordinates[1].0,
                                    p.coordinates[2].0,
                                )
                            } else {
                                panic!("Expected CartesianPoint")
                            }
                        })
                        .collect()
                })
                .collect();
            let bspline = nurbs::BSplineSurface::new(u_knot_vec, v_knot_vec, control_points);
            let sampled = nurbs::SampledSurface::new(bspline);
            Some(Surface::BSpline(sampled))
        }
        _ => {
            eprintln!(
                "Surface type not handled: {:?}",
                std::mem::discriminant(&s[surf])
            );
            None
        }
    }
}

fn axis2_placement_3d_debug(
    s: &StepFile,
    a: step::ap214::Axis2Placement3d,
) -> (DVec3, DVec3, DVec3) {
    let placement = match &s.0[a.0] {
        Entity::Axis2Placement3d(p) => p,
        _ => panic!("Expected Axis2Placement3d"),
    };
    let location = match &s.0[placement.location.0] {
        Entity::CartesianPoint(p) => {
            DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0)
        }
        _ => panic!("Expected CartesianPoint"),
    };
    let axis = match &s.0[placement.axis.0] {
        Entity::Direction(d) => DVec3::new(
            d.direction_ratios[0].0,
            d.direction_ratios[1].0,
            d.direction_ratios[2].0,
        )
        .normalize(),
        _ => panic!("Expected Direction"),
    };
    let ref_direction = match &s.0[placement.ref_direction.0] {
        Entity::Direction(d) => DVec3::new(
            d.direction_ratios[0].0,
            d.direction_ratios[1].0,
            d.direction_ratios[2].0,
        )
        .normalize(),
        _ => panic!("Expected Direction"),
    };
    (location, axis, ref_direction)
}

fn face_bound_3d(s: &StepFile, b: step::ap214::FaceBoundId) -> Vec<DVec3> {
    let (bound_id, _orientation) = match &s.0[b.0] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        _ => return vec![],
    };
    let edge_list = match &s.0[bound_id.0] {
        Entity::EdgeLoop(el) => &el.edge_list,
        Entity::VertexLoop(vl) => {
            if let Entity::VertexPoint(vp) = &s.0[vl.loop_vertex.0] {
                if let Entity::CartesianPoint(cp) = &s.0[vp.vertex_geometry.0] {
                    return vec![DVec3::new(
                        cp.coordinates[0].0,
                        cp.coordinates[1].0,
                        cp.coordinates[2].0,
                    )];
                }
            }
            return vec![];
        }
        _ => return vec![],
    };

    let mut pts = Vec::new();
    for oe_id in edge_list {
        if let Entity::OrientedEdge(oe) = &s.0[oe_id.0] {
            if let Entity::EdgeCurve(ec) = &s.0[oe.edge_element.0] {
                let start = get_vertex_pos(s, ec.edge_start);
                if let Some(p) = start {
                    pts.push(p);
                }
            }
        }
    }
    // Close the loop by repeating the first point
    if !pts.is_empty() {
        pts.push(pts[0]);
    }
    pts
}

fn get_vertex_pos(s: &StepFile, v: step::ap214::Vertex) -> Option<DVec3> {
    if let Entity::VertexPoint(vp) = &s.0[v.0] {
        if let Entity::CartesianPoint(cp) = &s.0[vp.vertex_geometry.0] {
            return Some(DVec3::new(
                cp.coordinates[0].0,
                cp.coordinates[1].0,
                cp.coordinates[2].0,
            ));
        }
    }
    None
}
