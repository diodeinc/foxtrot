use step::step_file::StepFile;
use step::ap214::*;
use triangulate::triangulate::triangulate;
use triangulate::surface::Surface;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let surface_ids: Vec<usize> = args[2..].iter().map(|s| s.parse().unwrap()).collect();

    let data = std::fs::read(path).expect("Could not open file");
    let flat = StepFile::strip_flatten(&data);
    let step = StepFile::parse(&flat);

    for sid in &surface_ids {
        eprintln!("\n=== Surface #{} ===", sid);
        match &step.0[*sid] {
            Entity::BSplineSurfaceWithKnots(b) => {
                eprintln!("  Type: BSplineSurfaceWithKnots");
                eprintln!("  u_degree: {}, v_degree: {}", b.u_degree, b.v_degree);
                eprintln!("  u_mults: {:?}, v_mults: {:?}", b.u_multiplicities, b.v_multiplicities);
                eprintln!("  control_points: {}x{}",
                    b.control_points_list.len(),
                    b.control_points_list.first().map(|r| r.len()).unwrap_or(0));
            },
            Entity::CylindricalSurface(_) => eprintln!("  Type: CylindricalSurface"),
            Entity::Plane(_) => eprintln!("  Type: Plane"),
            Entity::ConicalSurface(_) => eprintln!("  Type: ConicalSurface"),
            Entity::SphericalSurface(_) => eprintln!("  Type: SphericalSurface"),
            Entity::ToroidalSurface(_) => eprintln!("  Type: ToroidalSurface"),
            e => eprintln!("  Type: {:?}", std::mem::discriminant(e)),
        }

        // Find the face that references this surface
        for (i, entity) in step.0.iter().enumerate() {
            if let Entity::AdvancedFace(face) = entity {
                if face.face_geometry.0 == *sid {
                    eprintln!("  Face #{}, same_sense: {}, bounds: {}", i, face.same_sense, face.bounds.len());
                    for (bi, b) in face.bounds.iter().enumerate() {
                        let (bound_id, orientation) = match &step.0[b.0] {
                            Entity::FaceBound(b) => (b.bound, b.orientation),
                            Entity::FaceOuterBound(b) => (b.bound, b.orientation),
                            _ => continue,
                        };
                        if let Entity::EdgeLoop(el) = &step.0[bound_id.0] {
                            eprintln!("  Bound[{}]: {} edges, orientation={}", bi, el.edge_list.len(), orientation);
                            for (ei, oe_id) in el.edge_list.iter().enumerate() {
                                if let Entity::OrientedEdge(oe) = &step.0[oe_id.0] {
                                    if let Entity::EdgeCurve(ec) = &step.0[oe.edge_element.0] {
                                        let curve_type = match &step.0[ec.edge_geometry.0] {
                                            Entity::Line(_) => "Line",
                                            Entity::Circle(_) => "Circle",
                                            Entity::BSplineCurveWithKnots(_) => "BSpline",
                                            Entity::ComplexEntity(_) => "Complex",
                                            Entity::SurfaceCurve(_) => "SurfaceCurve",
                                            Entity::SeamCurve(_) => "SeamCurve",
                                            _ => "Other",
                                        };
                                        let s = get_vertex_pos(&step, ec.edge_start);
                                        let e = get_vertex_pos(&step, ec.edge_end);
                                        if let (Some(s), Some(e)) = (s, e) {
                                            eprintln!("    Edge[{}]: {} ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4}) same_sense={} orient={}",
                                                ei, curve_type, s.0, s.1, s.2, e.0, e.1, e.2,
                                                ec.same_sense, oe.orientation);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

fn get_vertex_pos(step: &StepFile, v: Vertex) -> Option<(f64, f64, f64)> {
    if let Entity::VertexPoint(vp) = &step.0[v.0] {
        if let Entity::CartesianPoint(cp) = &step.0[vp.vertex_geometry.0] {
            return Some((cp.coordinates[0].0, cp.coordinates[1].0, cp.coordinates[2].0));
        }
    }
    None
}
