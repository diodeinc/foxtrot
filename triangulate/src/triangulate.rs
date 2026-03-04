use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4, U32Vec3};
use log::{info, warn, error};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use step::{
    ap214, ap214::*, step_file::{FromEntity, StepFile}, id::Id, ap214::Entity,
};
use crate::{
    Error,
    curve::Curve,
    mesh, mesh::{Mesh, Triangle},
    stats::Stats,
    surface::Surface
};
use nurbs::{BSplineSurface, SampledCurve, SampledSurface, NURBSSurface, KnotVector};

/// Set the `SAVE_DEBUG_SVGS` environment variable to a directory path to save
/// SVG debug output for faces that error or panic during triangulation.
fn save_debug_svg_dir() -> Option<String> {
    std::env::var("SAVE_DEBUG_SVGS").ok()
}

/// `TransformStack` is a mapping of representations to transformed children.
type TransformStack<'a> =
    HashMap<Representation<'a>, Vec<(Representation<'a>, DMat4)>>;
fn build_transform_stack<'a>(s: &'a StepFile, flip: bool) -> TransformStack<'a> {
    // Store a map of parent -> (child, transform)
    let mut transform_stack: HashMap<_, Vec<_>> = HashMap::new();
    for r in s.0.iter()
        .filter_map(|e|
            RepresentationRelationshipWithTransformation_::try_from_entity(e))
    {
        let (a, b) = if flip {
            (r.rep_2, r.rep_1)
        } else {
            (r.rep_1, r.rep_2)
        };
        let mut mat = item_defined_transformation(s, r.transformation_operator.cast());
        if flip {
            mat = mat.try_inverse().expect("Could not invert transform matrix");
        }

        transform_stack.entry(b)
            .or_default()
            .push((a, mat));
    }
    transform_stack
}

fn transform_stack_roots<'a>(transform_stack: &TransformStack<'a>) -> Vec<Representation<'a>> {
    let children: HashSet<_> = transform_stack
        .values()
        .flat_map(|v| v.iter())
        .map(|v| v.0)
        .collect();
    transform_stack
        .keys()
        .filter(|k| !children.contains(k))
        .copied()
        .collect()
}

/// Convert an SiUnit with name Metre to a mm scale factor.
fn si_unit_to_mm(si: &SiUnit_) -> Option<f64> {
    if !matches!(si.name, SiUnitName::Metre) { return None; }
    Some(match &si.prefix {
        Some(SiPrefix::Milli) => 1.0,
        Some(SiPrefix::Centi) => 10.0,
        Some(SiPrefix::Micro) => 0.001,
        Some(SiPrefix::Nano) => 0.000_001,
        Some(SiPrefix::Kilo) => 1_000_000.0,
        None => 1000.0, // bare metres → mm
        _ => 1.0,
    })
}

/// Resolve a unit entity index to a mm scale factor.
/// Handles both direct SiUnit entities and ComplexEntity wrappers
/// (e.g. `(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))`).
fn resolve_length_unit_to_mm(s: &StepFile, idx: usize) -> Option<f64> {
    match &s.0[idx] {
        Entity::SiUnit(si) => si_unit_to_mm(si),
        Entity::ComplexEntity(subs) => {
            for sub in subs {
                if let Some(si) = SiUnit_::try_from_entity(sub) {
                    if let Some(scale) = si_unit_to_mm(si) {
                        return Some(scale);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Detect the length unit in a STEP file and return a scale factor to
/// convert coordinates to millimeters.  Returns 1.0 if the file already
/// uses mm or if the unit cannot be determined.
fn detect_length_scale_to_mm(s: &StepFile) -> f64 {
    // Helper: given a unit entity, return the mm scale if it's a length unit
    let unit_scale = |unit_entity: &Entity| -> Option<f64> {
        match unit_entity {
            Entity::SiUnit(si) => si_unit_to_mm(si),
            Entity::ConversionBasedUnit(cbu) => {
                let name = cbu.name.0.to_uppercase();
                if name.contains("INCH") {
                    return Some(25.4);
                } else if name.contains("FOOT") || name.contains("FT") {
                    return Some(304.8);
                }
                // Try to read the conversion factor and resolve its base unit
                let try_mwu = |value: &MeasureValue, unit_component: &Unit| -> Option<f64> {
                    if let MeasureValue::LengthMeasure(lm) = value {
                        let base_scale = resolve_length_unit_to_mm(s, unit_component.0)
                            .unwrap_or(1000.0); // fallback: assume metres
                        return Some(lm.0 * base_scale);
                    }
                    None
                };
                if let Entity::MeasureWithUnit(mwu) = &s.0[cbu.conversion_factor.0] {
                    if let Some(v) = try_mwu(&mwu.value_component, &mwu.unit_component) {
                        return Some(v);
                    }
                }
                if let Entity::LengthMeasureWithUnit(lmwu) = &s.0[cbu.conversion_factor.0] {
                    if let Some(v) = try_mwu(&lmwu.value_component, &lmwu.unit_component) {
                        return Some(v);
                    }
                }
                None
            },
            _ => None,
        }
    };

    for entity in s.0.iter() {
        // GlobalUnitAssignedContext may be standalone or inside a ComplexEntity
        let guacs: Vec<&GlobalUnitAssignedContext_> = match entity {
            Entity::GlobalUnitAssignedContext(g) => vec![g],
            Entity::ComplexEntity(subs) => subs.iter()
                .filter_map(|e| GlobalUnitAssignedContext_::try_from_entity(e))
                .collect(),
            _ => continue,
        };
        for guac in guacs {
            for unit_id in &guac.units {
                // The unit may be a direct entity or inside a ComplexEntity
                let check_entity = |e: &Entity| -> Option<f64> { unit_scale(e) };
                match &s.0[unit_id.0] {
                    Entity::ComplexEntity(subs) => {
                        for sub in subs {
                            if let Some(scale) = check_entity(sub) {
                                if (scale - 1.0).abs() > 1e-10 {
                                    info!("STEP length unit scale: {}", scale);
                                }
                                return scale;
                            }
                        }
                    },
                    e => {
                        if let Some(scale) = check_entity(e) {
                            if (scale - 1.0).abs() > 1e-10 {
                                info!("STEP length unit scale: {}", scale);
                            }
                            return scale;
                        }
                    },
                }
            }
        }
    }
    1.0 // default: assume mm
}

/// Fallback unit detection when the structured GUAC-based approach returns
/// the default (1.0).  Scans all entities for length-related SiUnits and
/// ConversionBasedUnits that may exist outside of a GUAC context (or whose
/// GUAC failed to parse).
fn detect_length_scale_fallback(s: &StepFile) -> f64 {
    let mut found_bare_metre = false;
    let mut found_milli_metre = false;
    let mut found_inch = false;

    for entity in s.0.iter() {
        let subs: &[Entity] = match entity {
            Entity::ComplexEntity(v) => v,
            _ => std::slice::from_ref(entity),
        };
        // Check if this entity group contains a LENGTH_UNIT marker
        let has_length_unit = subs.iter().any(|e| matches!(e, Entity::LengthUnit(_)));
        if !has_length_unit { continue; }

        for sub in subs {
            match sub {
                Entity::SiUnit(si) if matches!(si.name, SiUnitName::Metre) => {
                    match &si.prefix {
                        Some(SiPrefix::Milli) => found_milli_metre = true,
                        None => found_bare_metre = true,
                        _ => {},
                    }
                },
                Entity::ConversionBasedUnit(cbu) => {
                    if cbu.name.0.to_uppercase().contains("INCH") {
                        found_inch = true;
                    }
                },
                _ => {},
            }
        }
    }

    // Also check LengthMeasureWithUnit for known conversion factors.
    // CONVERSION_BASED_UNIT('INCH', ...) often fails to parse, but the
    // corresponding LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.0254), ...)
    // may still be present.
    if !found_inch {
        for entity in s.0.iter() {
            if let Entity::LengthMeasureWithUnit(lmwu) = entity {
                if let MeasureValue::LengthMeasure(lm) = &lmwu.value_component {
                    // 0.0254 m = 1 inch
                    if (lm.0 - 0.0254).abs() < 1e-6 {
                        found_inch = true;
                        break;
                    }
                }
            }
        }
    }

    if found_inch {
        info!("STEP fallback unit detection: INCH → scale 25.4");
        25.4
    } else if found_bare_metre && !found_milli_metre {
        info!("STEP fallback unit detection: bare METRE → scale 1000");
        1000.0
    } else {
        1.0
    }
}

pub fn triangulate(s: &StepFile) -> (Mesh, Stats) {
    let styled_items: Vec<_> = s.0.iter()
        .filter_map(|e| MechanicalDesignGeometricPresentationRepresentation_::try_from_entity(e))
        .flat_map(|m| m.items.iter())
        .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
        .collect();
    let styled_item_colors: HashMap<usize, DVec3> = styled_items.iter()
        .filter_map(|styled|
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0])
                    .map(|c| (styled.item.0, c))
            })
        .collect();

    // Store a map of parent -> (child, transform)
    let mut transform_stack = build_transform_stack(s, false);
    let mut roots = transform_stack_roots(&transform_stack);
    // The transformation graph isn't directional (because STEP is a Good File
    // Format), so if it's got more than one root, assume it's backwards.  We
    // are assuming that directions in the graph are consistent within the file,
    // until we find a counterexample.
    if roots.len() > 1 {
        info!("Flipping transform stack");
        transform_stack = build_transform_stack(s, true);
        roots = transform_stack_roots(&transform_stack);
    }
    let mut todo: Vec<_> = roots.into_iter()
        .map(|v| (v, DMat4::identity()))
        .collect();
    if todo.len() > 1 {
        warn!("Transformation stack has more than one root!");
    }

    // Store a map of ShapeRepresentationRelationships, which some models
    // use to map from axes to specific instances
    let mut shape_rep_relationship: HashMap<Id<_>, Vec<Id<_>>> = HashMap::new();
    for (r1, r2) in s.0.iter()
        .filter_map(|e| ShapeRepresentationRelationship_::try_from_entity(e))
        .map(|e| (e.rep_1, e.rep_2))
    {
        shape_rep_relationship.entry(r1).or_default().push(r2);
    }

    let mut to_mesh: HashMap<Id<_>, Vec<_>> = HashMap::new();
    while let Some((id, mat)) = todo.pop() {
        for child in shape_rep_relationship.get(&id).unwrap_or(&vec![]) {
            todo.push((*child, mat));
        }
        if let Some(children) = transform_stack.get(&id) {
            for (child, next_mat) in children {
                todo.push((*child, mat * next_mat));
            }
        } else {
            // Bind this transform to the RepresentationItem, which is
            // either a ManifoldSolidBrep or a ShellBasedSurfaceModel
            let items = match &s[id] {
                Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
                Entity::ShapeRepresentation(b) => &b.items,
                Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
                e => panic!("Could not get shape from {:?}", e),
            };

            for m in items.iter() {
                match &s[*m] {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) =>
                        to_mesh.entry(*m).or_default().push(mat),
                    Entity::Axis2Placement3d(_) => (),
                    e => warn!("Skipping {:?}", e),
                }
            }
        }
    }
    // If there are items in breps that aren't attached to a transformation
    // chain, then draw them individually (with an identity matrix)
    if to_mesh.is_empty() {
        s.0.iter()
            .enumerate()
            .filter(|(_i, e)|
                match e {
                    Entity::ManifoldSolidBrep(_)
                    | Entity::BrepWithVoids(_)
                    | Entity::ShellBasedSurfaceModel(_) => true,
                    _ => false,
                }
            )
            .map(|(i, _e)| Id::new(i))
            .for_each(|i| to_mesh.entry(i).or_default().push(DMat4::identity()));
    }

    let (to_mesh_iter, empty) = {
        #[cfg(feature = "rayon")]
        { (to_mesh.par_iter(), || (Mesh::default(), Stats::default())) }
        #[cfg(not(feature = "rayon"))]
        { (to_mesh.iter(), (Mesh::default(), Stats::default())) }
    };
    let mesh_fold = to_mesh_iter
        .fold(
            // Empty constructor
            empty,

            // Fold operation
            |(mut mesh, mut stats), (id, mats)| {
                info!("processing shape entity {} ({} transforms)", id.0,
                      mats.len());
                let v_start = mesh.verts.len();
                let t_start = mesh.triangles.len();
                let default_color = styled_item_colors.get(&id.0)
                    .copied()
                    .unwrap_or(DVec3::new(0.5, 0.5, 0.5));
                match &s[*id] {
                    Entity::ManifoldSolidBrep(b) =>
                        closed_shell(s, b.outer, &mut mesh, &mut stats,
                            &styled_item_colors, default_color),
                    Entity::ShellBasedSurfaceModel(b) =>
                        for v in &b.sbsm_boundary {
                            shell(s, *v, &mut mesh, &mut stats,
                                &styled_item_colors, default_color);
                        },
                    Entity::BrepWithVoids(b) =>
                        // TODO: handle voids
                        closed_shell(s, b.outer, &mut mesh, &mut stats,
                            &styled_item_colors, default_color),
                    _ => {
                        warn!("Skipping {:?} (not a known solid)", s[*id]);
                        return (mesh, stats);
                    },
                };

                // Build copies of the mesh by copying and applying transforms
                let v_end = mesh.verts.len();
                let t_end = mesh.triangles.len();
                for mat in &mats[1..] {
                    for v in v_start..v_end {
                        let p = mesh.verts[v].pos;
                        let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                        let pos = (mat * p_h).xyz();

                        let n = mesh.verts[v].norm;
                        let norm = (mat * glm::vec3_to_vec4(&n)).xyz();
                        let color = mesh.verts[v].color;

                        mesh.verts.push(mesh::Vertex { pos, norm, color });
                    }
                    let offset = mesh.verts.len() - v_end;
                    for t in t_start..t_end {
                        let mut tri = mesh.triangles[t];
                        tri.verts.add_scalar_mut(offset as u32);
                        mesh.triangles.push(tri);
                    }
                }

                // Now that we've built all of the other copies of the mesh,
                // re-use the original mesh and apply the first transform
                let mat = mats[0];
                for v in v_start..v_end {
                    let p = mesh.verts[v].pos;
                    let p_h = DVec4::new(p.x, p.y, p.z, 1.0);
                    mesh.verts[v].pos = (mat * p_h).xyz();

                    let n = mesh.verts[v].norm;
                    mesh.verts[v].norm = (mat * glm::vec3_to_vec4(&n)).xyz();
                }
                (mesh, stats)
            });

    let (mesh, stats) = {
        #[cfg(feature = "rayon")]
        { mesh_fold.reduce(empty,
                |a, b| (Mesh::combine(a.0, b.0), Stats::combine(a.1, b.1))) }
        #[cfg(not(feature = "rayon"))]
        {
            mesh_fold
        }
    };

    // Scale coordinates to millimeters based on the STEP file's length unit
    info!("all faces done, detecting length scale...");
    let mut scale = detect_length_scale_to_mm(s);
    if (scale - 1.0).abs() < 1e-10 {
        scale = detect_length_scale_fallback(s);
    }
    info!("length scale: {}", scale);
    let mut mesh = mesh;
    if (scale - 1.0).abs() > 1e-10 {
        info!("Applying unit scale factor: {}", scale);
        for v in &mut mesh.verts {
            v.pos *= scale;
        }
    }

    info!("num_shells: {}", stats.num_shells);
    info!("num_faces: {}", stats.num_faces);
    info!("num_errors: {}", stats.num_errors);
    info!("num_panics: {}", stats.num_panics);
    (mesh, stats)
}

fn item_defined_transformation(s: &StepFile, t: Id<ItemDefinedTransformation_>) -> DMat4 {
    let i = s.entity(t).expect("Could not get ItemDefinedTransform");

    let (location, axis, ref_direction) = axis2_placement_3d(s,
        i.transform_item_1.cast());
    let t1 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    let (location, axis, ref_direction) = axis2_placement_3d(s,
        i.transform_item_2.cast());
    let t2 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    t2 * t1.try_inverse().expect("Could not invert transform matrix")
}

fn presentation_style_color(s: &StepFile, p: PresentationStyleAssignment)
    -> Option<DVec3>
{
    // AAAAAHHHHH
    s.entity(p)
        .and_then(|p: &PresentationStyleAssignment_| {
                let mut surf = p.styles.iter().filter_map(|y| {
                    // This is an ambiguous parse, so we hard-code the first
                    // Entity item in the enum
                    use PresentationStyleSelect::PreDefinedPresentationStyle;
                    if let PreDefinedPresentationStyle(u) = y {
                        s.entity(u.cast::<SurfaceStyleUsage_>())
                    } else {
                        None
                    }});
                let out = surf.next();
                out
            })
        .and_then(|surf: &SurfaceStyleUsage_|
            s.entity(surf.style.cast::<SurfaceSideStyle_>()))
        .and_then(|surf: &SurfaceSideStyle_| if surf.styles.len() != 1 {
                None
            } else {
                s.entity(surf.styles[0].cast::<SurfaceStyleFillArea_>())
            })
        .map(|surf: &SurfaceStyleFillArea_|
            s.entity(surf.fill_area).expect("Could not get fill_area"))
        .and_then(|fill: &FillAreaStyle_| if fill.fill_styles.len() != 1 {
                None
            } else {
                s.entity(fill.fill_styles[0].cast::<FillAreaStyleColour_>())
            })
        .and_then(|f: &FillAreaStyleColour_|
            s.entity(f.fill_colour.cast::<ColourRgb_>()))
        .map(|c| DVec3::new(c.red, c.green, c.blue))
}

fn cartesian_point(s: &StepFile, a: Id<CartesianPoint_>) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0)
}

fn direction(s: &StepFile, a: Direction) -> DVec3 {
    let p = s.entity(a).expect("Could not get cartesian point");
    DVec3::new(p.direction_ratios[0],
               p.direction_ratios[1],
               p.direction_ratios[2])
}

fn axis2_placement_3d(s: &StepFile, t: Id<Axis2Placement3d_>) -> (DVec3, DVec3, DVec3) {
    let a = s.entity(t).expect("Could not get Axis2Placement3d");
    let location = cartesian_point(s, a.location);
    // TODO: this doesn't necessarily match the behavior of `build_axes`
    let axis = direction(s, a.axis.expect("Missing axis"));
    let ref_direction = match a.ref_direction {
        None => DVec3::new(1.0, 0.0, 0.0),
        Some(r) => direction(s, r),
    };
    (location, axis, ref_direction)
}

fn shell(
    s: &StepFile,
    c: Shell,
    mesh: &mut Mesh,
    stats: &mut Stats,
    styled_item_colors: &HashMap<usize, DVec3>,
    default_color: DVec3,
) {
    match &s[c] {
        Entity::ClosedShell(_) => closed_shell(
            s,
            c.cast(),
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ),
        Entity::OpenShell(_) => open_shell(
            s,
            c.cast(),
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ),
        h => warn!("Skipping {:?} (unknown Shell type)", h),
    }
}

fn open_shell(
    s: &StepFile,
    c: OpenShell,
    mesh: &mut Mesh,
    stats: &mut Stats,
    styled_item_colors: &HashMap<usize, DVec3>,
    default_color: DVec3,
) {
    let cs = s.entity(c).expect("Could not get OpenShell");
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(
            s,
            face.cast(),
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ) {
            error!("Failed to triangulate {:?}: {}", s[*face], err);
        }
    }
    stats.num_shells += 1;
}

fn closed_shell(
    s: &StepFile,
    c: ClosedShell,
    mesh: &mut Mesh,
    stats: &mut Stats,
    styled_item_colors: &HashMap<usize, DVec3>,
    default_color: DVec3,
) {
    let cs = s.entity(c).expect("Could not get ClosedShell");
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(
            s,
            face.cast(),
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ) {
            error!("Failed to triangulate {:?}: {}", s[*face], err);
        }
    }
    stats.num_shells += 1;
}

fn advanced_face(
    s: &StepFile,
    f: AdvancedFace,
    mesh: &mut Mesh,
    stats: &mut Stats,
    styled_item_colors: &HashMap<usize, DVec3>,
    default_color: DVec3,
) -> Result<(), Error> {
    let face = s.entity(f).expect("Could not get AdvancedFace");
    let face_color = styled_item_colors.get(&f.0).copied().unwrap_or(default_color);
    stats.num_faces += 1;
    info!("triangulating face {} (geometry {})", f.0, face.face_geometry.0);

    // Grab the surface, returning early if it's unimplemented
    let mut surf = get_surface(s, face.face_geometry)?;

    // This is the starting point at which we insert new vertices
    let offset = mesh.verts.len();

    // For each contour, project from 3D down to the surface, then
    // start collecting them as constrained edges for triangulation
    let mut edges = Vec::new();
    let v_start = mesh.verts.len();
    let mut num_pts = 0;
    for b in &face.bounds {
        let bound_contours = face_bound(s, *b)?;

        match bound_contours.len() {
            // We should always have non-zero items in the contour
            0 => panic!("Got empty contours for {:?}", face),

            // Special case for a single-vertex point, which shows up in
            // cones: we push it as a Steiner point, but without any
            // associated contours.
            1 => {
                num_pts += 1;
                mesh.verts.push(mesh::Vertex {
                    pos: bound_contours[0],
                    norm: DVec3::zeros(),
                    color: face_color,
                });
            },

            // Default for lists of contour points
            _ => {
                // Record the initial point to close the loop
                let start = num_pts;
                for pt in bound_contours {
                    // The contour marches forward!
                    edges.push((num_pts, num_pts + 1));

                    // Also store this vertex in the 3D triangulation
                    mesh.verts.push(mesh::Vertex {
                        pos: pt,
                        norm: DVec3::zeros(),
                        color: face_color,
                    });
                    num_pts += 1;
                }
                // The last point is a duplicate, because it closes the
                // contours, so we skip it here and reattach the contour to
                // the start.
                num_pts -= 1;
                mesh.verts.pop();

                // Close the loop by returning to the starting point
                edges.pop();
                edges.last_mut().unwrap().1 = start;
            }
        }
    }

    // We inject Stiner points based on the surface type to improve curvature,
    // e.g. for spherical sections.  However, we don't want triagulation to
    // _fail_ due to these points, so if that happens, we nuke the point (by
    // assigning it to the first point in the list, which causes it to get
    // deduplicated), then retry.
    let mut pts = surf.lower_verts(&mut mesh.verts[v_start..])?;
    resolve_crossing_edges(&mut pts, &mut edges, &mut mesh.verts, v_start);
    let bonus_points = pts.len();
    surf.add_steiner_points(&mut pts, &mut mesh.verts);
    let face_id = face.face_geometry.0;
    let n_steiner = pts.len() - bonus_points;
    info!("face {} cdt input: {} pts ({} boundary, {} steiner), {} edges",
          face_id, pts.len(), bonus_points, n_steiner, edges.len());
    let result = std::panic::catch_unwind(|| {
        // TODO: this is only needed because we use pts below to save a debug
        // SVG if this panics.  Once we're confident in never panicking, we
        // can remove this.
        let mut pts = pts.clone();
        let max_retries = n_steiner + 1;
        let mut retries = 0usize;
        loop {
            let mut t = match cdt::Triangulation::new_with_edges(&pts, &edges) {
                Err(e) => break Err(e),
                Ok(t) => t,
            };
            match t.run() {
                Ok(()) => break Ok(t),
                // If triangulation failed due to a Steiner point on a fixed
                // edge, then reassign that point to pts[0] (so it will be
                // ignored as a duplicate)
                Err(cdt::Error::PointOnFixedEdge(p)) if p >= bonus_points => {
                    retries += 1;
                    info!("face {}: PointOnFixedEdge({}), retry {}/{} \
                           ({} pts, {} edges, {} steiner)",
                           face_id, p, retries, max_retries,
                           pts.len(), edges.len(), n_steiner);
                    if retries > max_retries {
                        warn!("face {}: exceeded max retries ({}), giving up",
                              face_id, max_retries);
                        break Err(cdt::Error::PointOnFixedEdge(p));
                    }
                    pts[p] = pts[0];
                    continue;
                },
                Err(e) => {
                    if let Some(dir) = save_debug_svg_dir() {
                        let filename = format!("{}/err{}.svg", dir, face_id);
                        t.save_debug_svg(&filename)
                            .expect("Could not save debug SVG");
                    }
                    break Err(e)
                },
            }
        }
    });
    match result {
        Ok(Ok(t)) => {
            for (a, b, c) in t.triangles() {
                let a = (a + offset) as u32;
                let b = (b + offset) as u32;
                let c = (c + offset) as u32;
                mesh.triangles.push(Triangle { verts:
                    if face.same_sense {
                        U32Vec3::new(a, b, c)
                    } else {
                        U32Vec3::new(a, c, b)
                    }
                });
            }
        },
        Ok(Err(e)) => {
            error!("Got error while triangulating {}: {:?}",
                   face.face_geometry.0, e);
            stats.num_errors += 1;
        },
        Err(e) => {
            error!("Got panic while triangulating {}: {:?}",
                   face.face_geometry.0, e);
            if let Some(dir) = save_debug_svg_dir() {
                let filename = format!("{}/panic{}.svg", dir, face.face_geometry.0);
                cdt::save_debug_panic(&pts, &edges, &filename)
                    .expect("Could not save debug SVG");
            }
            stats.num_panics += 1;
        }
    }
    info!("face {} post-cdt: applying colors/normals ({} verts from v_start)",
          face_id, mesh.verts.len() - v_start);
    for v in &mut mesh.verts[v_start..] {
        v.color = face_color;
    }
    // Flip normals of new vertices, depending on the same_sense flag
    if !face.same_sense {
        for v in &mut mesh.verts[v_start..] {
            v.norm = -v.norm;
        }
    }
    info!("face {} done", face_id);
    Ok(())
}

fn get_surface(s: &StepFile, surf: ap214::Surface) -> Result<Surface, Error> {
    match &s[surf] {
        Entity::CylindricalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_cylinder(axis, ref_direction, location, c.radius.0.0.0))
        },
        Entity::ToroidalSurface(c) => {
            let (location, axis, _ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_torus(location, axis, c.major_radius.0.0.0, c.minor_radius.0.0.0))
        },
        Entity::Plane(p) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, axis, ref_direction) = axis2_placement_3d(s, p.position);
            Ok(Surface::new_plane(axis, ref_direction, location))
        },
        // We treat cones like planes, since that's a valid mapping into 2D
        Entity::ConicalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_cone(axis, ref_direction, location, c.semi_angle.0))
        },
        Entity::SphericalSurface(c) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, _axis, _ref_direction) = axis2_placement_3d(s, c.position);
            Ok(Surface::new_sphere(location, c.radius.0.0.0))
        },
        Entity::BSplineSurfaceWithKnots(b) =>
        {
            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = b.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = b.u_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                b.u_degree.try_into().expect("Got negative degree"),
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = b.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = b.v_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                b.v_degree.try_into().expect("Got negative degree"),
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(s, &b.control_points_list);

            let surf = BSplineSurface::new(
                b.u_closed.0.unwrap() == false,
                b.v_closed.0.unwrap() == false,
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::BSpline(SampledSurface::new(surf)))
        },
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineSurfaceWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType)
            };
            let rational = if let Entity::RationalBSplineSurface(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType)
            };

            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = bspline.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = bspline.u_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let u_knot_vec = KnotVector::from_multiplicities(
                bspline.u_degree.try_into().expect("Got negative degree"),
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = bspline.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = bspline.v_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let v_knot_vec = KnotVector::from_multiplicities(
                bspline.v_degree.try_into().expect("Got negative degree"),
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(
                    s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(ctrl, weight)|
                    ctrl.into_iter()
                        .zip(weight.into_iter())
                        .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                        .collect())
                .collect();

            let surf = NURBSSurface::new(
                bspline.u_closed.0.unwrap() == false,
                bspline.v_closed.0.unwrap() == false,
                u_knot_vec,
                v_knot_vec,
                control_points_list,
            );
            Ok(Surface::NURBS(SampledSurface::new(surf)))

        },
        e => {
            warn!("Could not get surface from {:?}", e);
            Err(Error::UnknownSurfaceType)
        },
    }
}

fn control_points_1d(s: &StepFile, row: &Vec<CartesianPoint>) -> Vec<DVec3> {
    row.iter().map(|p| cartesian_point(s, *p)).collect()
}

fn control_points_2d(s: &StepFile, rows: &Vec<Vec<CartesianPoint>>) -> Vec<Vec<DVec3>> {
    rows.iter()
        .map(|row| control_points_1d(s, row))
        .collect()
}

fn face_bound(s: &StepFile, b: FaceBound) -> Result<Vec<DVec3>, Error> {
    let (bound, orientation) = match &s[b] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        e => panic!("Could not get bound from {:?} at {:?}", e, b),
    };
    match &s[bound] {
        Entity::EdgeLoop(e) => {
            let mut d = edge_loop(s, &e.edge_list)?;
            if !orientation {
                d.reverse()
            }
            Ok(d)
        },
        Entity::VertexLoop(v) => {
            // This is an "edge loop" with a single vertex, which is
            // used for cones and not really anything else.
            Ok(vec![vertex_point(s, v.loop_vertex)])
        }
        e => panic!("{:?} is not an EdgeLoop", e),
    }
}

fn edge_loop(s: &StepFile, edge_list: &[OrientedEdge])
    -> Result<Vec<DVec3>, Error>
{
    let mut out = Vec::new();
    for (i, e) in edge_list.iter().enumerate() {
        // Remove the last item from the list, since it's the beginning
        // of the following list (hopefully)
        if i > 0 {
            out.pop();
        }
        let edge = s.entity(*e).expect("Could not get OrientedEdge");
        let o = edge_curve(s, edge.edge_element.cast(), edge.orientation)?;
        out.extend(o.into_iter());
    }
    Ok(out)
}

fn edge_curve(s: &StepFile, e: EdgeCurve, orientation: bool) -> Result<Vec<DVec3>, Error> {
    let edge_curve = s.entity(e).expect("Could not get EdgeCurve");
    let curve = curve(s, edge_curve, edge_curve.edge_geometry, orientation)?;

    let (start, end) = if orientation {
        (edge_curve.edge_start, edge_curve.edge_end)
    } else {
        (edge_curve.edge_end, edge_curve.edge_start)
    };
    let is_loop = edge_curve.edge_start == edge_curve.edge_end;
    let u = vertex_point(s, start);
    let v = vertex_point(s, end);
    Ok(curve.build(u, v, is_loop))
}

fn curve(s: &StepFile, edge_curve: &ap214::EdgeCurve_,
         curve_id: ap214::Curve, orientation: bool) -> Result<Curve, Error>
{
    Ok(match &s[curve_id] {
        Entity::Circle(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_circle(location, axis, ref_direction, c.radius.0.0.0,
                              edge_curve.edge_start == edge_curve.edge_end,
                              edge_curve.same_sense ^ !orientation)
        },
        Entity::Ellipse(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast());
            Curve::new_ellipse(location, axis, ref_direction,
                               c.semi_axis_1.0.0.0, c.semi_axis_2.0.0.0,
                               edge_curve.edge_start == edge_curve.edge_end,
                               edge_curve.same_sense ^ !orientation)
        },
        Entity::BSplineCurveWithKnots(c) => {
            if c.self_intersect.0 == Some(true) {
                return Err(Error::SelfIntersectingCurve);
            }

            let control_points_list = control_points_1d(
                s, &c.control_points_list);

            let knots: Vec<f64> = c.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = c.knot_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                c.degree.try_into().expect("Got negative degree"),
                &knots, &multiplicities);

            let open = c.closed_curve.0 != Some(true);
            let curve = nurbs::BSplineCurve::new(
                open,
                knot_vec,
                control_points_list,
            );
            Curve::BSplineCurveWithKnots(SampledCurve::new(curve))
        },
        Entity::ComplexEntity(v) if v.len() == 2 => {
            let bspline = if let Entity::BSplineCurveWithKnots(b) = &v[0] {
                b
            } else {
                warn!("Could not get BSplineCurveWithKnots from {:?}", v[0]);
                return Err(Error::UnknownCurveType)
            };
            let rational = if let Entity::RationalBSplineCurve(b) = &v[1] {
                b
            } else {
                warn!("Could not get RationalBSplineCurve from {:?}", v[1]);
                return Err(Error::UnknownCurveType)
            };
            let knots: Vec<f64> = bspline.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = bspline.knot_multiplicities.iter()
                .map(|&k| k.try_into().expect("Got negative multiplicity"))
                .collect();
            let knot_vec = KnotVector::from_multiplicities(
                bspline.degree.try_into().expect("Got negative degree"),
                &knots, &multiplicities);

            let control_points_list = control_points_1d(
                    s, &bspline.control_points_list)
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                .collect();

            let open = bspline.closed_curve.0 != Some(true);
            let curve = nurbs::NURBSCurve::new(
                open,
                knot_vec,
                control_points_list,
            );
            Curve::NURBSCurve(SampledCurve::new(curve))
        },
        Entity::SurfaceCurve(v) => {
            curve(s, edge_curve, v.curve_3d, orientation)?
        },
        Entity::SeamCurve(v) => {
            curve(s, edge_curve, v.curve_3d, orientation)?
        },
        // The Line type ignores pnt / dir and just uses u and v
        Entity::Line(_) => Curve::new_line(),
        e => {
            warn!("Could not get edge from {:?}", e);
            return Err(Error::UnknownCurveType);
        },
    })
}

fn vertex_point(s: &StepFile, v: Vertex) -> DVec3 {
    cartesian_point(s,
        s.entity(v.cast::<VertexPoint_>())
            .expect("Could not get VertexPoint")
            .vertex_geometry
            .cast())
}

/// Compute intersection parameters (t, s) for segments A-B and C-D.
/// Returns Some((t, s)) if the segments cross at interior points (not
/// at endpoints), where the intersection is at A + t*(B-A) = C + s*(D-C).
fn segment_intersection_params(
    a: (f64, f64), b: (f64, f64),
    c: (f64, f64), d: (f64, f64),
) -> Option<(f64, f64)> {
    let denom = (b.0 - a.0) * (d.1 - c.1) - (b.1 - a.1) * (d.0 - c.0);
    if denom.abs() < 1e-12 {
        return None; // parallel or coincident
    }
    let t = ((c.0 - a.0) * (d.1 - c.1) - (c.1 - a.1) * (d.0 - c.0)) / denom;
    let s = ((c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)) / denom;
    let eps = 1e-10;
    if t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps {
        Some((t, s))
    } else {
        None
    }
}

/// Pre-process edges to resolve any crossings before feeding them to the CDT.
/// When two constrained edges cross, split both at the intersection point by
/// inserting a new shared vertex.
fn resolve_crossing_edges(
    pts: &mut Vec<(f64, f64)>,
    edges: &mut Vec<(usize, usize)>,
    verts: &mut Vec<mesh::Vertex>,
    v_start: usize,
) {
    // Limit iterations to prevent pathological runaway
    for _ in 0..100 {
        let mut found = None;
        'outer: for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                // Skip edges that share an endpoint
                if edges[i].0 == edges[j].0 || edges[i].0 == edges[j].1
                || edges[i].1 == edges[j].0 || edges[i].1 == edges[j].1 {
                    continue;
                }
                if let Some((t, _s)) = segment_intersection_params(
                    pts[edges[i].0], pts[edges[i].1],
                    pts[edges[j].0], pts[edges[j].1],
                ) {
                    found = Some((i, j, t));
                    break 'outer;
                }
            }
        }
        let (i, j, t) = match found {
            Some(v) => v,
            None => break, // no more crossings
        };

        // Compute 2D intersection point
        let (ax, ay) = pts[edges[i].0];
        let (bx, by) = pts[edges[i].1];
        let new_2d = (ax + t * (bx - ax), ay + t * (by - ay));

        // Compute 3D vertex by interpolation along edge i
        let va = verts[v_start + edges[i].0];
        let vb = verts[v_start + edges[i].1];
        let new_3d = mesh::Vertex {
            pos: va.pos * (1.0 - t) + vb.pos * t,
            norm: DVec3::zeros(),
            color: DVec3::zeros(),
        };

        let new_idx = pts.len();
        pts.push(new_2d);
        verts.push(new_3d);

        // Split edge i: (a, b) → (a, new), (new, b)
        let (a, b) = edges[i];
        edges[i] = (a, new_idx);
        edges.push((new_idx, b));

        // Split edge j: (c, d) → (c, new), (new, d)
        let (c, d) = edges[j];
        edges[j] = (c, new_idx);
        edges.push((new_idx, d));
    }
}
