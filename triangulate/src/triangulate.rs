use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

use nalgebra_glm as glm;
use glm::{DVec3, DVec4, DMat4, U32Vec3};
use log::{debug, error, info, warn};

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

#[derive(Copy, Clone, Debug)]
struct OccurrenceInstance<'a> {
    child_product: ProductDefinition<'a>,
    parent_rep: Representation<'a>,
    child_rep: Representation<'a>,
    transform: DMat4,
}

fn transformed_representation_relationship<'a>(
    s: &'a StepFile,
    id: ShapeRepresentationRelationship<'a>,
) -> Option<&'a RepresentationRelationshipWithTransformation_<'a>> {
    match &s.0[id.0] {
        Entity::RepresentationRelationshipWithTransformation(rel) => Some(rel),
        Entity::ComplexEntity(subs) => subs.iter()
            .find_map(|sub| RepresentationRelationshipWithTransformation_::try_from_entity(sub)),
        _ => None,
    }
}

fn collect_shape_instances<'a>(
    s: &'a StepFile,
    rep_instances: &HashMap<Representation<'a>, Vec<DMat4>>,
    shape_rep_relationship: &HashMap<Representation<'a>, Vec<Representation<'a>>>,
) -> HashMap<RepresentationItem<'a>, Vec<DMat4>> {
    let mut todo: Vec<_> = rep_instances
        .iter()
        .flat_map(|(rep, mats)| mats.iter().copied().map(move |mat| (*rep, mat)))
        .collect();
    let mut to_mesh: HashMap<RepresentationItem<'a>, Vec<DMat4>> = HashMap::new();

    while let Some((id, mat)) = todo.pop() {
        if let Some(children) = shape_rep_relationship.get(&id) {
            for child in children {
                todo.push((*child, mat));
            }
        }
        // Bind this transform to the RepresentationItem, which is
        // either a ManifoldSolidBrep or a ShellBasedSurfaceModel
        let items = match &s[id] {
            Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
            Entity::ShapeRepresentation(b) => &b.items,
            Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
            e => {
                warn!("Skipping {:?} (not a supported representation)", e);
                continue;
            },
        };

        for m in items.iter() {
            match &s[*m] {
                Entity::ManifoldSolidBrep(_)
                | Entity::BrepWithVoids(_)
                | Entity::ShellBasedSurfaceModel(_) => {
                    to_mesh.entry(*m).or_default().push(mat);
                }
                Entity::Axis2Placement3d(_) | Entity::MappedItem(_) => (),
                e => warn!("Skipping {:?}", e),
            }
        }
    }

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
            .for_each(|i| {
                to_mesh.entry(i).or_default().push(DMat4::identity());
            });
    }

    to_mesh
}

fn collect_product_roots(s: &StepFile) -> HashSet<usize> {
    let all_products: HashSet<_> = s.0.iter()
        .enumerate()
        .filter(|(_i, e)| matches!(e, Entity::ProductDefinition(_)))
        .map(|(i, _)| i)
        .collect();
    let child_products: HashSet<_> = s.0.iter()
        .filter_map(|e| NextAssemblyUsageOccurrence_::try_from_entity(e))
        .map(|rel| rel.related_product_definition.0)
        .collect();
    all_products.into_iter()
        .filter(|idx| !child_products.contains(idx))
        .collect()
}

fn collect_product_representations<'a>(
    s: &'a StepFile,
) -> HashMap<ProductDefinition<'a>, Vec<Representation<'a>>> {
    let mut reps: HashMap<ProductDefinition<'a>, Vec<Representation<'a>>> = HashMap::new();
    for sdr in s.0.iter()
        .filter_map(|e| ShapeDefinitionRepresentation_::try_from_entity(e))
    {
        let Some(pds) = s.entity::<ProductDefinitionShape_>(sdr.definition.cast()) else {
            continue;
        };
        let Some(_product) = s.entity::<ProductDefinition_>(pds.definition.cast()) else {
            continue;
        };
        reps.entry(pds.definition.cast())
            .or_default()
            .push(sdr.used_representation);
    }

    for reps in reps.values_mut() {
        reps.sort_by_key(|rep| rep.0);
        reps.dedup();
    }
    reps
}

fn collect_occurrence_instances<'a>(
    s: &'a StepFile,
    product_reps: &HashMap<ProductDefinition<'a>, Vec<Representation<'a>>>,
) -> HashMap<ProductDefinition<'a>, Vec<OccurrenceInstance<'a>>> {
    let occurrence_shape_defs: HashMap<_, _> = s.0.iter()
        .enumerate()
        .filter_map(|(idx, e)| {
            let pds = ProductDefinitionShape_::try_from_entity(e)?;
            s.entity::<NextAssemblyUsageOccurrence_>(pds.definition.cast())
                .map(|occ| (Id::new(idx), occ))
        })
        .collect();

    let mut occurrences: HashMap<ProductDefinition<'a>, Vec<OccurrenceInstance<'a>>> = HashMap::new();
    for cdsr in s.0.iter()
        .filter_map(|e| ContextDependentShapeRepresentation_::try_from_entity(e))
    {
        let Some(occ) = occurrence_shape_defs.get(&cdsr.represented_product_relation) else {
            continue;
        };
        let Some(rel) = transformed_representation_relationship(s, cdsr.representation_relation) else {
            warn!(
                "Skipping context-dependent shape representation {:?}: expected transformed representation relationship",
                cdsr
            );
            continue;
        };
        let transform = match item_defined_transformation(s, rel.transformation_operator.cast()) {
            Ok(mat) => mat,
            Err(err) => {
                warn!("Skipping transform relationship {:?}: {}", rel, err);
                continue;
            }
        };

        let parent_reps = product_reps
            .get(&occ.relating_product_definition)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let child_reps = product_reps
            .get(&occ.related_product_definition)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let rep1_is_parent = parent_reps.contains(&rel.rep_1);
        let rep2_is_parent = parent_reps.contains(&rel.rep_2);
        let rep1_is_child = child_reps.contains(&rel.rep_1);
        let rep2_is_child = child_reps.contains(&rel.rep_2);

        let oriented = if rep1_is_child && rep2_is_parent {
            Some((rel.rep_2, rel.rep_1, transform))
        } else if rep1_is_parent && rep2_is_child {
            let Some(inv) = transform.try_inverse() else {
                warn!("Skipping non-invertible transform relationship {:?}", rel);
                continue;
            };
            Some((rel.rep_1, rel.rep_2, inv))
        } else {
            None
        };

        let Some((parent_rep, child_rep, transform)) = oriented else {
            warn!(
                "Skipping occurrence parent_product=#{} child_product=#{} rel reps (#{} -> #{}) due to ambiguous ownership",
                occ.relating_product_definition.0,
                occ.related_product_definition.0,
                rel.rep_1.0,
                rel.rep_2.0,
            );
            continue;
        };

        occurrences.entry(occ.relating_product_definition)
            .or_default()
            .push(OccurrenceInstance {
                child_product: occ.related_product_definition,
                parent_rep,
                child_rep,
                transform,
            });
    }
    occurrences
}

fn collect_rep_instances<'a>(s: &'a StepFile) -> HashMap<Representation<'a>, Vec<DMat4>> {
    let product_roots = collect_product_roots(s);
    let product_reps = collect_product_representations(s);
    let occurrences = collect_occurrence_instances(s, &product_reps);

    let mut todo = Vec::new();
    for product_idx in product_roots {
        let product = Id::new(product_idx);
        if let Some(reps) = product_reps.get(&product) {
            todo.extend(
                reps.iter()
                    .copied()
                    .map(|rep| (product, rep, DMat4::identity()))
            );
        }
    }

    let mut rep_instances: HashMap<Representation<'a>, Vec<DMat4>> = HashMap::new();
    while let Some((product, rep, mat)) = todo.pop() {
        rep_instances.entry(rep).or_default().push(mat);
        if let Some(children) = occurrences.get(&product) {
            for occ in children {
                if occ.parent_rep != rep {
                    continue;
                }
                todo.push((
                    occ.child_product,
                    occ.child_rep,
                    mat * occ.transform,
                ));
            }
        }
    }

    for mats in rep_instances.values_mut() {
        mats.shrink_to_fit();
    }
    rep_instances
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
    let styled_item_colors: HashMap<usize, DVec3> = s.0.iter()
        .filter_map(|e| MechanicalDesignGeometricPresentationRepresentation_::try_from_entity(e))
        .flat_map(|m| m.items.iter())
        .filter_map(|item| s.entity(item.cast::<StyledItem_>()))
        .filter_map(|styled|
            if styled.styles.len() != 1 {
                None
            } else {
                presentation_style_color(s, styled.styles[0])
                    .map(|c| (styled.item.0, c))
            })
        .collect();

    // Store a map of ShapeRepresentationRelationships, which some models
    // use to map from a product-level ShapeRepresentation to the concrete
    // AdvancedBrepShapeRepresentation / ManifoldSurfaceShapeRepresentation
    // that contains the meshable items.
    //
    // The STEP attribute names (`rep_1`, `rep_2`) are not a reliable
    // parent→child direction in practice: e.g. Inventor/PDElib files often
    // write `SHAPE_REPRESENTATION_RELATIONSHIP(...,#advanced_brep,#shape)`.
    // Build the traversal edge from the non-mesh representation to the
    // mesh-bearing representation, regardless of attribute order.
    let representation_has_mesh_items = |rep: Id<_>| -> bool {
        let items = match &s[rep] {
            Entity::AdvancedBrepShapeRepresentation(b) => &b.items,
            Entity::ShapeRepresentation(b) => &b.items,
            Entity::ManifoldSurfaceShapeRepresentation(b) => &b.items,
            _ => return false,
        };
        items.iter().any(|m| matches!(
            &s[*m],
            Entity::ManifoldSolidBrep(_)
                | Entity::BrepWithVoids(_)
                | Entity::ShellBasedSurfaceModel(_)
        ))
    };

    let mut shape_rep_relationship: HashMap<Id<_>, Vec<Id<_>>> = HashMap::new();
    for (r1, r2) in s.0.iter()
        .filter_map(|e| ShapeRepresentationRelationship_::try_from_entity(e))
        .map(|e| (e.rep_1, e.rep_2))
    {
        match (representation_has_mesh_items(r1), representation_has_mesh_items(r2)) {
            (true, false) => shape_rep_relationship.entry(r2).or_default().push(r1),
            (false, true) => shape_rep_relationship.entry(r1).or_default().push(r2),
            _ => continue,
        };
    }

    let rep_instances = collect_rep_instances(s);
    if rep_instances.is_empty() {
        warn!("No semantic representation instances found");
    } else {
        info!("Semantic representation instances: {}", rep_instances.len());
    }
    let to_mesh = collect_shape_instances(
        s,
        &rep_instances,
        &shape_rep_relationship,
    );

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
                crate::timing::time("shape:mesh_faces", || match &s[*id] {
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
                    },
                });

                // Build copies of the mesh by copying and applying transforms
                let v_end = mesh.verts.len();
                let t_end = mesh.triangles.len();
                crate::timing::time("shape:instance_copies", || {
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
                });
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
    if stats.num_errors > 0 || stats.num_panics > 0 {
        warn!(
            "triangulation finished with {} face errors and {} panics",
            stats.num_errors, stats.num_panics
        );
    }
    (mesh, stats)
}

fn item_defined_transformation(s: &StepFile, t: Id<ItemDefinedTransformation_>)
    -> Result<DMat4, Error>
{
    let i = s.entity(t).ok_or(Error::InvalidStepEntity("ItemDefinedTransformation"))?;

    let (location, axis, ref_direction) = axis2_placement_3d(s, i.transform_item_1.cast())?;
    let t1 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    let (location, axis, ref_direction) = axis2_placement_3d(s, i.transform_item_2.cast())?;
    let t2 = Surface::make_affine_transform(axis,
        ref_direction,
        axis.cross(&ref_direction),
        location);

    let t1i = t1.try_inverse()
        .ok_or(Error::SingularTransform("item-defined transformation"))?;
    Ok(t2 * t1i)
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
        .and_then(|surf: &SurfaceStyleFillArea_|
            s.entity(surf.fill_area))
        .and_then(|fill: &FillAreaStyle_| if fill.fill_styles.len() != 1 {
                None
            } else {
                s.entity(fill.fill_styles[0].cast::<FillAreaStyleColour_>())
            })
        .and_then(|f: &FillAreaStyleColour_|
            s.entity(f.fill_colour.cast::<ColourRgb_>()))
        .map(|c| DVec3::new(c.red, c.green, c.blue))
}

fn cartesian_point(s: &StepFile, a: Id<CartesianPoint_>) -> Result<DVec3, Error> {
    let p = s.entity(a).ok_or(Error::InvalidStepEntity("CartesianPoint"))?;
    if p.coordinates.len() < 3 {
        return Err(Error::InvalidGeometry("cartesian point has fewer than 3 coordinates"));
    }
    Ok(DVec3::new(p.coordinates[0].0, p.coordinates[1].0, p.coordinates[2].0))
}

fn direction(s: &StepFile, a: Direction) -> Result<DVec3, Error> {
    let p = s.entity(a).ok_or(Error::InvalidStepEntity("Direction"))?;
    if p.direction_ratios.len() < 3 {
        return Err(Error::InvalidGeometry("direction has fewer than 3 ratios"));
    }
    Ok(DVec3::new(p.direction_ratios[0],
               p.direction_ratios[1],
               p.direction_ratios[2]))
}

fn axis2_placement_3d(s: &StepFile, t: Id<Axis2Placement3d_>)
    -> Result<(DVec3, DVec3, DVec3), Error>
{
    let a = s.entity(t).ok_or(Error::InvalidStepEntity("Axis2Placement3d"))?;
    let location = cartesian_point(s, a.location)?;
    // TODO: this doesn't necessarily match the behavior of `build_axes`
    let axis = direction(s, a.axis.ok_or(Error::MissingStepField("Axis2Placement3d.axis"))?)?;
    let ref_direction = match a.ref_direction {
        None => DVec3::new(1.0, 0.0, 0.0),
        Some(r) => direction(s, r)?,
    };
    Ok((location, axis, ref_direction))
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
    let Some(cs) = s.entity(c) else {
        error!("Failed to get OpenShell {:?}", c);
        stats.num_errors += 1;
        return;
    };
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(
            s,
            *face,
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ) {
            // Per-face failures are common on large boards and summarised
            // once at the end of triangulate(); keep the per-face detail off
            // the console (console logging is expensive in wasm workers).
            stats.num_errors += 1;
            debug!("Failed to triangulate {:?}: {}", s[*face], err);
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
    let Some(cs) = s.entity(c) else {
        error!("Failed to get ClosedShell {:?}", c);
        stats.num_errors += 1;
        return;
    };
    for face in &cs.cfs_faces {
        if let Err(err) = advanced_face(
            s,
            *face,
            mesh,
            stats,
            styled_item_colors,
            default_color,
        ) {
            // Per-face failures are common on large boards and summarised
            // once at the end of triangulate(); keep the per-face detail off
            // the console (console logging is expensive in wasm workers).
            stats.num_errors += 1;
            debug!("Failed to triangulate {:?}: {}", s[*face], err);
        }
    }
    stats.num_shells += 1;
}

fn advanced_face(
    s: &StepFile,
    f: Face,
    mesh: &mut Mesh,
    stats: &mut Stats,
    styled_item_colors: &HashMap<usize, DVec3>,
    default_color: DVec3,
) -> Result<(), Error> {
    // Closed shells may legally reference either ADVANCED_FACE or the more
    // general FACE_SURFACE; OCCT/KiCad translate both through FaceSurface.
    let (bounds, face_geometry, same_sense) = match &s[f] {
        Entity::AdvancedFace(face) => (&face.bounds[..], face.face_geometry, face.same_sense),
        Entity::FaceSurface(face) => (&face.bounds[..], face.face_geometry, face.same_sense),
        _ => return Err(Error::InvalidStepEntity("FaceSurface")),
    };
    let face_color = styled_item_colors.get(&f.0).copied().unwrap_or(default_color);
    stats.num_faces += 1;
    info!("triangulating face {} (geometry {})", f.0, face_geometry.0);

    // This is the starting point at which we insert new vertices
    let offset = mesh.verts.len();

    // For each contour, project from 3D down to the surface, then
    // start collecting them as constrained edges for triangulation
    let mut edges = Vec::new();
    let mut unwrap_ranges = Vec::new();
    let mut boundary_points = Vec::new();
    let v_start = mesh.verts.len();
    let mut num_pts = 0;
    for b in bounds {
        let (bound_contours, edge_loop_len) =
            crate::timing::time("face:face_bound", || face_bound(s, *b))?;
        boundary_points.extend_from_slice(&bound_contours);

        match bound_contours.len() {
            // We should always have non-zero items in the contour
            0 => return Err(Error::InvalidGeometry("face bound produced empty contour")),

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
                let edge_start = edges.len();
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
                let last = edges.last_mut()
                    .ok_or(Error::InvalidGeometry("contour loop had no edges"))?;
                last.1 = start;
                if edge_loop_len > 0 {
                    unwrap_ranges.push((edge_start, edges.len(), edge_loop_len == 1));
                }
            }
        }
    }

    // Swept surfaces use the actual trims to choose a finite NURBS domain.
    let mut surf = crate::timing::time("face:get_surface",
        || get_surface(s, face_geometry, &boundary_points))?;

    // Add curvature samples before constraint insertion. The CDT subdivides
    // constraints at existing vertices, including samples exactly on an edge.
    let mut pts = crate::timing::time("face:lower_verts",
        || surf.lower_verts(&mut mesh.verts[v_start..], &edges, same_sense))?;
    crate::timing::time("face:unwrap_periodic",
        || surf.unwrap_periodic(&mut pts, &edges, &unwrap_ranges));
    crate::timing::time("face:resolve_crossing_edges",
        || resolve_crossing_edges(&mut pts, &mut edges, &mut mesh.verts, v_start));
    let bonus_points = pts.len();
    crate::timing::time("face:add_steiner_points",
        || surf.add_steiner_points(&mut pts, &mut mesh.verts));
    let face_id = face_geometry.0;
    let n_steiner = pts.len() - bonus_points;
    info!("face {} cdt input: {} pts ({} boundary, {} steiner), {} edges",
          face_id, pts.len(), bonus_points, n_steiner, edges.len());
    if std::env::var("DUMP_FACE").ok().as_deref() == Some(&face_id.to_string()) {
        eprintln!("DUMP_FACE {}: pts={:?}", face_id, pts);
        eprintln!("DUMP_FACE {}: edges={:?}", face_id, edges);
    }
    // Preserve per-face panic diagnostics without mutating the input on error.
    let result = crate::timing::time("face:cdt", ||
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut t = cdt::Triangulation::new_with_edges(&pts, &edges)?;
        if let Err(e) = t.run() {
            if let Some(dir) = save_debug_svg_dir() {
                let filename = format!("{}/err{}.svg", dir, face_id);
                if let Err(err) = t.save_debug_svg(&filename) {
                    warn!("Could not save debug SVG {}: {}", filename, err);
                }
            }
            return Err(e);
        }
        Ok(t)
    })));
    match result {
        Err(_panic) => {
            warn!("face {}: panicked during CDT triangulation, skipping face", face_id);
            stats.num_panics += 1;
        },
        Ok(Ok(t)) => {
            let triangle_start = mesh.triangles.len();
            for (a, b, c) in t.triangles() {
                let a = (a + offset) as u32;
                let b = (b + offset) as u32;
                let c = (c + offset) as u32;
                mesh.triangles.push(Triangle { verts:
                    if same_sense {
                        U32Vec3::new(a, b, c)
                    } else {
                        U32Vec3::new(a, c, b)
                    }
                });
            }
            if mesh.triangles.len() == triangle_start {
                debug!("Got error while triangulating {}: empty face tessellation", face_id);
                stats.num_errors += 1;
            }
        },
        Ok(Err(e)) => {
            debug!(
                "Got error while triangulating {}: {:?}",
                face_geometry.0,
                e
            );
            stats.num_errors += 1;
        },
    }
    info!("face {} post-cdt: applying colors/normals ({} verts from v_start)",
          face_id, mesh.verts.len() - v_start);
    for v in &mut mesh.verts[v_start..] {
        v.color = face_color;
    }
    // Flip normals of new vertices, depending on the same_sense flag
    if !same_sense {
        for v in &mut mesh.verts[v_start..] {
            v.norm = -v.norm;
        }
    }
    info!("face {} done", face_id);
    Ok(())
}

#[derive(Debug)]
struct HomogeneousCurve {
    open: bool,
    knots: KnotVector,
    control_points: Vec<DVec4>,
}

// Tangent-intersection controls for four rational quadratic circle spans.
// Reuse the first control at the seam exactly, without sin(2π) roundoff.
const CIRCLE_CONTROLS: [(f64, f64); 9] = [
    (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
    (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (1.0, 0.0),
];

fn homogeneous_curve(s: &StepFile, curve: ap214::Curve) -> Result<HomogeneousCurve, Error> {
    match &s[curve] {
        Entity::BSplineCurveWithKnots(b) => {
            let points = control_points_1d(s, &b.control_points_list)?
                .into_iter().map(|p| DVec4::new(p.x, p.y, p.z, 1.0)).collect();
            Ok(HomogeneousCurve {
                open: b.closed_curve.0 != Some(true),
                knots: curve_knot_vector(b)?,
                control_points: points,
            })
        },
        Entity::ComplexEntity(parts) => {
            let b = parts.iter().find_map(|e| match e {
                Entity::BSplineCurveWithKnots(b) => Some(b), _ => None,
            }).ok_or(Error::UnknownCurveType)?;
            let r = parts.iter().find_map(|e| match e {
                Entity::RationalBSplineCurve(r) => Some(r), _ => None,
            }).ok_or(Error::UnknownCurveType)?;
            let points = control_points_1d(s, &b.control_points_list)?.into_iter()
                .zip(r.weights_data.iter())
                .map(|(p, &w)| DVec4::new(p.x * w, p.y * w, p.z * w, w))
                .collect();
            Ok(HomogeneousCurve {
                open: b.closed_curve.0 != Some(true),
                knots: curve_knot_vector(b)?,
                control_points: points,
            })
        },
        Entity::Circle(c) => conic_curve(s, c.position.cast(), c.radius.0.0.0, c.radius.0.0.0),
        Entity::Ellipse(c) => conic_curve(s, c.position.cast(), c.semi_axis_1.0.0.0, c.semi_axis_2.0.0.0),
        _ => Err(Error::UnknownCurveType),
    }
}

fn curve_knot_vector(b: &BSplineCurveWithKnots_) -> Result<KnotVector, Error> {
    let knots: Vec<f64> = b.knots.iter().map(|k| k.0).collect();
    let multiplicities: Vec<usize> = b.knot_multiplicities.iter()
        .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative curve multiplicity")))
        .collect::<Result<_, _>>()?;
    Ok(KnotVector::from_multiplicities(
        b.degree.try_into().map_err(|_| Error::NumericConversion("negative curve degree"))?,
        &knots, &multiplicities))
}

fn conic_curve(s: &StepFile, position: Axis2Placement3d, a: f64, b: f64)
    -> Result<HomogeneousCurve, Error>
{
    let (center, axis, x) = axis2_placement_3d(s, position)?;
    let z = axis.normalize();
    let x = (x - z * x.dot(&z)).normalize();
    let y = z.cross(&x);
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let control_points = CIRCLE_CONTROLS.iter().enumerate().map(|(i, &(u, v))| {
        let w = if i % 2 == 0 { 1.0 } else { q };
        let p = center + x * (a * u) + y * (b * v);
        DVec4::new(p.x * w, p.y * w, p.z * w, w)
    }).collect();
    Ok(HomogeneousCurve {
        open: false,
        knots: KnotVector::from_multiplicities(2, &[0.0, 0.25, 0.5, 0.75, 1.0], &[3, 2, 2, 2, 3]),
        control_points,
    })
}

fn extrusion_surface(curve: HomogeneousCurve, vector: DVec3, boundary: &[DVec3])
    -> Result<Surface, Error>
{
    let denominator = vector.norm_squared();
    if denominator == 0.0 { return Err(Error::InvalidGeometry("zero extrusion vector")); }
    let mut range = (f64::INFINITY, f64::NEG_INFINITY);
    for h in &curve.control_points {
        let p = h.xyz() / h.w;
        for b in boundary {
            let v = (b - p).dot(&vector) / denominator;
            range.0 = range.0.min(v);
            range.1 = range.1.max(v);
        }
    }
    if !range.0.is_finite() || range.0 == range.1 {
        return Err(Error::InvalidGeometry("extrusion has empty parameter range"));
    }
    let controls = curve.control_points.into_iter().map(|h| vec![
        h + DVec4::new(vector.x * h.w * range.0, vector.y * h.w * range.0, vector.z * h.w * range.0, 0.0),
        h + DVec4::new(vector.x * h.w * range.1, vector.y * h.w * range.1, vector.z * h.w * range.1, 0.0),
    ]).collect();
    let surface = NURBSSurface::new(curve.open, true, curve.knots,
        KnotVector::from_multiplicities(1, &[range.0, range.1], &[2, 2]), controls);
    Ok(Surface::NURBS(SampledSurface::new(surface)))
}

fn revolution_surface(curve: HomogeneousCurve, origin: DVec3, axis: DVec3)
    -> Result<Surface, Error>
{
    if axis.norm_squared() == 0.0 {
        return Err(Error::InvalidGeometry("zero revolution axis"));
    }
    let axis = axis.normalize();
    let q = std::f64::consts::FRAC_1_SQRT_2;
    // ISO 10303-42 defines revolution as the first parameter and the basis
    // curve as the second. Swapping them reverses the surface normal.
    let controls = CIRCLE_CONTROLS.iter().enumerate().map(|(j, &(x, y))| {
        let wj = if j % 2 == 0 { 1.0 } else { q };
        curve.control_points.iter().map(|h| {
            let p = h.xyz() / h.w;
            let axial = origin + axis * (p - origin).dot(&axis);
            let radial = p - axial;
            // The first parameter is the standard four-span rational
            // quadratic circle, not an angular parameter.
            let point = axial + radial * x + axis.cross(&radial) * y;
            let weight = h.w * wj;
            DVec4::new(point.x * weight, point.y * weight, point.z * weight, weight)
        }).collect()
    }).collect();
    let circle_knots = KnotVector::from_multiplicities(
        2, &[0.0, 0.25, 0.5, 0.75, 1.0], &[3, 2, 2, 2, 3]);
    let surface = NURBSSurface::new(false, curve.open, circle_knots, curve.knots, controls);
    Ok(Surface::NURBS(SampledSurface::new(surface)))
}

fn get_surface(s: &StepFile, surf: ap214::Surface, boundary: &[DVec3]) -> Result<Surface, Error> {
    match &s[surf] {
        Entity::CylindricalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position)?;
            Surface::new_cylinder(axis, ref_direction, location, c.radius.0.0.0)
        },
        Entity::ToroidalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position)?;
            Surface::new_torus_with_ref_direction(
                location, axis, ref_direction,
                c.major_radius.0.0.0, c.minor_radius.0.0.0)
        },
        Entity::DegenerateToroidalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position)?;
            let major = c.major_radius.0.0.0;
            let minor = c.minor_radius.0.0.0;
            if !(0.0 < major && major < minor) {
                return Err(Error::InvalidGeometry("degenerate torus requires 0 < major < minor"));
            }
            // For the inner lemon, u' = u + π and v' = π - v turn the
            // negative radial branch into a positive one with major radius
            // -R. This also selects the spec's outward normal: away from the
            // furthest, rather than nearest, point on the major circle.
            Surface::new_torus_with_ref_direction(location, axis, ref_direction,
                if c.select_outer { major } else { -major }, minor)
        },
        Entity::Plane(p) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, axis, ref_direction) = axis2_placement_3d(s, p.position)?;
            Surface::new_plane(axis, ref_direction, location)
        },
        // We treat cones like planes, since that's a valid mapping into 2D
        Entity::ConicalSurface(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position)?;
            Surface::new_cone(axis, ref_direction, location, c.semi_angle.0)
        },
        Entity::SphericalSurface(c) => {
            // We'll ignore axis and ref_direction in favor of building an
            // orthonormal basis later on
            let (location, _axis, _ref_direction) = axis2_placement_3d(s, c.position)?;
            Surface::new_sphere(location, c.radius.0.0.0)
        },
        Entity::SurfaceOfLinearExtrusion(e) => {
            let v = s.entity(e.extrusion_axis)
                .ok_or(Error::InvalidStepEntity("Vector"))?;
            let vector = direction(s, v.orientation)?.normalize() * v.magnitude.0;
            extrusion_surface(homogeneous_curve(s, e.swept_curve)?, vector, boundary)
        },
        Entity::SurfaceOfRevolution(r) => {
            let placement = s.entity(r.axis_position)
                .ok_or(Error::InvalidStepEntity("Axis1Placement"))?;
            let origin = cartesian_point(s, placement.location)?;
            let axis = direction(s, placement.axis
                .ok_or(Error::MissingStepField("Axis1Placement.axis"))?)?;
            revolution_surface(homogeneous_curve(s, r.swept_curve)?, origin, axis)
        },
        Entity::BSplineSurfaceWithKnots(b) =>
        {
            // TODO: make KnotVector::from_multiplicies accept iterators?
            let u_knots: Vec<f64> = b.u_knots.iter().map(|k| k.0).collect();
            let u_multiplicities: Vec<usize> = b.u_multiplicities.iter()
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative u multiplicity")))
                .collect::<Result<_, _>>()?;
            let u_knot_vec = KnotVector::from_multiplicities(
                b.u_degree.try_into().map_err(|_| Error::NumericConversion("negative u degree"))?,
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = b.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = b.v_multiplicities.iter()
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative v multiplicity")))
                .collect::<Result<_, _>>()?;
            let v_knot_vec = KnotVector::from_multiplicities(
                b.v_degree.try_into().map_err(|_| Error::NumericConversion("negative v degree"))?,
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(s, &b.control_points_list)?;

            let surf = BSplineSurface::new(
                b.u_closed.0 != Some(true),
                b.v_closed.0 != Some(true),
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
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative u multiplicity")))
                .collect::<Result<_, _>>()?;
            let u_knot_vec = KnotVector::from_multiplicities(
                bspline.u_degree.try_into().map_err(|_| Error::NumericConversion("negative u degree"))?,
                &u_knots, &u_multiplicities);

            let v_knots: Vec<f64> = bspline.v_knots.iter().map(|k| k.0).collect();
            let v_multiplicities: Vec<usize> = bspline.v_multiplicities.iter()
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative v multiplicity")))
                .collect::<Result<_, _>>()?;
            let v_knot_vec = KnotVector::from_multiplicities(
                bspline.v_degree.try_into().map_err(|_| Error::NumericConversion("negative v degree"))?,
                &v_knots, &v_multiplicities);

            let control_points_list = control_points_2d(
                    s, &bspline.control_points_list)?
                .into_iter()
                .zip(rational.weights_data.iter())
                .map(|(ctrl, weight)|
                    ctrl.into_iter()
                        .zip(weight.into_iter())
                        .map(|(p, w)| DVec4::new(p.x * w, p.y * w, p.z * w, *w))
                        .collect())
                .collect();

            let surf = NURBSSurface::new(
                bspline.u_closed.0 != Some(true),
                bspline.v_closed.0 != Some(true),
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

fn control_points_1d(s: &StepFile, row: &Vec<CartesianPoint>) -> Result<Vec<DVec3>, Error> {
    row.iter().map(|p| cartesian_point(s, *p)).collect()
}

fn control_points_2d(s: &StepFile, rows: &Vec<Vec<CartesianPoint>>) -> Result<Vec<Vec<DVec3>>, Error> {
    rows.iter()
        .map(|row| control_points_1d(s, row))
        .collect()
}

fn face_bound(s: &StepFile, b: FaceBound) -> Result<(Vec<DVec3>, usize), Error> {
    let (bound, orientation) = match &s[b] {
        Entity::FaceBound(b) => (b.bound, b.orientation),
        Entity::FaceOuterBound(b) => (b.bound, b.orientation),
        _ => return Err(Error::InvalidStepEntity("FaceBound")),
    };
    match &s[bound] {
        Entity::EdgeLoop(e) => {
            let mut d = edge_loop(s, &e.edge_list)?;
            if !orientation {
                d.reverse()
            }
            Ok((d, e.edge_list.len()))
        },
        Entity::VertexLoop(v) => {
            // This is an "edge loop" with a single vertex, which is
            // used for cones and not really anything else.
            Ok((vec![vertex_point(s, v.loop_vertex)?], 0))
        }
        _ => Err(Error::InvalidStepEntity("FaceBound.bound")),
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
        let edge = s.entity(*e).ok_or(Error::InvalidStepEntity("OrientedEdge"))?;
        let o = edge_curve(s, edge.edge_element.cast(), edge.orientation)?;
        out.extend(o.into_iter());
    }
    Ok(out)
}

fn edge_curve(s: &StepFile, e: EdgeCurve, orientation: bool) -> Result<Vec<DVec3>, Error> {
    let edge_curve = s.entity(e).ok_or(Error::InvalidStepEntity("EdgeCurve"))?;
    let curve = curve(s, edge_curve, edge_curve.edge_geometry, orientation)?;

    let (start, end) = if orientation {
        (edge_curve.edge_start, edge_curve.edge_end)
    } else {
        (edge_curve.edge_end, edge_curve.edge_start)
    };
    let is_loop = edge_curve.edge_start == edge_curve.edge_end;
    let u = vertex_point(s, start)?;
    let v = vertex_point(s, end)?;
    curve.build(u, v, is_loop)
}

fn curve(s: &StepFile, edge_curve: &ap214::EdgeCurve_,
         curve_id: ap214::Curve, orientation: bool) -> Result<Curve, Error>
{
    Ok(match &s[curve_id] {
        Entity::Circle(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast())?;
            Curve::new_circle(location, axis, ref_direction, c.radius.0.0.0,
                              edge_curve.edge_start == edge_curve.edge_end,
                              edge_curve.same_sense ^ !orientation)?
        },
        Entity::Ellipse(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast())?;
            Curve::new_ellipse(location, axis, ref_direction,
                               c.semi_axis_1.0.0.0, c.semi_axis_2.0.0.0,
                               edge_curve.edge_start == edge_curve.edge_end,
                               edge_curve.same_sense ^ !orientation)?
        },
        Entity::Hyperbola(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast())?;
            Curve::new_hyperbola(location, axis, ref_direction,
                                 c.semi_axis.0.0.0, c.semi_imag_axis.0.0.0)?
        },
        Entity::Parabola(c) => {
            let (location, axis, ref_direction) = axis2_placement_3d(s, c.position.cast())?;
            Curve::new_parabola(location, axis, ref_direction, c.focal_dist.0)?
        },
        Entity::BSplineCurveWithKnots(c) => {
            if c.self_intersect.0 == Some(true) {
                return Err(Error::SelfIntersectingCurve);
            }

            let control_points_list = control_points_1d(s, &c.control_points_list)?;

            let knots: Vec<f64> = c.knots.iter().map(|k| k.0).collect();
            let multiplicities: Vec<usize> = c.knot_multiplicities.iter()
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative curve multiplicity")))
                .collect::<Result<_, _>>()?;
            let knot_vec = KnotVector::from_multiplicities(
                c.degree.try_into().map_err(|_| Error::NumericConversion("negative curve degree"))?,
                &knots, &multiplicities);

            let open = c.closed_curve.0 != Some(true);
            let curve = nurbs::BSplineCurve::new(
                open,
                knot_vec,
                control_points_list,
            );
            Curve::BSplineCurveWithKnots {
                curve: SampledCurve::new(curve),
                dir: edge_curve.same_sense ^ !orientation,
            }
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
                .map(|&k| k.try_into().map_err(|_| Error::NumericConversion("negative curve multiplicity")))
                .collect::<Result<_, _>>()?;
            let knot_vec = KnotVector::from_multiplicities(
                bspline.degree.try_into().map_err(|_| Error::NumericConversion("negative curve degree"))?,
                &knots, &multiplicities);

            let control_points_list = control_points_1d(s, &bspline.control_points_list)?
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
            Curve::NURBSCurve {
                curve: SampledCurve::new(curve),
                dir: edge_curve.same_sense ^ !orientation,
            }
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

fn vertex_point(s: &StepFile, v: Vertex) -> Result<DVec3, Error> {
    let v = s.entity(v.cast::<VertexPoint_>())
        .ok_or(Error::InvalidStepEntity("VertexPoint"))?;
    cartesian_point(s, v.vertex_geometry.cast())
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

/// Find the lexicographically-smallest pair of constrained edges `(i, j)`
/// (i < j) that cross at interior points, along with the intersection
/// parameter `t` along edge `i`.
///
/// Uses a sweep over x-sorted edge bounding boxes so that faces with many
/// non-overlapping contours (e.g. PCB outlines with thousands of via holes)
/// avoid the full O(E²) pair scan.
fn find_first_crossing(
    pts: &[(f64, f64)],
    edges: &[(usize, usize)],
) -> Option<(usize, usize, f64)> {
    // (xmin, xmax, ymin, ymax, edge index), sorted by xmin
    let mut boxes: Vec<(f64, f64, f64, f64, usize)> = edges.iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            let (ax, ay) = pts[a];
            let (bx, by) = pts[b];
            (ax.min(bx), ax.max(bx), ay.min(by), ay.max(by), i)
        })
        .collect();
    boxes.sort_by(|p, q| p.0.partial_cmp(&q.0)
        .unwrap_or(std::cmp::Ordering::Equal));

    let mut best: Option<(usize, usize, f64)> = None;
    for bi in 0..boxes.len() {
        let (_, xmax_i, ymin_i, ymax_i, ei) = boxes[bi];
        for bj in (bi + 1)..boxes.len() {
            let (xmin_j, _, ymin_j, ymax_j, ej) = boxes[bj];
            if xmin_j > xmax_i {
                break; // sorted by xmin: no later box can overlap either
            }
            if ymin_j > ymax_i || ymax_j < ymin_i {
                continue;
            }
            // Orient the pair as (i < j) to match the original scan order
            let (i, j) = (ei.min(ej), ei.max(ej));
            if let Some((bi2, bj2, _)) = best {
                if (i, j) >= (bi2, bj2) {
                    continue;
                }
            }
            // Skip edges that share an endpoint
            if edges[i].0 == edges[j].0 || edges[i].0 == edges[j].1
            || edges[i].1 == edges[j].0 || edges[i].1 == edges[j].1 {
                continue;
            }
            if let Some((t, _s)) = segment_intersection_params(
                pts[edges[i].0], pts[edges[i].1],
                pts[edges[j].0], pts[edges[j].1],
            ) {
                best = Some((i, j, t));
            }
        }
    }
    best
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
        let (i, j, t) = match find_first_crossing(pts, edges) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nurbs::AbstractSurface;

    #[test]
    fn empty_face_tessellations_are_counted_as_errors() {
        let flat = StepFile::strip_flatten(include_bytes!("../../examples/cuboid.step")).unwrap();
        let mut step = StepFile::parse(&flat).unwrap();
        for entity in &mut step.0 {
            if let Entity::AdvancedFace(face) = entity {
                // Two identical boundaries cancel under even-odd fill. That
                // is a valid empty CDT result, not a successfully meshed face.
                face.bounds.extend_from_within(..);
            }
        }
        let (mesh, stats) = triangulate(&step);
        assert!(stats.num_faces > 0);
        assert_eq!(stats.num_errors, stats.num_faces);
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn degenerate_torus_selects_branch_and_outward_normal() {
        for (outer, v) in [(true, 0.3_f64), (false, std::f64::consts::PI + 0.3)] {
            let text = format!("ISO-10303-21;HEADER;ENDSEC;DATA;\
                #1=CARTESIAN_POINT('',(0.,0.,0.));\
                #2=DIRECTION('',(0.,0.,1.));\
                #3=DIRECTION('',(1.,0.,0.));\
                #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\
                #5=DEGENERATE_TOROIDAL_SURFACE('',#4,1.,2.,.{}.);\
                ENDSEC;END-ISO-10303-21;", if outer { "T" } else { "F" });
            let flat = StepFile::strip_flatten(text.as_bytes()).unwrap();
            let step = StepFile::parse(&flat).unwrap();
            let mut surface = get_surface(&step, Id::new(5), &[]).unwrap();
            let mut verts: Vec<_> = [0.5_f64, 0.7, 0.9].iter().map(|&u| {
                let radius = 1.0 + 2.0 * v.cos();
                mesh::Vertex {
                    pos: DVec3::new(radius * u.cos(), radius * u.sin(), 2.0 * v.sin()),
                    norm: DVec3::zeros(), color: DVec3::zeros(),
                }
            }).collect();
            let uv = surface.lower_verts(&mut verts, &[], true).unwrap();
            for (i, u) in [0.5_f64, 0.7, 0.9].iter().enumerate() {
                let expected = DVec3::new(v.cos() * u.cos(), v.cos() * u.sin(), v.sin());
                assert!(verts[i].norm.dot(&expected) > 1.0 - 1e-12);
                assert!((surface.raise(glm::DVec2::new(uv[i].0, uv[i].1)).unwrap()
                    - verts[i].pos).norm() < 1e-12);
            }
        }
    }

    fn test_curve(rational: bool) -> HomogeneousCurve {
        let points = [DVec3::new(2.0, -1.0, 0.5), DVec3::new(3.0, 1.0, 2.0)];
        let weights = if rational { [1.0, 2.0] } else { [1.0, 1.0] };
        HomogeneousCurve {
            open: true,
            knots: KnotVector::from_multiplicities(1, &[0.0, 1.0], &[2, 2]),
            control_points: points.into_iter().zip(weights).map(|(p, w)|
                DVec4::new(p.x * w, p.y * w, p.z * w, w)).collect(),
        }
    }

    fn nurbs(surface: Surface) -> SampledSurface<4> {
        match surface { Surface::NURBS(s) => s, _ => panic!("expected NURBS") }
    }

    #[test]
    fn exact_extrusion_points_and_normal() {
        let vector = DVec3::new(0.5, -1.0, 2.0);
        let boundary = [DVec3::new(-10.0, -10.0, -10.0), DVec3::new(10.0, 10.0, 10.0)];
        let surface = nurbs(extrusion_surface(test_curve(true), vector, &boundary).unwrap());
        let uv = glm::DVec2::new(0.4, 0.3);
        let basis = (DVec3::new(2.0, -1.0, 0.5) * 0.6
            + DVec3::new(3.0, 1.0, 2.0) * 0.8) / 1.4;
        assert!((surface.surf.point(uv) - (basis + vector * uv.y)).norm() < 1e-12);
        let d = surface.surf.derivs::<1>(uv);
        let expected = d[1][0].cross(&vector).normalize();
        assert!(d[1][0].cross(&d[0][1]).normalize().dot(&expected) > 1.0 - 1e-12);
    }

    #[test]
    fn exact_revolution_points_and_normal_about_skew_axis() {
        let origin = DVec3::new(-0.5, 0.25, 1.0);
        let axis = DVec3::new(1.0, 2.0, -1.0).normalize();
        let surface = nurbs(revolution_surface(test_curve(false), origin, axis).unwrap());
        let uv = glm::DVec2::new(0.125, 0.4);
        let p = DVec3::new(2.0, -1.0, 0.5) * 0.6 + DVec3::new(3.0, 1.0, 2.0) * 0.4;
        let axial = origin + axis * (p - origin).dot(&axis);
        let radial = p - axial;
        let expected = axial + radial * std::f64::consts::FRAC_1_SQRT_2
            + axis.cross(&radial) * std::f64::consts::FRAC_1_SQRT_2;
        assert!((surface.surf.point(uv) - expected).norm() < 1e-12);
        let d = surface.surf.derivs::<1>(uv);
        let normal = d[1][0].cross(&d[0][1]);
        assert!(normal.norm() > 1e-6);
        let tangent = DVec3::new(1.0, 2.0, 1.5);
        let parallel = axis * tangent.dot(&axis);
        let rotated = parallel + ((tangent - parallel) + axis.cross(&tangent))
            * std::f64::consts::FRAC_1_SQRT_2;
        let expected_normal = axis.cross(&(expected - origin)).cross(&rotated);
        assert!(normal.normalize().dot(&expected_normal.normalize()) > 1.0 - 1e-12);
        assert_eq!(surface.surf.point(glm::DVec2::new(0.0, 0.4)),
                   surface.surf.point(glm::DVec2::new(1.0, 0.4)));
    }

    #[test]
    fn counts_unsupported_face_surfaces_in_both_shell_types() {
        for shell_type in ["OPEN_SHELL", "CLOSED_SHELL"] {
            let data = format!(
                "ISO-10303-21;HEADER;ENDSEC;DATA;\
                 #1=CARTESIAN_POINT('',(0.,0.,0.));\
                 #2=DIRECTION('',(0.,0.,1.));\
                 #3=AXIS2_PLACEMENT_3D('',#1,#2,$);\
                 #4=TOROIDAL_SURFACE('',#3,2.,1.);\
                 #5=OFFSET_SURFACE('',#4,1.,.F.);\
                 #6=ADVANCED_FACE('',(),#5,.T.);\
                 #7={shell_type}('',(#6));\
                 #8=SHELL_BASED_SURFACE_MODEL('',(#7));\
                 ENDSEC;END-ISO-10303-21;"
            );
            let flat = StepFile::strip_flatten(data.as_bytes()).unwrap();
            let step = StepFile::parse(&flat).unwrap();
            let (mesh, stats) = triangulate(&step);

            assert_eq!(stats.num_shells, 1, "{shell_type}");
            assert_eq!(stats.num_faces, 1, "{shell_type}");
            assert_eq!(stats.num_errors, 1, "{shell_type}");
            assert_eq!(stats.num_panics, 0, "{shell_type}");
            assert!(mesh.triangles.is_empty(), "{}", shell_type);
        }
    }

    #[test]
    fn triangulates_face_surface_in_closed_shell() {
        let data = br#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''), '2;1');
FILE_NAME('face_surface_square.step','','',(''),(''),'','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=VERTEX_POINT('',#1);
#3=CARTESIAN_POINT('',(1.,0.,0.));
#4=VERTEX_POINT('',#3);
#5=CARTESIAN_POINT('',(1.,1.,0.));
#6=VERTEX_POINT('',#5);
#7=CARTESIAN_POINT('',(0.,1.,0.));
#8=VERTEX_POINT('',#7);
#9=DIRECTION('',(1.,0.,0.));
#10=VECTOR('',#9,1.);
#11=LINE('',#1,#10);
#12=EDGE_CURVE('',#2,#4,#11,.T.);
#13=DIRECTION('',(0.,1.,0.));
#14=VECTOR('',#13,1.);
#15=LINE('',#3,#14);
#16=EDGE_CURVE('',#4,#6,#15,.T.);
#17=DIRECTION('',(-1.,0.,0.));
#18=VECTOR('',#17,1.);
#19=LINE('',#5,#18);
#20=EDGE_CURVE('',#6,#8,#19,.T.);
#21=DIRECTION('',(0.,-1.,0.));
#22=VECTOR('',#21,1.);
#23=LINE('',#7,#22);
#24=EDGE_CURVE('',#8,#2,#23,.T.);
#25=ORIENTED_EDGE('',*,*,#12,.T.);
#26=ORIENTED_EDGE('',*,*,#16,.T.);
#27=ORIENTED_EDGE('',*,*,#20,.T.);
#28=ORIENTED_EDGE('',*,*,#24,.T.);
#29=EDGE_LOOP('',(#25,#26,#27,#28));
#30=FACE_OUTER_BOUND('',#29,.T.);
#31=DIRECTION('',(0.,0.,1.));
#32=DIRECTION('',(1.,0.,0.));
#33=AXIS2_PLACEMENT_3D('',#1,#31,#32);
#34=PLANE('',#33);
#35=FACE_SURFACE('',(#30),#34,.T.);
#36=CLOSED_SHELL('',(#35));
#37=SHELL_BASED_SURFACE_MODEL('',(#36));
ENDSEC;
END-ISO-10303-21;
"#;

        let flat = StepFile::strip_flatten(data).unwrap();
        let step = StepFile::parse(&flat).unwrap();
        let (mesh, stats) = triangulate(&step);

        assert_eq!(stats.num_faces, 1);
        assert_eq!(stats.num_errors, 0);
        assert_eq!(stats.num_panics, 0);
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.verts.len(), 4);
    }
}
