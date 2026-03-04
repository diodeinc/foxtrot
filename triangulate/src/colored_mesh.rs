//! Color-grouped mesh output from STEP tessellation.
//!
//! Takes the flat `Mesh` produced by [`triangulate()`](crate::triangulate::triangulate)
//! and re-buckets triangles by quantised vertex colour, producing a
//! `TessellatedMesh` of per-colour `ColoredSubmesh` slices ready for
//! instanced GPU rendering.

use std::collections::HashMap;
use std::convert::TryFrom;

use crate::mesh::Mesh;
use crate::stats::Stats;
use crate::triangulate::triangulate;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A collection of per-colour sub-meshes produced from a STEP model.
#[derive(Debug, Clone)]
pub struct TessellatedMesh {
    pub submeshes: Vec<ColoredSubmesh>,
}

/// One colour group: all triangles sharing the same quantised RGBA colour.
#[derive(Debug, Clone)]
pub struct ColoredSubmesh {
    pub color: [f32; 4],
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

/// Lightweight statistics from tessellation.
#[derive(Debug, Clone)]
pub struct TessellationDiagnostics {
    pub num_shells: usize,
    pub num_faces: usize,
    pub num_errors: usize,
    pub num_panics: usize,
}

impl From<&Stats> for TessellationDiagnostics {
    fn from(s: &Stats) -> Self {
        Self {
            num_shells: s.num_shells,
            num_faces: s.num_faces,
            num_errors: s.num_errors,
            num_panics: s.num_panics,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse and tessellate raw STEP bytes, returning colour-grouped sub-meshes.
///
/// This is the main entry point for downstream consumers that want
/// colour-bucketed geometry without pulling in the `step` crate directly.
pub fn tessellate_step_bytes(
    step_bytes: &[u8],
) -> Result<(TessellatedMesh, TessellationDiagnostics), String> {
    let flattened = step::step_file::StepFile::strip_flatten(step_bytes);
    let step = step::step_file::StepFile::parse(&flattened);
    let (mesh, stats) = triangulate(&step);
    let diag = TessellationDiagnostics::from(&stats);
    let tess = group_mesh_by_color(&mesh)?;
    Ok((tess, diag))
}

/// Group an already-triangulated `Mesh` into per-colour sub-meshes.
pub fn group_mesh_by_color(mesh: &Mesh) -> Result<TessellatedMesh, String> {
    let mut buckets: HashMap<ColorKey, ColoredSubmesh> = HashMap::new();

    for tri in mesh.triangles.iter() {
        let ia = tri.verts.x as usize;
        let ib = tri.verts.y as usize;
        let ic = tri.verts.z as usize;
        let va = mesh
            .verts
            .get(ia)
            .ok_or_else(|| format!("triangle index out of range: {ia}"))?;
        let vb = mesh
            .verts
            .get(ib)
            .ok_or_else(|| format!("triangle index out of range: {ib}"))?;
        let vc = mesh
            .verts
            .get(ic)
            .ok_or_else(|| format!("triangle index out of range: {ic}"))?;

        let key = triangle_color_key(va, vb, vc);
        let sub = buckets.entry(key).or_insert_with(|| ColoredSubmesh {
            color: key.to_rgba(),
            positions: Vec::new(),
            indices: Vec::new(),
        });

        let base =
            u32::try_from(sub.positions.len()).map_err(|_| "too many vertices for u32 index")?;
        sub.positions
            .push([va.pos.x as f32, va.pos.y as f32, va.pos.z as f32]);
        sub.positions
            .push([vb.pos.x as f32, vb.pos.y as f32, vb.pos.z as f32]);
        sub.positions
            .push([vc.pos.x as f32, vc.pos.y as f32, vc.pos.z as f32]);
        sub.indices.extend([base, base + 1, base + 2]);
    }

    let mut submeshes = buckets
        .into_values()
        .filter(|s| !s.positions.is_empty() && !s.indices.is_empty())
        .collect::<Vec<_>>();
    // Largest groups first for deterministic ordering.
    submeshes.sort_by(|a, b| a.indices.len().cmp(&b.indices.len()).reverse());
    Ok(TessellatedMesh { submeshes })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Quantised RGBA colour key for bucketing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ColorKey(u8, u8, u8, u8);

impl ColorKey {
    fn from_rgb(color: [f32; 3]) -> Self {
        let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
        Self(to_u8(color[0]), to_u8(color[1]), to_u8(color[2]), 255)
    }

    fn to_rgba(self) -> [f32; 4] {
        [
            self.0 as f32 / 255.0,
            self.1 as f32 / 255.0,
            self.2 as f32 / 255.0,
            self.3 as f32 / 255.0,
        ]
    }
}

/// Pick a dominant quantised colour for a triangle from its three vertices.
///
/// Colours should generally be uniform per STEP face, but we choose a
/// majority-vote quantised vertex colour rather than an average to avoid
/// synthetic blended bucket colours.
fn triangle_color_key(
    va: &crate::mesh::Vertex,
    vb: &crate::mesh::Vertex,
    vc: &crate::mesh::Vertex,
) -> ColorKey {
    let ka = ColorKey::from_rgb([va.color.x as f32, va.color.y as f32, va.color.z as f32]);
    let kb = ColorKey::from_rgb([vb.color.x as f32, vb.color.y as f32, vb.color.z as f32]);
    let kc = ColorKey::from_rgb([vc.color.x as f32, vc.color.y as f32, vc.color.z as f32]);
    if ka == kb || ka == kc {
        ka
    } else if kb == kc {
        kb
    } else {
        ka
    }
}
