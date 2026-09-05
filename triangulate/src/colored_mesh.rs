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
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl ColoredSubmesh {
    fn new(color: [f32; 4]) -> Self {
        Self {
            color,
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn insert_vertex(
        &mut self,
        key: ColorKey,
        mesh_index: usize,
        vertex: &crate::mesh::Vertex,
        vertex_indices: &mut [Option<(ColorKey, u32)>],
    ) -> Result<u32, String> {
        if let Some((existing_key, existing_index)) = vertex_indices[mesh_index] {
            if existing_key == key {
                return Ok(existing_index);
            }
        }

        let index =
            u32::try_from(self.positions.len()).map_err(|_| "too many vertices for u32 index")?;
        self.positions.push([
            vertex.pos.x as f32,
            vertex.pos.y as f32,
            vertex.pos.z as f32,
        ]);
        self.normals.push([
            vertex.norm.x as f32,
            vertex.norm.y as f32,
            vertex.norm.z as f32,
        ]);
        if vertex_indices[mesh_index].is_none() {
            vertex_indices[mesh_index] = Some((key, index));
        }
        Ok(index)
    }
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
    let mut vertex_indices = vec![None; mesh.verts.len()];

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
        let bucket = buckets
            .entry(key)
            .or_insert_with(|| ColoredSubmesh::new(key.to_rgba()));
        let a = bucket.insert_vertex(key, ia, va, &mut vertex_indices)?;
        let b = bucket.insert_vertex(key, ib, vb, &mut vertex_indices)?;
        let c = bucket.insert_vertex(key, ic, vc, &mut vertex_indices)?;
        bucket.indices.extend([a, b, c]);
    }

    let mut submeshes = buckets.into_values().collect::<Vec<_>>();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Triangle, Vertex};
    use nalgebra_glm::{DVec3, U32Vec3};

    fn vertex(pos: DVec3, norm: DVec3) -> Vertex {
        Vertex {
            pos,
            norm,
            color: DVec3::new(0.25, 0.5, 0.75),
        }
    }

    #[test]
    fn preserves_normals_and_reuses_mesh_vertices() {
        let mesh = Mesh {
            verts: vec![
                vertex(DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(1.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
            ],
            triangles: vec![
                Triangle {
                    verts: U32Vec3::new(0, 1, 2),
                },
                Triangle {
                    verts: U32Vec3::new(0, 2, 3),
                },
            ],
        };

        let tess = group_mesh_by_color(&mesh).unwrap();
        let submesh = &tess.submeshes[0];
        assert_eq!(submesh.positions.len(), 4);
        assert_eq!(submesh.normals, vec![[0.0, 0.0, 1.0]; 4]);
        assert_eq!(submesh.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn groups_shared_vertices_by_color() {
        let mut mesh = Mesh {
            verts: (0..5)
                .map(|i| vertex(DVec3::new(i as f64, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)))
                .collect(),
            triangles: vec![],
        };
        // Unreferenced vertices must not produce empty color groups.
        assert!(group_mesh_by_color(&mesh).unwrap().submeshes.is_empty());
        for (i, v) in mesh.verts.iter_mut().enumerate() {
            v.color = if i < 3 { DVec3::x() } else { DVec3::y() };
        }
        mesh.triangles = vec![
            Triangle {
                verts: U32Vec3::new(0, 1, 2),
            },
            Triangle {
                verts: U32Vec3::new(0, 3, 4),
            },
            Triangle {
                verts: U32Vec3::new(0, 2, 1),
            },
        ];

        let tess = group_mesh_by_color(&mesh).unwrap();
        assert_eq!(tess.submeshes.len(), 2);
        let red = &tess.submeshes[0];
        assert_eq!(red.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            red.positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        );
        assert_eq!(red.normals, [[0.0, 0.0, 1.0]; 3]);
        assert_eq!(red.indices, [0, 1, 2, 0, 2, 1]);
        let green = &tess.submeshes[1];
        assert_eq!(green.color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            green.positions,
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        );
        assert_eq!(green.normals, [[0.0, 0.0, 1.0]; 3]);
        assert_eq!(green.indices, [0, 1, 2]);
    }

    #[test]
    fn keeps_coincident_vertices_separate_across_hard_edges() {
        let mesh = Mesh {
            verts: vec![
                vertex(DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 1.0)),
                vertex(DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
                vertex(DVec3::new(0.0, 1.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
                vertex(DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 0.0)),
            ],
            triangles: vec![
                Triangle {
                    verts: U32Vec3::new(0, 1, 2),
                },
                Triangle {
                    verts: U32Vec3::new(3, 4, 5),
                },
            ],
        };

        let tess = group_mesh_by_color(&mesh).unwrap();
        let submesh = &tess.submeshes[0];
        assert_eq!(submesh.positions.len(), 6);
        assert_eq!(submesh.normals[..3], [[0.0, 0.0, 1.0]; 3]);
        assert_eq!(submesh.normals[3..], [[1.0, 0.0, 0.0]; 3]);
    }
}
