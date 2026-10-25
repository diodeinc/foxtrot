//! Color-grouped mesh output from STEP tessellation.
//!
//! Takes the flat `Mesh` produced by [`triangulate()`](crate::triangulate::triangulate)
//! and re-buckets triangles by quantised vertex colour and surface material,
//! producing a `TessellatedMesh` of `ColoredSubmesh` slices ready for
//! instanced GPU rendering.

use std::collections::HashMap;
use std::convert::TryFrom;
use crate::mesh::Mesh;
use crate::stats::Stats;
use crate::triangulate::triangulate_with_materials;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A collection of per-colour sub-meshes produced from a STEP model.
#[derive(Debug, Clone)]
pub struct TessellatedMesh {
    pub submeshes: Vec<ColoredSubmesh>,
}

/// Authored STEP rendering properties. `None` means unspecified, not zero.
///
/// Ambient reflectance and the shading/interpolation method have no direct
/// material equivalent here. Apply diffuse strength after converting `color`
/// RGB to the consumer's linear lighting space, not before that conversion.
/// No metal classification or conversion to a PBR lighting model is performed.
/// RGB has no source color-space metadata. Finite normalized properties are
/// clamped to [0, 1]; invalid/nonfinite properties remain unspecified.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceMaterial {
    /// One minus STEP transparency, in [0, 1]. Also supplies `color[3]`.
    pub opacity: Option<f32>,
    /// STEP diffuse reflectance in [0, 1], independent of surface RGB.
    pub diffuse_strength: Option<f32>,
    /// Authored RGB highlight color in [0, 1], independent of its strength.
    pub specular_color: Option<[f32; 3]>,
    /// STEP specular reflectance in [0, 1], not a Fresnel coefficient.
    pub specular_strength: Option<f32>,
    /// Raw nonnegative STEP specular exponent, not normalized to [0, 1].
    pub shininess: Option<f32>,
}

/// All triangles sharing quantised RGB and the same supported material values.
#[derive(Debug, Clone)]
pub struct ColoredSubmesh {
    pub color: [f32; 4],
    pub material: SurfaceMaterial,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl ColoredSubmesh {
    fn new(mut color: [f32; 4], material: SurfaceMaterial) -> Self {
        color[3] = material.opacity.unwrap_or(1.0);
        Self {
            color,
            material,
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn insert_vertex(
        &mut self,
        key: usize,
        mesh_index: usize,
        vertex: &crate::mesh::Vertex,
        vertex_indices: &mut [Option<(usize, u32)>],
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse and tessellate raw STEP bytes, returning colour-grouped sub-meshes.
///
/// This is the main entry point for downstream consumers that want
/// colour-bucketed geometry without pulling in the `step` crate directly.
/// A face style replaces its shell's, and a shell style its solid's, including
/// unspecified properties. An over-riding style beats the style it overrides.
/// The first supported surface usage wins; rendering RGB takes precedence over
/// fill RGB. Separate front/back styles and context-dependent overrides are not
/// represented. RGB retains the existing 8-bit quantization; other properties
/// retain f32 precision and participate in grouping, including their absence.
pub fn tessellate_step_bytes(
    step_bytes: &[u8],
) -> Result<(TessellatedMesh, Stats), String> {
    let flattened = step::step_file::StepFile::strip_flatten(step_bytes)
        .map_err(|e| e.to_string())?;
    let step = step::step_file::StepFile::parse(&flattened)
        .map_err(|e| e.to_string())?;
    let (mesh, stats, materials) = triangulate_with_materials(&step);
    let tess = group_mesh(&mesh, materials.into_iter().flat_map(|(count, material)| {
        std::iter::repeat(material).take(count)
    }))?;
    Ok((tess, stats))
}

/// Group an already-triangulated RGB-only `Mesh` with unspecified materials.
pub fn group_mesh_by_color(mesh: &Mesh) -> Result<TessellatedMesh, String> {
    group_mesh(mesh, std::iter::repeat(SurfaceMaterial::default()))
}

fn group_mesh(mesh: &Mesh, mut materials: impl Iterator<Item = SurfaceMaterial>) -> Result<TessellatedMesh, String> {
    let mut buckets: HashMap<MaterialKey, (usize, ColoredSubmesh)> = HashMap::new();
    let mut vertex_indices = vec![None; mesh.verts.len()];

    for tri in &mesh.triangles {
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

        let color = triangle_color_key(va, vb, vc);
        let material = materials.next().expect("material for each triangle");
        let key = MaterialKey::new(color, material);
        let next_bucket = buckets.len();
        let (bucket_id, bucket) = buckets
            .entry(key)
            .or_insert_with(|| (next_bucket, ColoredSubmesh::new(color.to_rgba(), material)));
        let a = bucket.insert_vertex(*bucket_id, ia, va, &mut vertex_indices)?;
        let b = bucket.insert_vertex(*bucket_id, ib, vb, &mut vertex_indices)?;
        let c = bucket.insert_vertex(*bucket_id, ic, vc, &mut vertex_indices)?;
        bucket.indices.extend([a, b, c]);
    }

    let mut submeshes = buckets.into_iter().collect::<Vec<_>>();
    // Largest groups first. Color and material break ties deterministically.
    submeshes.sort_by_key(|(key, (_, submesh))| (std::cmp::Reverse(submesh.indices.len()), *key));
    Ok(TessellatedMesh {
        submeshes: submeshes.into_iter().map(|(_, (_, submesh))| submesh).collect(),
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MaterialKey(ColorKey, [Option<u32>; 7]);

impl MaterialKey {
    fn new(color: ColorKey, material: SurfaceMaterial) -> Self {
        let specular = material.specular_color;
        Self(color, [
            material.opacity,
            material.diffuse_strength,
            specular.map(|c| c[0]),
            specular.map(|c| c[1]),
            specular.map(|c| c[2]),
            material.specular_strength,
            material.shininess,
        ].map(|v| v.map(|v| if v == 0.0 { 0 } else { v.to_bits() })))
    }
}

/// Quantised RGBA colour key for bucketing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    fn orders_equal_sized_color_groups_the_same_every_time() {
        let mut mesh = Mesh {
            verts: (0..6)
                .map(|i| vertex(DVec3::new(i as f64, (i % 3) as f64, 0.0), DVec3::new(0.0, 0.0, 1.0)))
                .collect(),
            triangles: vec![
                Triangle { verts: U32Vec3::new(0, 1, 2) },
                Triangle { verts: U32Vec3::new(3, 4, 5) },
            ],
        };
        for (i, v) in mesh.verts.iter_mut().enumerate() {
            v.color = if i < 3 { DVec3::x() } else { DVec3::y() };
        }
        // Each call buckets colours in a new map, with new random keys.
        let colors = || group_mesh_by_color(&mesh).unwrap().submeshes.iter().map(|s| s.color).collect::<Vec<_>>();
        let first = colors();
        assert_eq!(first.len(), 2);
        for _ in 0..16 {
            assert_eq!(colors(), first);
        }
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
