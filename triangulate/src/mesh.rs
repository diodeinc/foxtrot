use std::convert::TryInto;
use nalgebra_glm::{DVec3, U32Vec3};

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: DVec3,
    pub norm: DVec3,
    pub color: DVec3,
}
#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub verts: U32Vec3,
}

#[derive(Default)]
pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    /// Browser triangle soup: position, normal, color. Center referenced
    /// geometry in f64 before the only f32 conversion; longest extent is 200.
    pub fn to_triangle_buffer(&self) -> Vec<f32> {
        let positions = || self.triangles.iter().flat_map(|t| t.verts.iter())
            .map(|&i| self.verts[i as usize].pos);
        let mut min = DVec3::repeat(f64::INFINITY);
        let mut max = DVec3::repeat(f64::NEG_INFINITY);
        for p in positions() {
            min = min.inf(&p);
            max = max.sup(&p);
        }
        let extent = max - min;
        let center = min + extent * 0.5;
        let scale = extent.max();
        self.triangles.iter().flat_map(|t| t.verts.iter()).flat_map(|&i| {
            let v = self.verts[i as usize];
            let p = (v.pos - center) / scale * 200.;
            [p.x, p.y, p.z, v.norm.x, v.norm.y, v.norm.z, v.color.x, v.color.y, v.color.z]
                .map(|x| x as f32)
        }).collect()
    }

    // Combine two triangulations with an associative binary operator
    // (why yes, this _is_ a monoid)
    pub fn combine(mut a: Self, b: Self) -> Self {
        let dv = a.verts.len().try_into().expect("too many triangles");
        a.verts.extend(b.verts);
        a.triangles.extend(b.triangles.into_iter()
            .map(|t| Triangle { verts: t.verts.add_scalar(dv) }));
        a
    }

    /// Writes the triangulation to a STL, for debugging
    pub fn save_stl(&self, filename: &str) -> std::io::Result<()> {
        let mut out: Vec<u8> = Vec::new();
        for _ in 0..80 { // header
            out.push('x' as u8);
        }
        let u: u32 = self.triangles.len().try_into()
            .expect("Too many triangles");
        out.extend(&u.to_le_bytes());
        for t in self.triangles.iter() {
            out.extend(std::iter::repeat(0).take(12)); // normal
            for v in t.verts.iter() {
                let v = self.verts[*v as usize];
                out.extend(&(v.pos.x as f32).to_le_bytes());
                out.extend(&(v.pos.y as f32).to_le_bytes());
                out.extend(&(v.pos.z as f32).to_le_bytes());
            }
            out.extend(std::iter::repeat(0).take(2)); // attributes
        }
        std::fs::write(filename, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_buffer_centers_all_axes_before_rounding() {
        let origin = DVec3::new(1e9, -2e9, 3e9);
        let mesh = Mesh {
            verts: [DVec3::zeros(), DVec3::new(2., 0., 0.), DVec3::new(0., 0., 4.)]
                .iter().map(|p| Vertex { pos: origin + p, norm: DVec3::new(0., -1., 0.),
                    color: DVec3::new(1., 0.5, 0.) }).collect(),
            triangles: vec![Triangle { verts: U32Vec3::new(0, 1, 2) }],
        };
        let buffer = mesh.to_triangle_buffer();
        assert_eq!(&buffer[0..9], &[-50., 0., -100., 0., -1., 0., 1., 0.5, 0.]);
        assert_eq!(&buffer[9..12], &[50., 0., -100.]);
        assert_eq!(&buffer[18..21], &[-50., 0., 100.]);
        assert!(Mesh::default().to_triangle_buffer().is_empty());
    }
}
