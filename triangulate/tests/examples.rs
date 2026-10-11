use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

#[test]
fn checked_in_models_tessellate_without_errors() {
    for name in &["abstract_pca.step", "cube_hole.step", "cuboid.step"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../examples")
            .join(name);
        let data = std::fs::read(path).expect("Could not read example model");
        let flat = StepFile::strip_flatten(&data).unwrap();
        let step = StepFile::parse(&flat).unwrap();
        let (mesh, stats) = triangulate(&step);

        assert!(!mesh.triangles.is_empty(), "{} produced no triangles", name);
        assert!(stats.num_faces > 0, "{} contained no faces", name);
        assert_eq!(stats.num_errors(), 0, "{} had tessellation errors", name);
        assert_eq!(stats.num_panics(), 0, "{} had tessellation panics", name);

        // The models are closed solids. Faces that share an edge share its
        // points, so every edge borders exactly two triangles.
        let mut ids = std::collections::HashMap::new();
        let id: Vec<usize> = mesh
            .verts
            .iter()
            .map(|v| {
                let next = ids.len();
                *ids.entry([v.pos.x, v.pos.y, v.pos.z].map(f64::to_bits)).or_insert(next)
            })
            .collect();
        let mut uses = std::collections::HashMap::new();
        for t in &mesh.triangles {
            let [a, b, c] = [t.verts.x, t.verts.y, t.verts.z].map(|i| id[i as usize]);
            for (p, q) in [(a, b), (b, c), (c, a)] {
                *uses.entry((p.min(q), p.max(q))).or_insert(0) += 1;
            }
        }
        let open = uses.values().filter(|&&n| n != 2).count();
        assert_eq!(open, 0, "{} has {} edges without exactly two triangles", name, open);
    }
}
