use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

#[test]
fn checked_in_models_tessellate_without_errors() {
    for name in &["abstract_pca.step", "cube_hole.step", "cuboid.step"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../examples")
            .join(name);
        let data = std::fs::read(path).expect("Could not read example model");
        let flat = StepFile::strip_flatten(&data);
        let step = StepFile::parse(&flat);
        let (mesh, stats) = triangulate(&step);

        assert!(!mesh.triangles.is_empty(), "{} produced no triangles", name);
        assert!(stats.num_faces > 0, "{} contained no faces", name);
        assert_eq!(stats.num_errors, 0, "{} had tessellation errors", name);
        assert_eq!(stats.num_panics, 0, "{} had tessellation panics", name);
    }
}
