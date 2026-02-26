use step::step_file::StepFile;
use triangulate::triangulate::triangulate;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];

    let data = std::fs::read(path).expect("Could not open file");
    let flat = StepFile::strip_flatten(&data);
    let step = StepFile::parse(&flat);
    let (mesh, stats) = triangulate(&step);

    eprintln!("Faces: {}, Errors: {}, Panics: {}, Triangles: {}, Vertices: {}",
        stats.num_faces, stats.num_errors, stats.num_panics,
        mesh.triangles.len(), mesh.verts.len());

    if !mesh.verts.is_empty() {
        let mut min = mesh.verts[0].pos;
        let mut max = mesh.verts[0].pos;
        for v in &mesh.verts {
            min = nalgebra_glm::min2(&min, &v.pos);
            max = nalgebra_glm::max2(&max, &v.pos);
        }
        let size = max - min;
        eprintln!("BBox: ({:.6}, {:.6}, {:.6}) to ({:.6}, {:.6}, {:.6})",
                  min.x, min.y, min.z, max.x, max.y, max.z);
        eprintln!("Size: {:.6} x {:.6} x {:.6} (max dim: {:.6})",
                  size.x, size.y, size.z, size.x.max(size.y).max(size.z));
    }

    if std::env::args().any(|a| a == "--stl") {
        let out = args.get(2).map(|s| s.as_str()).unwrap_or("out.stl");
        mesh.save_stl(out).expect("Could not save STL");
        eprintln!("Saved to {}", out);
    }
}
