use step::step_file::StepFile;
use triangulate::colored_mesh::{tessellate_step_bytes, SurfaceMaterial, TessellatedMesh};
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

const CUBOID: &str = include_str!("../../examples/cuboid.step");

fn tessellate(text: &str) -> TessellatedMesh {
    let (mesh, stats) = tessellate_step_bytes(text.as_bytes()).unwrap();
    assert!(stats.is_complete(), "{:?}", stats.failures);
    assert_eq!(stats.num_faces, 6);
    mesh
}

fn add_entities(text: &str, entities: &str) -> String {
    let end = text.rfind("ENDSEC;").unwrap();
    format!("{}\n{}\n{}", &text[..end], entities, &text[end..])
}

fn geometry(mesh: &TessellatedMesh) -> Vec<[[u32; 6]; 3]> {
    let mut triangles = Vec::new();
    for sub in &mesh.submeshes {
        for tri in sub.indices.chunks_exact(3) {
            triangles.push(std::array::from_fn(|i| {
                let p = sub.positions[tri[i] as usize];
                let n = sub.normals[tri[i] as usize];
                [p[0], p[1], p[2], n[0], n[1], n[2]].map(f32::to_bits)
            }));
        }
    }
    triangles.sort();
    triangles
}

#[test]
fn color_only_and_mixed_styles_keep_defaults_and_geometry() {
    let plain = tessellate(CUBOID);
    assert_eq!(plain.submeshes.len(), 1);
    assert_eq!(plain.submeshes[0].color, [157. / 255., 207. / 255., 237. / 255., 1.]);
    assert_eq!(plain.submeshes[0].material, SurfaceMaterial::default());
    assert_eq!(plain.submeshes[0].indices.len(), 36);

    let mixed = CUBOID
        .replace("(#114),#121", "(#204,#114),#121")
        .replace("#116=SURFACE_SIDE_STYLE('',(#117));", "#116=SURFACE_SIDE_STYLE('',(#202,#117));")
        .replace("#118=FILL_AREA_STYLE('',(#119));", "#118=FILL_AREA_STYLE('',(#119,#203));");
    let mixed = tessellate(&add_entities(&mixed, "
        #200=CURVE_STYLE('',#201,POSITIVE_LENGTH_MEASURE(0.1),#120);
        #201=DRAUGHTING_PRE_DEFINED_CURVE_FONT('continuous');
        #202=SURFACE_STYLE_BOUNDARY(#200);
        #203=FILL_AREA_STYLE_COLOUR('another fill',#120);
        #204=PRESENTATION_STYLE_ASSIGNMENT((#200));
    "));
    assert_eq!(mixed.submeshes.len(), 1);
    assert_eq!(mixed.submeshes[0].color, plain.submeshes[0].color);
    assert_eq!(mixed.submeshes[0].material, SurfaceMaterial::default());
    assert_eq!(geometry(&mixed), geometry(&plain));

    let unstyled = tessellate(&CUBOID.replace(
        "#184=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#113),\n#185);", ""));
    assert_eq!(unstyled.submeshes[0].color, [128. / 255., 128. / 255., 128. / 255., 1.]);
    assert_eq!(unstyled.submeshes[0].material, SurfaceMaterial::default());
    assert_eq!(geometry(&unstyled), geometry(&plain));
}

#[test]
fn rendering_properties_preserve_authored_values_without_changing_geometry() {
    let styled = CUBOID.replace("#116=SURFACE_SIDE_STYLE('',(#117));",
        "#116=SURFACE_SIDE_STYLE('',(#117,#210));");
    let styled = add_entities(&styled, "
        #210=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#213,(#211,#212));
        #211=SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,0.5,0.35,256.,#214);
        #212=SURFACE_STYLE_TRANSPARENT(0.3);
        #213=COLOUR_RGB('',0.8,0.4,0.2);
        #214=COLOUR_RGB('',0.9,0.6,0.3);
    ");
    let mesh = tessellate(&styled);
    assert_eq!(mesh.submeshes.len(), 1);
    let sub = &mesh.submeshes[0];
    assert_eq!(sub.color, [0.8, 0.4, 0.2, 0.7]);
    assert_eq!(sub.material, SurfaceMaterial {
        opacity: Some(0.7),
        diffuse_strength: Some(0.5),
        specular_color: Some([0.9, 0.6, 0.3]),
        specular_strength: Some(0.35),
        shininess: Some(256.),
    });
    assert_eq!(geometry(&mesh), geometry(&tessellate(CUBOID)));

    let rendering_only = styled.replace("(#117,#210)", "(#210)");
    assert_eq!(tessellate(&rendering_only).submeshes[0].material, sub.material);
    let diffuse_only = styled.replace(
        "SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,0.5,0.35,256.,#214)",
        "SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE(0.1,0.5)");
    let diffuse = tessellate(&diffuse_only);
    assert_eq!(diffuse.submeshes[0].color, sub.color);
    assert_eq!(diffuse.submeshes[0].material, SurfaceMaterial {
        opacity: Some(0.7), diffuse_strength: Some(0.5), ..SurfaceMaterial::default()
    });
    let simple = styled.replace(
        "SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#213,(#211,#212))",
        "SURFACE_STYLE_RENDERING(.NORMAL_SHADING.,#213)");
    let simple = tessellate(&simple);
    assert_eq!(simple.submeshes[0].color, [0.8, 0.4, 0.2, 1.]);
    assert_eq!(simple.submeshes[0].material, SurfaceMaterial::default());
    for (transparency, opacity) in [("0.", 1.), ("1.", 0.)] {
        let boundary = styled.replace("0.35,256.", "0.,0.")
            .replace("TRANSPARENT(0.3)", &format!("TRANSPARENT({})", transparency));
        let boundary = tessellate(&boundary);
        assert_eq!(boundary.submeshes[0].color[3], opacity);
        assert_eq!(boundary.submeshes[0].material, SurfaceMaterial {
            opacity: Some(opacity), specular_strength: Some(0.), shininess: Some(0.), ..sub.material
        });
    }
}

#[test]
fn same_rgb_faces_retain_distinct_materials_and_override_the_solid() {
    let styled = CUBOID
        .replace("#116=SURFACE_SIDE_STYLE('',(#117));", "#116=SURFACE_SIDE_STYLE('',(#210));")
        .replace("(#113),", "(#300,#310,#320,#330,#340,#113),");
    let styled = add_entities(&styled, "
        #210=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#120,(#211,#212));
        #211=SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,1.,0.35,96.,#213);
        #212=SURFACE_STYLE_TRANSPARENT(0.2);
        #213=COLOUR_RGB('',0.9,0.6,0.3);
        #300=STYLED_ITEM('',(#301),#106);
        #301=PRESENTATION_STYLE_ASSIGNMENT((#302));
        #302=SURFACE_STYLE_USAGE(.BOTH.,#303);
        #303=SURFACE_SIDE_STYLE('',(#117));
        #310=STYLED_ITEM('',(#311),#107);
        #311=PRESENTATION_STYLE_ASSIGNMENT((#312));
        #312=SURFACE_STYLE_USAGE(.BOTH.,#313);
        #313=SURFACE_SIDE_STYLE('',(#314));
        #314=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#120,(#211,#315));
        #315=SURFACE_STYLE_TRANSPARENT(0.5);
        #320=STYLED_ITEM('',(#321),#108);
        #321=PRESENTATION_STYLE_ASSIGNMENT((#322));
        #322=SURFACE_STYLE_USAGE(.BOTH.,#323);
        #323=SURFACE_SIDE_STYLE('',(#324));
        #324=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#120,(#325,#212));
        #325=SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,1.,0.7,96.,#213);
        #330=STYLED_ITEM('',(#331),#109);
        #331=PRESENTATION_STYLE_ASSIGNMENT((#332));
        #332=SURFACE_STYLE_USAGE(.BOTH.,#333);
        #333=SURFACE_SIDE_STYLE('',(#334));
        #334=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#120,(#335,#212));
        #335=SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,1.,0.35,96.,#336);
        #336=COLOUR_RGB('',0.2,0.5,0.8);
        #340=STYLED_ITEM('',(#341),#110);
        #341=PRESENTATION_STYLE_ASSIGNMENT((#342));
        #342=SURFACE_STYLE_USAGE(.BOTH.,#343);
        #343=SURFACE_SIDE_STYLE('',(#344));
        #344=SURFACE_STYLE_RENDERING_WITH_PROPERTIES(.NORMAL_SHADING.,#120,(#345,#212));
        #345=SURFACE_STYLE_REFLECTANCE_AMBIENT_DIFFUSE_SPECULAR(0.1,1.,0.35,32.,#213);
    ");
    let mesh = tessellate(&styled);
    assert_eq!(mesh.submeshes.len(), 6);
    let base = SurfaceMaterial { opacity: Some(0.8), specular_color: Some([0.9, 0.6, 0.3]),
        diffuse_strength: Some(1.), specular_strength: Some(0.35), shininess: Some(96.) };
    for (expected, normal) in [
        (SurfaceMaterial::default(), [-1., 0., 0.]),
        (base, [0., 0., -1.]),
        (SurfaceMaterial { opacity: Some(0.5), ..base }, [0., -1., 0.]),
        (SurfaceMaterial { specular_strength: Some(0.7), ..base }, [1., 0., 0.]),
        (SurfaceMaterial { specular_color: Some([0.2, 0.5, 0.8]), ..base }, [0., 1., 0.]),
        (SurfaceMaterial { shininess: Some(32.), ..base }, [0., 0., 1.]),
    ] {
        let sub = mesh.submeshes.iter().find(|s| s.material == expected).unwrap();
        assert_eq!(sub.indices.len(), 6);
        assert!(sub.normals.iter().all(|&n| n == normal));
        assert_eq!(&sub.color[..3], &[157. / 255., 207. / 255., 237. / 255.]);
        assert_eq!(sub.color[3], expected.opacity.unwrap_or(1.));
    }
    assert_eq!(geometry(&mesh), geometry(&tessellate(CUBOID)));
    assert_eq!(mesh.submeshes.iter().map(|s| s.material).collect::<Vec<_>>(),
        tessellate(&styled).submeshes.iter().map(|s| s.material).collect::<Vec<_>>());
    let diffuse = tessellate(&styled.replace("(0.1,1.,0.7,96.,#213)", "(0.1,0.4,0.35,96.,#213)"));
    assert_eq!(diffuse.submeshes.len(), 6);
    assert!(diffuse.submeshes.iter().any(|s| s.material == SurfaceMaterial {
        diffuse_strength: Some(0.4), ..base
    }));
}

/// A right-handed AXIS2_PLACEMENT_3D frame in millimetres: location, then
/// its x (ref_direction), y = z × x, and z (axis) directions.
type Frame = [[f64; 3]; 4];

fn frame(location: [f64; 3], z: [f64; 3], x: [f64; 3]) -> Frame {
    let y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    [location, x, y, z]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Map a point given in `origin`'s world coordinates to the point with the
/// same local coordinates in `target`, which may also scale (CTO).
fn remap(p: [f64; 3], origin: &Frame, target: &Frame, scale: f64, is_point: bool) -> [f64; 3] {
    let d = if is_point { std::array::from_fn(|i| p[i] - origin[0][i]) } else { p };
    let local = [dot(d, origin[1]), dot(d, origin[2]), dot(d, origin[3])];
    std::array::from_fn(|i| {
        let t = if is_point { target[0][i] } else { 0. };
        t + scale * (local[0] * target[1][i] + local[1] * target[2][i] + local[2] * target[3][i])
    })
}

fn xyz(v: &triangulate::mesh::Vertex) -> ([f64; 3], [f64; 3]) {
    ([v.pos.x, v.pos.y, v.pos.z], [v.norm.x, v.norm.y, v.norm.z])
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

fn triangulate_text(text: &str) -> (triangulate::mesh::Mesh, triangulate::stats::Stats) {
    let flat = StepFile::strip_flatten(text.as_bytes()).unwrap();
    let step = StepFile::parse(&flat).unwrap();
    triangulate(&step)
}

/// Every vertex of `mesh[range]` is the matching cuboid vertex taken
/// through `f` (points) and `g` (normals), and its triangles face outward.
fn assert_instance(
    mesh: &triangulate::mesh::Mesh,
    plain: &triangulate::mesh::Mesh,
    copy: usize,
    f: &dyn Fn([f64; 3]) -> [f64; 3],
    g: &dyn Fn([f64; 3]) -> [f64; 3],
) {
    let n = plain.verts.len();
    for (i, v) in plain.verts.iter().enumerate() {
        let (p, norm) = xyz(v);
        let (q, qn) = xyz(&mesh.verts[copy * n + i]);
        assert!(close(q, f(p), 1e-9), "vertex {}: {:?} != {:?}", i, q, f(p));
        assert!(close(qn, g(norm), 1e-12), "normal {}: {:?} != {:?}", i, qn, g(norm));
    }
    let t = plain.triangles.len();
    for tri in &mesh.triangles[copy * t..(copy + 1) * t] {
        let [a, b, c] = [tri.verts.x, tri.verts.y, tri.verts.z].map(|i| xyz(&mesh.verts[i as usize]));
        let (u, w) = ([0, 1, 2].map(|k| b.0[k] - a.0[k]), [0, 1, 2].map(|k| c.0[k] - a.0[k]));
        let cross = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        assert!(dot(cross, a.1) > 0., "triangle winding disagrees with its normal");
    }
}

/// The cuboid's solid, ABSR #19, reached only through MAPPED_ITEM #300 in
/// the product's SHAPE_REPRESENTATION #123. Origin and target are both
/// rotated and translated, so any other composition order lands elsewhere.
const MAPPED: &str = "
    #300=MAPPED_ITEM('',#301,#310);
    #301=REPRESENTATION_MAP(#302,#19);
    #302=AXIS2_PLACEMENT_3D('',#303,#304,#305);
    #303=CARTESIAN_POINT('',(0.01,0.,0.));
    #304=DIRECTION('',(0.,0.,1.));
    #305=DIRECTION('',(0.,1.,0.));
    #310=AXIS2_PLACEMENT_3D('',#311,#312,#313);
    #311=CARTESIAN_POINT('',(0.,0.02,0.005));
    #312=DIRECTION('',(1.,0.,0.));
    #313=DIRECTION('',(0.,1.,0.));
";

fn mapped_only(extra: &str) -> String {
    let text = CUBOID
        .replace("#18=SHAPE_REPRESENTATION_RELATIONSHIP('','',#123,#19);", "")
        .replace("#123=SHAPE_REPRESENTATION('Part 1',(#124),#185);",
            "#123=SHAPE_REPRESENTATION('Part 1',(#124,#300),#185);");
    add_entities(&text, &format!("{}{}", MAPPED, extra))
}

#[test]
fn mapped_items_instance_their_representation_at_target_times_inverse_origin() {
    let (plain, _) = triangulate_text(CUBOID);
    // The model is in metres; the mesh is in millimetres.
    let origin = frame([10., 0., 0.], [0., 0., 1.], [0., 1., 0.]);
    let target = frame([0., 20., 5.], [1., 0., 0.], [0., 1., 0.]);
    let f = |p| remap(p, &origin, &target, 1., true);
    let g = |n| remap(n, &origin, &target, 1., false);

    // Only mapped geometry: one instance at the target, none stacked at the origin.
    let (mesh, stats) = triangulate_text(&mapped_only(""));
    assert!(stats.is_complete() && stats.num_faces == 6);
    assert_eq!(mesh.verts.len(), plain.verts.len());
    assert_instance(&mesh, &plain, 0, &f, &g);

    // The coloured path shares the traversal and keeps the solid's style.
    let (colored, stats) = tessellate_step_bytes(mapped_only("").as_bytes()).unwrap();
    assert!(stats.is_complete() && stats.num_faces == 6);
    let reference = tessellate(CUBOID);
    assert_eq!(colored.submeshes.len(), 1);
    assert_eq!(colored.submeshes[0].color, reference.submeshes[0].color);
    assert_eq!(colored.submeshes[0].positions.len(), reference.submeshes[0].positions.len());
    for (q, p) in colored.submeshes[0].positions.iter().zip(&reference.submeshes[0].positions) {
        let expected = f(p.map(f64::from));
        assert!(close(q.map(f64::from), expected, 1e-4), "{:?} != {:?}", q, expected);
    }

    // Nested: #19 is reached through a second, inner MAPPED_ITEM whose target
    // is a CARTESIAN_TRANSFORMATION_OPERATOR_3D that scales by 0.5 and is
    // left-handed (axis3 = -(axis1 × axis2)), i.e. mirrored.
    let nested = mapped_only("
        #320=SHAPE_REPRESENTATION('',(#321),#185);
        #321=MAPPED_ITEM('',#322,#330);
        #322=REPRESENTATION_MAP(#323,#19);
        #323=AXIS2_PLACEMENT_3D('',#324,#313,#304);
        #324=CARTESIAN_POINT('',(0.,0.,0.002));
        #330=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','',$,#331,#332,#333,0.5,#334);
        #331=DIRECTION('',(0.,0.,1.));
        #332=DIRECTION('',(1.,0.,0.));
        #333=CARTESIAN_POINT('',(0.003,-0.004,0.006));
        #334=DIRECTION('',(0.,-1.,0.));
    ").replace("#301=REPRESENTATION_MAP(#302,#19);", "#301=REPRESENTATION_MAP(#302,#320);");
    let inner_origin = frame([0., 0., 2.], [0., 1., 0.], [0., 0., 1.]);
    let inner_target: Frame = [[3., -4., 6.], [0., 0., 1.], [1., 0., 0.], [0., -1., 0.]];
    let f2 = |p| f(remap(p, &inner_origin, &inner_target, 0.5, true));
    let g2 = |n| g(remap(n, &inner_origin, &inner_target, 1., false));
    let (mesh, stats) = triangulate_text(&nested);
    assert!(stats.is_complete() && stats.num_faces == 6);
    assert_eq!(mesh.verts.len(), plain.verts.len());
    assert_instance(&mesh, &plain, 0, &f2, &g2);

    // Without axis2, an operator target completes a right-handed frame:
    // these are rotations, not a mirror (axis3 = -z) or a degenerate frame
    // (axis3 = x).
    for (axes, z, x) in [
        ("#350,$,#311,$,#351", [0., 0., -1.], [1., 0., 0.]),
        ("$,$,#311,$,#350", [1., 0., 0.], [0., 1., 0.]),
    ] {
        let text = mapped_only(&format!("
            #360=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','',$,{});
            #350=DIRECTION('',(1.,0.,0.));
            #351=DIRECTION('',(0.,0.,-1.));
        ", axes)).replace("#300=MAPPED_ITEM('',#301,#310);", "#300=MAPPED_ITEM('',#301,#360);");
        let target = frame([0., 20., 5.], z, x);
        let (mesh, stats) = triangulate_text(&text);
        assert!(stats.is_complete() && stats.num_faces == 6);
        assert_eq!(mesh.verts.len(), plain.verts.len());
        assert_instance(&mesh, &plain, 0,
            &|p| remap(p, &origin, &target, 1., true), &|n| remap(n, &origin, &target, 1., false));
    }

    // A product representation leads to a related representation that also
    // holds only MAPPED_ITEMs, whichever way the relationship is written.
    let linked = mapped_only("
        #370=SHAPE_REPRESENTATION('',(#371),#185);
        #371=MAPPED_ITEM('',#301,#310);
        #372=SHAPE_REPRESENTATION_RELATIONSHIP('','',#370,#123);
    ");
    let (mesh, stats) = triangulate_text(&linked);
    assert!(stats.is_complete() && stats.num_faces == 6);
    assert_eq!(mesh.verts.len(), 2 * plain.verts.len());
    assert_instance(&mesh, &plain, 0, &f, &g);
    assert_instance(&mesh, &plain, 1, &f, &g);

    // A solid instanced only at an unreadable placement is skipped, not
    // stacked at its untransformed position.
    let unresolved = mapped_only("").replace("#300=MAPPED_ITEM('',#301,#310);", "#300=MAPPED_ITEM('',#301,#311);");
    let (mesh, stats) = triangulate_text(&unresolved);
    assert_eq!((mesh.verts.len(), stats.num_faces), (0, 0));

    // A MAPPED_ITEM of non-mesh geometry leaves the fallback for solids the
    // product structure does not reach.
    let wireframe = mapped_only("
        #380=GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('',(#302),#185);
    ").replace("#301=REPRESENTATION_MAP(#302,#19);", "#301=REPRESENTATION_MAP(#302,#380);");
    let (mesh, stats) = triangulate_text(&wireframe);
    assert!(stats.is_complete() && stats.num_faces == 6);
    assert_eq!(mesh.verts.len(), plain.verts.len());
    assert_instance(&mesh, &plain, 0, &|p| p, &|n| n);
}

#[test]
fn mapped_instances_add_to_direct_bodies_and_cycles_terminate() {
    let (plain, _) = triangulate_text(CUBOID);
    let origin = frame([10., 0., 0.], [0., 0., 1.], [0., 1., 0.]);
    let target = frame([0., 20., 5.], [1., 0., 0.], [0., 1., 0.]);
    let f = |p| remap(p, &origin, &target, 1., true);
    let g = |n| remap(n, &origin, &target, 1., false);

    // #19, reached through the SHAPE_REPRESENTATION_RELATIONSHIP, holds the
    // solid directly and maps another representation of the same solid.
    let mixed = add_entities(
        &CUBOID.replace("#19=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#121),#185);",
            "#19=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#121,#300),#185);"),
        &format!("{}#320=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#121),#185);", MAPPED)
            .replace("#301=REPRESENTATION_MAP(#302,#19);", "#301=REPRESENTATION_MAP(#302,#320);"),
    );
    // #320 also maps #19, which maps #320 again.
    let cyclic = mixed
        .replace("#320=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#121),#185);",
            "#320=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#121,#340),#185);
             #340=MAPPED_ITEM('',#341,#310);#341=REPRESENTATION_MAP(#302,#19);");
    for text in [mixed, cyclic] {
        let (mesh, stats) = triangulate_text(&text);
        assert!(stats.is_complete() && stats.num_faces == 6);
        assert_eq!(mesh.verts.len(), 2 * plain.verts.len());
        // Instances are ordered by their matrices, not by discovery.
        let identity = |p| p;
        let mapped_first = close(xyz(&mesh.verts[0]).0, f(xyz(&plain.verts[0]).0), 1e-9);
        let (first, second): (&dyn Fn(_) -> _, &dyn Fn(_) -> _) =
            if mapped_first { (&f, &identity) } else { (&identity, &f) };
        let (first_n, second_n): (&dyn Fn(_) -> _, &dyn Fn(_) -> _) =
            if mapped_first { (&g, &identity) } else { (&identity, &g) };
        assert_instance(&mesh, &plain, 0, first, first_n);
        assert_instance(&mesh, &plain, 1, second, second_n);
    }
}

#[test]
fn presentation_mapped_items_do_not_change_the_mesh() {
    // Creo writes saved views as MAPPED_ITEMs of the part's representation
    // in draughting and presentation models; those are not geometry.
    let views = add_entities(CUBOID, &format!("{}
        #400=DRAUGHTING_MODEL('',(#300),#185);
        #401=PRESENTATION_VIEW('',(#402),#185);
        #402=MAPPED_ITEM('',#403,#310);
        #403=REPRESENTATION_MAP(#302,#123);
        #404=PRESENTATION_AREA('',(#405),#185);
        #405=MAPPED_ITEM('',#406,#311);
        #406=REPRESENTATION_MAP(#311,#401);
    ", MAPPED));
    let (plain, _) = triangulate_text(CUBOID);
    let (mesh, stats) = triangulate_text(&views);
    assert!(stats.is_complete() && stats.num_faces == 6);
    assert_eq!(mesh.verts.len(), plain.verts.len());
    assert_instance(&mesh, &plain, 0, &|p| p, &|n| n);
    assert_eq!(geometry(&tessellate(&views)), geometry(&tessellate(CUBOID)));
}

#[test]
fn scaled_mapped_instances_are_meshed_to_the_same_physical_tolerance() {
    // cube_hole's solid has a cylindrical hole: chords that are within
    // tolerance at scale 1 deviate ten times as far once scaled by 10.
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/cube_hole.step"),
    )
    .unwrap();
    let triangles = |scale: &str| {
        let text = source
            .replace("#18=SHAPE_REPRESENTATION_RELATIONSHIP('','',#143,#19);", "")
            .replace("#143=SHAPE_REPRESENTATION('Part 1',(#144),#219);",
                "#143=SHAPE_REPRESENTATION('Part 1',(#144,#900),#219);");
        let text = add_entities(&text, &format!("
            #900=MAPPED_ITEM('',#901,#902);
            #901=REPRESENTATION_MAP(#144,#19);
            #902=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','',$,$,$,#903,{},$);
            #903=CARTESIAN_POINT('',(0.,0.,0.));
        ", scale));
        let (mesh, stats) = triangulate_text(&text);
        assert!(stats.is_complete() && stats.num_faces > 0);
        mesh.triangles.len()
    };
    assert!(triangles("10.") > 2 * triangles("1."));
}
