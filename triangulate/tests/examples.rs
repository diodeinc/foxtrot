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

/// A surface colour style for `target`, overriding `over` when given.
fn color_style(id: u32, target: u32, over: Option<u32>, colour: &str) -> String {
    let item = match over {
        Some(over) => format!("OVER_RIDING_STYLED_ITEM('',(#{}),#{target},#{over})", id + 1),
        None => format!("STYLED_ITEM('',(#{}),#{target})", id + 1),
    };
    let [a, b, c, d, e, f] = std::array::from_fn(|i| id + 1 + i as u32);
    format!("#{id}={item};
        #{a}=PRESENTATION_STYLE_ASSIGNMENT((#{b}));
        #{b}=SURFACE_STYLE_USAGE(.BOTH.,#{c});
        #{c}=SURFACE_SIDE_STYLE('',(#{d}));
        #{d}=SURFACE_STYLE_FILL_AREA(#{e});
        #{e}=FILL_AREA_STYLE('',(#{f}));
        #{f}=FILL_AREA_STYLE_COLOUR('',#{}),;
        #{}={colour};", f + 1, f + 1).replace("),;", ");")
}

#[test]
fn overrides_beat_solid_and_shell_styles_and_predefined_colours_resolve() {
    // Overrides precede the styles they override, so entity order cannot
    // decide. #400 restyles the solid; faces #106-#108 are overridden.
    let styled = CUBOID.replace("(#113),", "(#430,#420,#410,#400,#113,#440),");
    let styled = add_entities(&styled, &[
        color_style(400, 121, Some(113), "COLOUR_RGB('',0.2,0.4,0.6)"),
        color_style(410, 106, Some(113), "DRAUGHTING_PRE_DEFINED_COLOUR('red')"),
        // Overrides the solid's override, through a context style assignment.
        color_style(420, 107, Some(400), "DRAUGHTING_PRE_DEFINED_COLOUR('black')")
            .replace("PRESENTATION_STYLE_ASSIGNMENT((#422))",
                "PRESENTATION_STYLE_BY_CONTEXT((#422),#121)"),
        color_style(430, 108, Some(440), "PRE_DEFINED_COLOUR('yellow')"),
        color_style(440, 108, None, "DRAUGHTING_PRE_DEFINED_COLOUR('green')"),
    ].concat());
    let colors = |mesh: &TessellatedMesh| {
        let mut colors: Vec<_> = mesh.submeshes.iter().flat_map(|s| s.normals.iter()
            .map(move |n| (n.map(|v| v as i8), s.color.map(|c| (c * 255.).round() as u8))))
            .collect();
        colors.sort();
        colors.dedup();
        colors
    };
    let blue = [51, 102, 153, 255];
    let faces = |rest: [u8; 4]| {
        let mut expected = vec![
            ([-1, 0, 0], [255, 0, 0, 255]),
            ([0, -1, 0], [0, 0, 0, 255]),
            ([1, 0, 0], [255, 255, 0, 255]),
            ([0, 0, -1], rest),
            ([0, 1, 0], rest),
            ([0, 0, 1], rest),
        ];
        expected.sort();
        expected
    };
    let mesh = tessellate(&styled);
    assert_eq!(colors(&mesh), faces(blue));
    assert_eq!(geometry(&mesh), geometry(&tessellate(CUBOID)));

    // A shell style beats the solid's but not the faces' overrides.
    let shell = styled.replace("#113,#440),", "#113,#440,#450),");
    let shell = add_entities(&shell, &color_style(450, 112, None, "DRAUGHTING_PRE_DEFINED_COLOUR('cyan')"));
    assert_eq!(colors(&tessellate(&shell)), faces([0, 255, 255, 255]));
}
