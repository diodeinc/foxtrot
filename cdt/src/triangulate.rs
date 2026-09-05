use std::collections::VecDeque;

use spade::{handles::FixedVertexHandle, ConstrainedDelaunayTriangulation, Point2,
            Triangulation as SpadeTriangulation};

use crate::{Error, Point};

type Engine = ConstrainedDelaunayTriangulation<Point2<f64>, (), bool>;

/// An incrementally-built constrained Delaunay triangulation.
///
/// Each call to [`step`](Self::step) inserts exactly one input point.  After
/// the last point, one final call installs every constraint and classifies
/// faces by even/odd boundary parity. Duplicate coordinates share a vertex,
/// but every input index remains mapped to that vertex.
pub struct Triangulation {
    points: Vec<Point>,
    edges: Vec<(usize, usize)>,
    engine: Engine,
    input_handles: Vec<FixedVertexHandle>,
    vertex_input: Vec<usize>,
    selected: Vec<bool>,
    next: usize,
    complete: bool,
}

impl Triangulation {
    /// Builds a complete unconstrained triangulation.
    pub fn build(points: &[Point]) -> Result<Self, Error> {
        let mut result = Self::new(points)?;
        result.run()?;
        Ok(result)
    }

    /// Builds a complete triangulation with fixed boundary edges.
    pub fn build_with_edges<'a, E>(points: &[Point], edges: E) -> Result<Self, Error>
    where E: IntoIterator<Item = &'a (usize, usize)> + Copy {
        let mut result = Self::new_with_edges(points, edges)?;
        result.run()?;
        Ok(result)
    }

    /// Builds a complete triangulation from closed contours.
    pub fn build_from_contours<V>(points: &[Point], contours: &[V]) -> Result<Self, Error>
    where for<'a> &'a V: IntoIterator<Item = &'a usize> {
        let mut result = Self::new_from_contours(points, contours)?;
        result.run()?;
        Ok(result)
    }

    /// Creates an empty engine which will incrementally insert `points`.
    pub fn new(points: &[Point]) -> Result<Self, Error> {
        Self::new_with_edges(points, &[])
    }

    /// Creates an empty engine which will incrementally insert `points` and
    /// then install `edges` in a final step.
    pub fn new_with_edges<'a, E>(points: &[Point], edges: E) -> Result<Self, Error>
    where E: IntoIterator<Item = &'a (usize, usize)> + Copy {
        if points.is_empty() { return Err(Error::EmptyInput); }
        if points.len() < 3 { return Err(Error::TooFewPoints); }
        for &p in points {
            spade::validate_vertex(&Point2::new(p.0, p.1)).map_err(|_| Error::InvalidInput)?;
        }
        let edges: Vec<_> = edges.into_iter().copied().collect();
        if edges.iter().any(|&(a, b)| a >= points.len() || b >= points.len() || a == b) {
            return Err(Error::InvalidEdge);
        }
        Ok(Self {
            points: points.to_vec(), edges, engine: Engine::new(),
            input_handles: Vec::with_capacity(points.len()), vertex_input: Vec::new(),
            selected: Vec::new(), next: 0, complete: false,
        })
    }

    /// Creates an incremental triangulation from closed contours.
    pub fn new_from_contours<V>(points: &[Point], contours: &[V]) -> Result<Self, Error>
    where for<'a> &'a V: IntoIterator<Item = &'a usize> {
        let mut edges = Vec::new();
        for contour in contours {
            let vertices: Vec<_> = contour.into_iter().copied().collect();
            if vertices.len() >= 2 && vertices.first() != vertices.last() {
                return Err(Error::OpenContour);
            }
            edges.extend(vertices.windows(2).map(|v| (v[0], v[1])));
        }
        Self::new_with_edges(points, &edges)
    }

    /// Runs all remaining insertion and finalization steps.
    pub fn run(&mut self) -> Result<(), Error> {
        while !self.done() { self.step()?; }
        Ok(())
    }

    /// Returns whether constraints and face classification are complete.
    pub fn done(&self) -> bool { self.complete }

    /// Advances by one point insertion, or finalizes after all points exist.
    pub fn step(&mut self) -> Result<(), Error> {
        if self.complete { return Err(Error::NoMorePoints); }
        if self.next < self.points.len() {
            let p = self.points[self.next];
            let before = self.engine.num_vertices();
            let handle = self.engine.insert(Point2::new(p.0, p.1)).map_err(|_| Error::InvalidInput)?;
            if self.engine.num_vertices() != before {
                debug_assert_eq!(handle.index(), self.vertex_input.len());
                self.vertex_input.push(self.next);
            }
            self.input_handles.push(handle);
            self.next += 1;
            return Ok(());
        }
        self.finalize()
    }

    fn finalize(&mut self) -> Result<(), Error> {
        if self.engine.num_inner_faces() == 0 { return Err(Error::CannotInitialize); }
        for &(a, b) in &self.edges {
            let from = self.input_handles[a];
            let to = self.input_handles[b];
            if from == to { continue; }
            let added = self.engine.try_add_constraint(from, to);
            if added.is_empty() { return Err(Error::CrossingFixedEdge); }
            for edge in added {
                let parity = self.engine.undirected_edge_data_mut(edge.as_undirected()).data_mut();
                *parity ^= true;
            }
        }
        self.classify()?;
        self.complete = true;
        Ok(())
    }

    fn classify(&mut self) -> Result<(), Error> {
        if self.edges.is_empty() {
            self.selected = vec![true; self.engine.num_inner_faces()];
            return Ok(());
        }
        let mut state = vec![None; self.engine.num_inner_faces() + 1];
        state[0] = Some(false); // Spade's outer face always has fixed index zero.
        let mut queue = VecDeque::from([0usize]);
        while let Some(face_index) = queue.pop_front() {
            let value = state[face_index].unwrap();
            let edges: Vec<_> = if face_index == 0 {
                self.engine.convex_hull().map(|e| e.fix()).collect()
            } else {
                self.engine.inner_faces().nth(face_index - 1).expect("valid queued face")
                    .adjacent_edges().iter().map(|e| e.fix()).collect()
            };
            for edge in edges {
                let dynamic = self.engine.directed_edge(edge);
                let other = dynamic.rev().face().fix().index();
                let parity = *dynamic.as_undirected().data().data();
                let expected = value ^ parity;
                match state[other] {
                    None => { state[other] = Some(expected); queue.push_back(other); }
                    Some(found) if found != expected => return Err(Error::OpenContour),
                    _ => {}
                }
            }
        }
        if state.iter().any(Option::is_none) { return Err(Error::HalfEdgeInvariant); }
        self.selected = state[1..].iter().map(|v| v == &Some(true)).collect();
        Ok(())
    }

    /// Checks mapping, constraint and classified-face invariants.
    pub fn check(&self) {
        assert_eq!(self.input_handles.len(), self.next);
        assert_eq!(self.vertex_input.len(), self.engine.num_vertices());
        assert!(self.input_handles.iter().all(|h| h.index() < self.engine.num_vertices()));
        if self.complete { assert_eq!(self.selected.len(), self.engine.num_inner_faces()); }
    }

    /// Iterates triangles as counter-clockwise original input indices.
    pub fn triangles(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.engine.inner_faces().enumerate().filter_map(move |(i, face)| {
            if self.complete && !self.selected[i] { return None; }
            let v = face.vertices();
            Some((self.vertex_input[v[0].fix().index()], self.vertex_input[v[1].fix().index()],
                  self.vertex_input[v[2].fix().index()]))
        })
    }

    /// Returns whether `point` is covered by an emitted triangle.
    pub fn inside(&self, point: Point) -> bool {
        self.triangles().any(|(a, b, c)| contains(self.points[a], self.points[b], self.points[c], point))
    }

    /// Writes the current triangulation to an SVG file.
    pub fn save_svg(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(filename, self.to_svg(false))
    }

    /// Writes an SVG with pending source constraints highlighted.
    pub fn save_debug_svg(&self, filename: &str) -> std::io::Result<()> {
        std::fs::write(filename, self.to_svg(true))
    }

    /// Renders points, current engine edges, constraints, and selected faces.
    pub fn to_svg(&self, debug: bool) -> String {
        let (mut min_x, mut max_x, mut min_y, mut max_y) =
            (self.points[0].0, self.points[0].0, self.points[0].1, self.points[0].1);
        for &(x, y) in &self.points { min_x=min_x.min(x); max_x=max_x.max(x); min_y=min_y.min(y); max_y=max_y.max(y); }
        let span = (max_x-min_x).max(max_y-min_y).max(1.0e-30);
        let s=800.0/span; let x=|v| (v-min_x)*s+2.0; let y=|v| (max_y-v)*s+2.0;
        let mut out=format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="804" height="804"><rect width="100%" height="100%" fill="black"/>"#);
        for edge in self.engine.undirected_edges() { let p=edge.positions(); let fixed=edge.is_constraint_edge(); out.push_str(&format!(r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,x(p[0].x),y(p[0].y),x(p[1].x),y(p[1].y),if fixed{"white"}else{"red"})); }
        if debug && !self.complete { for &(a,b) in &self.edges { let (p,q)=(self.points[a],self.points[b]); out.push_str(&format!(r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="lime" stroke-width="2" stroke-dasharray="4"/>"#,x(p.0),y(p.1),x(q.0),y(q.1))); } }
        for &(px,py) in &self.points[..self.next] { out.push_str(&format!(r#"<circle cx="{}" cy="{}" r="2" fill="pink"/>"#,x(px),y(py))); }
        out.push_str("</svg>"); out
    }
}

fn contains(a: Point, b: Point, c: Point, p: Point) -> bool {
    let cross=|u:Point,v:Point,w:Point| (v.0-u.0)*(w.1-u.1)-(v.1-u.1)*(w.0-u.0);
    cross(a,b,p)>=0.0 && cross(b,c,p)>=0.0 && cross(c,a,p)>=0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holes_and_nested_loops_use_even_odd_parity() {
        let p=[(0.,0.),(4.,0.),(4.,4.),(0.,4.),(1.,1.),(3.,1.),(3.,3.),(1.,3.)];
        let e=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4)];
        let t=Triangulation::build_with_edges(&p,&e).unwrap();
        assert!(t.inside((0.5,0.5))); assert!(!t.inside((2.,2.))); assert!(!t.inside((5.,2.)));
    }

    #[test]
    fn duplicates_and_collinear_boundary_vertices_preserve_provenance() {
        let p=[(0.,0.),(4.,0.),(4.,4.),(0.,4.),(2.,0.),(4.,4.),(2.,2.)];
        let e=[(0,1),(1,5),(5,3),(3,0)];
        let t=Triangulation::build_with_edges(&p,&e).unwrap();
        t.check();
        assert!(t.inside((2.,2.)));
        assert!(t.triangles().all(|(a,b,c)| a<p.len() && b<p.len() && c<p.len()));
    }

    #[test]
    fn duplicate_boundary_xor_cancels_without_unlocking() {
        let p=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)];
        let e=[(0,1),(1,2),(2,3),(3,0),(0,1),(1,2),(2,3),(3,0)];
        let t=Triangulation::build_with_edges(&p,&e).unwrap();
        assert_eq!(t.triangles().count(),0);
        assert_eq!(t.engine.num_constraints(),4);
    }

    #[test]
    fn crossing_open_and_degenerate_inputs_are_rejected() {
        let p=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)];
        assert_eq!(Triangulation::build_with_edges(&p,&[(0,2),(1,3)]).err(),Some(Error::CrossingFixedEdge));
        assert_eq!(Triangulation::build_with_edges(&p,&[(0,1)]).err(),Some(Error::OpenContour));
        assert_eq!(Triangulation::build(&[(0.,0.),(1.,0.),(2.,0.)]).err(),Some(Error::CannotInitialize));
    }

    #[test]
    fn thin_translated_polygon_is_not_snapped() {
        let p=[(1e9,1e9),(1e9+1.,1e9),(1e9+1.,1e9+1e-5),(1e9,1e9+1e-5)];
        let e=[(0,1),(1,2),(2,3),(3,0)];
        let t=Triangulation::build_with_edges(&p,&e).unwrap();
        assert_eq!(t.triangles().count(),2);
    }
}
