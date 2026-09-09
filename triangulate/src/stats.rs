use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Completion { Complete, Partial }

/// Stable, source-identified reason why part of a model was not tessellated.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    Geometry,
    Unsupported,
    Panic,
    InvalidEntity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TessellationFailure {
    /// STEP entity id of the shell or face which failed.
    pub entity_id: usize,
    /// STEP surface entity id, when a face identified one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_id: Option<usize>,
    pub kind: FailureKind,
    pub message: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Stats {
    pub num_shells: usize,
    pub num_faces: usize,
    pub failures: Vec<TessellationFailure>,
}

impl Stats {
    pub fn combine(mut a: Self, mut b: Self) -> Self {
        a.num_shells += b.num_shells;
        a.num_faces += b.num_faces;
        a.failures.append(&mut b.failures);
        a
    }

    pub fn is_complete(&self) -> bool { self.failures.is_empty() }
    pub fn completion(&self) -> Completion {
        if self.is_complete() { Completion::Complete } else { Completion::Partial }
    }
    pub fn num_errors(&self) -> usize {
        self.failures.iter().filter(|f| f.kind != FailureKind::Panic).count()
    }
    pub fn num_panics(&self) -> usize {
        self.failures.iter().filter(|f| f.kind == FailureKind::Panic).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_counts_cannot_disagree_with_failures() {
        let stats = Stats { failures: vec![
            TessellationFailure { entity_id: 1, surface_id: Some(2), kind: FailureKind::Geometry, message: "bad".into() },
            TessellationFailure { entity_id: 3, surface_id: None, kind: FailureKind::Panic, message: "panic".into() },
        ], ..Stats::default() };
        assert_eq!((stats.num_errors(), stats.num_panics()), (1, 1));
        assert!(!stats.is_complete());
    }
}
