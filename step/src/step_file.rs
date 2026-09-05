use log::warn;
use std::fmt;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{
    ap214::Entity,
    id::Id,
    parse::{parse_entity_decl, parse_entity_fallback},
};

#[derive(Debug)]
pub struct StepFile<'a>(pub Vec<Entity<'a>>);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StepParseError {
    message: String,
}

impl StepParseError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for StepParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "STEP parse error: {}", self.message)
    }
}

impl std::error::Error for StepParseError {}

impl<'a> StepFile<'a> {
    /// Parses a STEP file from a raw array of bytes
    /// `data` must be preprocessed by [`strip_flatten`] first
    pub fn parse(data: &'a [u8]) -> Result<Self, StepParseError> {
        let blocks = Self::into_blocks(data)?;
        if blocks.first().copied() != Some(b"ISO-10303-21;") {
            return Err(StepParseError::new("missing ISO-10303-21 start marker"));
        }
        let header_start = blocks.iter().position(|b| b == b"HEADER;")
            .ok_or_else(|| StepParseError::new("missing HEADER section"))?;
        let data_start = blocks.iter()
            .position(|b| b == b"DATA;")
            .ok_or_else(|| StepParseError::new("missing DATA section"))? + 1;
        if header_start >= data_start - 1 || !blocks[header_start + 1..data_start - 1]
            .iter().any(|b| b == b"ENDSEC;")
        {
            return Err(StepParseError::new("HEADER section missing ENDSEC"));
        }
        let data_end = blocks.iter()
            .skip(data_start)
            .position(|b| b == b"ENDSEC;")
            .map(|i| i + data_start)
            .ok_or_else(|| StepParseError::new("DATA section missing ENDSEC"))?;
        if !blocks[data_end + 1..].iter().any(|b| b == b"END-ISO-10303-21;") {
            return Err(StepParseError::new("missing END-ISO-10303-21 marker"));
        }

        // Parse every block, accumulating a Vec of Results.  We parse in
        // single-threaded mode in WASM builds, because there's no thread
        // pool.
        let block_iter = {
            let block_slice = &blocks[data_start..data_end];
            #[cfg(feature = "rayon")]
            { block_slice.par_iter() }
            #[cfg(not(feature = "rayon"))]
            { block_slice.iter() }
        };

        let parsed: Result<Vec<(usize, Entity)>, StepParseError> = block_iter
            .map(|b| parse_entity_decl(*b)
                .and_then(|(remaining, value)| {
                    // Complex entity parsing consumes the full declaration;
                    // simple entities leave the record terminator.
                    if remaining.is_empty() || remaining == ";" {
                        Ok((remaining, value))
                    } else {
                        Err(nom::Err::Error(nom::error::Error::new(
                            remaining, nom::error::ErrorKind::Eof)))
                    }
                })
                .or_else(|e| {
                    warn!("Failed to parse {}: {:?}",
                        std::str::from_utf8(b).unwrap_or("[INVALID UTF-8]"),
                              e);
                    parse_entity_fallback(*b).and_then(|(remaining, value)| {
                        if is_fallback_entity_record(remaining) {
                            Ok((remaining, value))
                        } else {
                            Err(nom::Err::Error(nom::error::Error::new(
                                remaining, nom::error::ErrorKind::Eof)))
                        }
                    })
                })
                .map(|b| b.1)
                .map_err(|_| StepParseError::new(format!(
                    "invalid DATA record: {}",
                    String::from_utf8_lossy(b)))))
            .collect();
        let parsed = parsed?;

        // Awkward construction because `Entity` is not `Clone`
        let max_id = parsed.iter().map(|b| b.0).max().unwrap_or(0);
        let mut out: Vec<Entity> = (0..=max_id)
            .map(|_| Entity::_EmptySlot)
            .collect();

        for p in parsed.into_iter() {
            out[p.0] = p.1;
        }

        Ok(Self(out))
    }

    /// Flattens a STEP file, removing comments and whitespace
    pub fn strip_flatten(data: &[u8]) -> Result<Vec<u8>, StepParseError> {
        let mut out = Vec::with_capacity(data.len());
        let mut i = 0;
        let mut in_string = false;
        let mut in_comment = false;
        while i < data.len() {
            if in_comment {
                if data[i..].starts_with(b"*/") {
                    in_comment = false;
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }

            if !in_string && data[i..].starts_with(b"/*") {
                in_comment = true;
                i += 2;
                continue;
            }

            let c = data[i];
            if c == b'\'' {
                out.push(c);
                if in_string && data.get(i + 1) == Some(&b'\'') {
                    out.push(b'\'');
                    i += 2;
                    continue;
                }
                in_string = !in_string;
            } else if !in_string && c.is_ascii_whitespace() {
                // Whitespace is insignificant outside literals.
            } else if !c.is_ascii() {
                // Preserve the established lossy policy for legacy encoded
                // files while ensuring the borrowed parser receives UTF-8.
                out.push(b'?');
            } else {
                out.push(c);
            }
            i += 1;
        }
        if in_comment {
            Err(StepParseError::new("unterminated comment"))
        } else if in_string {
            Err(StepParseError::new("unterminated string literal"))
        } else {
            Ok(out)
        }
    }

    /// Splits a STEP file into individual blocks.  The input must be pre-processed
    /// by [`strip_flatten`] beforehand.
    fn into_blocks(data: &[u8]) -> Result<Vec<&[u8]>, StepParseError> {
        let mut blocks = Vec::new();
        let mut start = 0;
        let mut in_string = false;
        let mut i = 0;
        while i < data.len() {
            if data[i] == b'\'' {
                if in_string && data.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_string = !in_string;
            } else if data[i] == b';' && !in_string {
                blocks.push(&data[start..=i]);
                start = i + 1;
            }
            i += 1;
        }
        if in_string {
            return Err(StepParseError::new("unterminated string literal"));
        }
        if data[start..].iter().any(|b| !b.is_ascii_whitespace()) {
            return Err(StepParseError::new("unterminated record at end of file"));
        }
        Ok(blocks)
    }

    pub fn entity<T: FromEntity<'a>>(&'a self, i: Id<T>) -> Option<&'a T> {
        self.0.get(i.0).and_then(T::try_from_entity)
    }
}

fn is_fallback_entity_record(s: &str) -> bool {
    let Some(body) = s.strip_prefix('=') else { return false };
    let Some(open) = body.find('(') else { return false };
    body.ends_with(");") && !body[..open].is_empty() && body[..open].bytes()
        .all(|c| c == b'_' || c.is_ascii_uppercase() || c.is_ascii_digit())
}

impl<'a, T> std::ops::Index<Id<T>> for StepFile<'a> {
    type Output = Entity<'a>;

    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.0[id.0]
    }
}

pub trait FromEntity<'a> {
    fn try_from_entity(e: &'a Entity<'a>) -> Option<&'a Self>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &[u8] = b"ISO-10303-21;HEADER;ENDSEC;DATA;#1=NOT_IN_AP214('a; b''s /* text */');ENDSEC;END-ISO-10303-21;";

    #[test]
    fn flatten_respects_literals_and_comments() {
        let flat = StepFile::strip_flatten(b" A /* remove; ' */ 'two words; /* keep */ it''s' ").unwrap();
        assert_eq!(flat, b"A'two words; /* keep */ it''s'");
    }

    #[test]
    fn parses_semicolons_and_quotes_in_literals() {
        let flat = StepFile::strip_flatten(MINIMAL).unwrap();
        let step = StepFile::parse(&flat).unwrap();
        assert_eq!(step.0.len(), 2);
    }

    #[test]
    fn malformed_and_non_step_inputs_are_errors() {
        assert!(StepFile::parse(b"**PARASOLID !").unwrap_err().to_string()
            .contains("unterminated record"));
        assert!(StepFile::parse(b"ISO-10303-21;HEADER;ENDSEC;").unwrap_err().to_string()
            .contains("missing DATA"));
        assert!(StepFile::strip_flatten(b"/* never closed").unwrap_err().to_string()
            .contains("unterminated comment"));
    }

    #[test]
    fn entity_returns_none_for_missing_id() {
        let file = StepFile(Vec::new());
        assert!(file.entity::<crate::ap214::CartesianPoint_>(Id::new(4)).is_none());
    }
}
