use log::warn;
use std::{fmt, ops::Range, sync::atomic::{AtomicBool, Ordering}};

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
        Self::parse_in_chunks(data, BYTES_PER_CHUNK, RECORDS_PER_CHUNK)
    }

    /// Parses with the given work sizes, which must not change the result.
    fn parse_in_chunks(data: &'a [u8], bytes_per_chunk: usize, records_per_chunk: usize)
        -> Result<Self, StepParseError>
    {
        let blocks = Self::into_blocks(data, bytes_per_chunk)?;
        let is = |i: usize, text: &[u8]| &data[blocks[i].clone()] == text;
        if blocks.is_empty() || !is(0, b"ISO-10303-21;") {
            return Err(StepParseError::new("missing ISO-10303-21 start marker"));
        }
        let header_start = (0..blocks.len()).position(|i| is(i, b"HEADER;"))
            .ok_or_else(|| StepParseError::new("missing HEADER section"))?;
        let data_start = (0..blocks.len())
            .position(|i| is(i, b"DATA;"))
            .ok_or_else(|| StepParseError::new("missing DATA section"))? + 1;
        if header_start >= data_start - 1 || !(header_start + 1..data_start - 1)
            .any(|i| is(i, b"ENDSEC;"))
        {
            return Err(StepParseError::new("HEADER section missing ENDSEC"));
        }
        let data_end = (data_start..blocks.len())
            .find(|&i| is(i, b"ENDSEC;"))
            .ok_or_else(|| StepParseError::new("DATA section missing ENDSEC"))?;
        if !(data_end + 1..blocks.len()).any(|i| is(i, b"END-ISO-10303-21;")) {
            return Err(StepParseError::new("missing END-ISO-10303-21 marker"));
        }

        let records = &blocks[data_start..data_end];
        let chunks: Vec<_> = records.chunks(records_per_chunk).collect();
        // Records are adjacent, so one UTF-8 check usually covers a chunk.
        let texts = map_in_order(chunks.len(), 1, |c| {
            let c = chunks[c];
            std::str::from_utf8(&data[c[0].start..c[c.len() - 1].end]).ok()
        });
        let parse_nth = |r: usize| {
            let (c, range) = (r / records_per_chunk, &records[r]);
            let text = match texts[c] {
                Some(text) => {
                    let base = chunks[c][0].start;
                    Some(&text[range.start - base..range.end - base])
                }
                None => std::str::from_utf8(&data[range.clone()]).ok(),
            };
            text.and_then(|s| parse_record(s).ok()).map(|(_, (_, entity))| entity)
        };

        // Size the ID table from record headers and map each ID to its last
        // record, which a sequential parse would have kept. A header that
        // does not parse belongs to an invalid record, reported below.
        let ids: Vec<Option<usize>> = map_in_order(chunks.len(), 1, |c| {
            chunks[c].iter().map(|r| record_id(&data[r.clone()])).collect::<Vec<_>>()
        }).into_iter().flatten().collect();
        let max_id = ids.iter().flatten().copied().max().unwrap_or(0);
        let mut record_of = vec![NO_RECORD; max_id + 1];
        for (r, id) in ids.iter().enumerate() {
            if let Some(id) = *id {
                record_of[id] = r;
            }
        }

        // Parse each ID's record straight into its slot, in parallel. Records
        // shadowed by a later duplicate are parsed only to detect errors.
        let failed = AtomicBool::new(false);
        let out: Vec<Entity> = map_in_order(max_id + 1, records_per_chunk, |id| match record_of[id] {
            NO_RECORD => Entity::_EmptySlot,
            r => parse_nth(r).unwrap_or_else(|| {
                failed.store(true, Ordering::Relaxed);
                Entity::_EmptySlot
            }),
        });
        if failed.into_inner() || ids.iter().enumerate().any(|(r, id)| match *id {
            Some(id) => record_of[id] != r && parse_nth(r).is_none(),
            None => true,
        }) {
            let first = map_in_order(chunks.len(), 1, |c| {
                let start = c * records_per_chunk;
                (start..start + chunks[c].len()).find(|&r| parse_nth(r).is_none())
            });
            let r = first.into_iter().flatten().next().expect("an invalid record");
            return Err(StepParseError::new(format!(
                "invalid DATA record: {}",
                String::from_utf8_lossy(&data[records[r].clone()]))));
        }

        // One bit per ID keeps reference lookups in cache.
        let mut defined = vec![0u64; max_id / 64 + 1];
        for id in (0..=max_id).filter(|&id| record_of[id] != NO_RECORD) {
            defined[id / 64] |= 1 << (id % 64);
        }
        let errors = map_in_order(chunks.len(), 1, |c| chunks[c].iter()
            .find_map(|r| check_references(&data[r.clone()], &defined)));
        if let Some(e) = errors.into_iter().flatten().next() {
            return Err(e);
        }

        Ok(Self(out))
    }

    /// Flattens a STEP file, removing comments and whitespace
    pub fn strip_flatten(data: &[u8]) -> Result<Vec<u8>, StepParseError> {
        Self::flatten_in_chunks(data, BYTES_PER_CHUNK)
    }

    /// Flattens with the given work size, which must not change the result.
    fn flatten_in_chunks(data: &[u8], bytes_per_chunk: usize) -> Result<Vec<u8>, StepParseError> {
        // Chunks end after a newline, so no `/*` or `*/` spans two chunks.
        // Each chunk is flattened as if it starts in code; a chunk that
        // actually starts inside a literal or comment is flattened again.
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < data.len() {
            let end = match start.checked_add(bytes_per_chunk) {
                Some(target) if target < data.len() => memchr::memchr(b'\n', &data[target..])
                    .map_or(data.len(), |i| target + i + 1),
                _ => data.len(),
            };
            chunks.push(start..end);
            start = end;
        }
        let mut flat = map_in_order(chunks.len(), 1,
            |c| flatten_chunk(&data[chunks[c].clone()], Lexical::Code));
        let mut state = Lexical::Code;
        for (r, chunk) in chunks.iter().zip(&mut flat) {
            if state != Lexical::Code {
                *chunk = flatten_chunk(&data[r.clone()], state);
            }
            state = chunk.1;
        }
        match state {
            Lexical::Comment => Err(StepParseError::new("unterminated comment")),
            Lexical::Literal => Err(StepParseError::new("unterminated string literal")),
            Lexical::Code if flat.len() <= 1 => Ok(flat.pop().map(|(f, _)| f).unwrap_or_default()),
            Lexical::Code => {
                let mut out = vec![0; flat.iter().map(|(f, _)| f.len()).sum()];
                let mut dst = Vec::with_capacity(flat.len());
                let mut rest = &mut out[..];
                for (f, _) in &flat {
                    let (head, tail) = rest.split_at_mut(f.len());
                    dst.push(head);
                    rest = tail;
                }
                #[cfg(feature = "rayon")]
                dst.into_par_iter().zip(&flat).for_each(|(d, (f, _))| d.copy_from_slice(f));
                #[cfg(not(feature = "rayon"))]
                dst.into_iter().zip(&flat).for_each(|(d, (f, _))| d.copy_from_slice(f));
                Ok(out)
            }
        }
    }

    /// Splits a STEP file into individual blocks, returned as byte ranges.
    /// The input must be pre-processed by [`strip_flatten`] beforehand.
    fn into_blocks(data: &[u8], bytes_per_chunk: usize) -> Result<Vec<Range<usize>>, StepParseError> {
        // A semicolon ends a block unless an odd number of quotes precede it
        // (doubled quotes toggle twice). Each chunk records its semicolons by
        // the parity of the quotes before them within the chunk, so chunks
        // are scanned independently and joined by parity.
        let chunks: Vec<_> = (0..data.len()).step_by(bytes_per_chunk)
            .map(|start| start..data.len().min(start.saturating_add(bytes_per_chunk)))
            .collect();
        let scanned = map_in_order(chunks.len(), 1, |c| {
            let r = &chunks[c];
            let mut semicolons = [Vec::new(), Vec::new()];
            let mut odd = false;
            for i in memchr::memchr2_iter(b'\'', b';', &data[r.clone()]) {
                if data[r.start + i] == b'\'' {
                    odd = !odd;
                } else {
                    semicolons[odd as usize].push(r.start + i);
                }
            }
            (semicolons, odd)
        });
        let mut blocks = Vec::with_capacity(scanned.iter().map(|(s, _)| s[0].len()).sum());
        let mut start = 0;
        let mut in_string = false;
        for (semicolons, odd) in scanned {
            for i in &semicolons[in_string as usize] {
                blocks.push(start..i + 1);
                start = i + 1;
            }
            in_string ^= odd;
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

/// Input bytes per parallel flattening or block-splitting task. Without
/// `rayon` the whole file is one chunk, which is the sequential algorithm.
const BYTES_PER_CHUNK: usize = if cfg!(feature = "rayon") { 1 << 20 } else { usize::MAX };
/// DATA records per parallel parsing or validation task.
const RECORDS_PER_CHUNK: usize = if cfg!(feature = "rayon") { 4096 } else { usize::MAX };

/// Marks an ID that no DATA record declares.
const NO_RECORD: usize = usize::MAX;

/// Returns `[f(0), f(1), ..., f(n - 1)]`. With the `rayon` feature, tasks of
/// at least `min_len` items run in parallel; smaller inputs stay on the
/// calling thread.
fn map_in_order<R: Send>(n: usize, min_len: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    #[cfg(feature = "rayon")]
    if n > min_len {
        return (0..n).into_par_iter().with_min_len(min_len).map(f).collect();
    }
    (0..n).map(f).collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Lexical {
    Code,
    Literal,
    Comment,
}

/// Byte classes for [`flatten_chunk`] outside literals and comments.
const DROP: u8 = 0;
const KEEP: u8 = 1;
const SPECIAL: u8 = 2;
const CODE_CLASS: [u8; 256] = {
    let mut class = [KEEP; 256];
    // Whitespace (as in `u8::is_ascii_whitespace`) is insignificant
    // outside literals.
    class[b' ' as usize] = DROP;
    class[b'\t' as usize] = DROP;
    class[b'\n' as usize] = DROP;
    class[b'\x0C' as usize] = DROP;
    class[b'\r' as usize] = DROP;
    class[b'\'' as usize] = SPECIAL;
    class[b'/' as usize] = SPECIAL;
    let mut c = 0x80;
    while c < 256 {
        class[c] = SPECIAL;
        c += 1;
    }
    class
};

/// Returns `data` without comments, and without whitespace outside literals,
/// starting in `state`, and the state at the end of `data`.
fn flatten_chunk(data: &[u8], mut state: Lexical) -> (Vec<u8>, Lexical) {
    // Output never exceeds input, so write at `j` into a sized buffer.
    let mut out = vec![0; data.len()];
    let mut j = 0;
    let mut i = 0;
    while i < data.len() {
        match state {
            Lexical::Comment => match memchr::memmem::find(&data[i..], b"*/") {
                Some(n) => {
                    i += n + 2;
                    state = Lexical::Code;
                }
                None => break,
            },
            Lexical::Code => {
                // Copy blocks without whitespace or special bytes at once;
                // otherwise copy bytes, dropping whitespace without branches.
                'code: while i < data.len() {
                    let block = &data[i..data.len().min(i + 16)];
                    if block.len() == 16 && block.iter().fold(true, |all, &c|
                        all & (c > b' ') & (c < 0x80) & (c != b'\'') & (c != b'/'))
                    {
                        out[j..j + 16].copy_from_slice(block);
                        i += 16;
                        j += 16;
                        continue;
                    }
                    for &c in block {
                        let class = CODE_CLASS[c as usize];
                        if class == SPECIAL {
                            break 'code;
                        }
                        out[j] = c;
                        j += class as usize;
                        i += 1;
                    }
                }
                let Some(&c) = data.get(i) else { break };
                i += 1;
                if c == b'/' && data.get(i) == Some(&b'*') {
                    i += 1;
                    state = Lexical::Comment;
                    continue;
                }
                out[j] = match c {
                    b'\'' => {
                        state = Lexical::Literal;
                        c
                    }
                    b'/' => c,
                    // Preserve the established lossy policy for legacy
                    // encoded files while ensuring the borrowed parser
                    // receives UTF-8.
                    _ => b'?',
                };
                j += 1;
            }
            Lexical::Literal => {
                let n = data[i..].iter().position(|c| *c == b'\'' || !c.is_ascii())
                    .unwrap_or(data.len() - i);
                out[j..j + n].copy_from_slice(&data[i..i + n]);
                i += n;
                j += n;
                if let Some(&c) = data.get(i) {
                    i += 1;
                    if c == b'\'' {
                        // Escaped quotes toggle twice with no bytes between them.
                        out[j] = c;
                        state = Lexical::Code;
                    } else {
                        out[j] = b'?';
                    }
                    j += 1;
                }
            }
        }
    }
    out.truncate(j);
    (out, state)
}

fn parse_record(s: &str) -> crate::parse::IResult<'_, (usize, Entity<'_>)> {
    parse_entity_decl(s)
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
            warn!("Failed to parse {}: {:?}", s, e);
            parse_entity_fallback(s).and_then(|(remaining, value)| {
                if is_fallback_entity_record(remaining) {
                    Ok((remaining, value))
                } else {
                    Err(nom::Err::Error(nom::error::Error::new(
                        remaining, nom::error::ErrorKind::Eof)))
                }
            })
        })
}

/// Reads the ID from a record's `#ID=` header.
fn record_id(b: &[u8]) -> Option<usize> {
    let b = b.strip_prefix(b"#")?;
    let end = b.iter().position(|c| *c == b'=')?;
    std::str::from_utf8(&b[..end]).ok()?.parse::<usize>().ok()
}

/// Checks explicit entity references in an original record.  This is kept
/// separate from `Entity::upstream`, because `$` is represented internally as
/// ID 0 and fallback entities do not expose their parameters there.
fn check_references(block: &[u8], defined: &[u64]) -> Option<StepParseError> {
    let equals = block.iter().position(|c| *c == b'=').expect("parsed DATA declaration");
    let mut in_string = false;
    for offset in memchr::memchr2_iter(b'\'', b'#', &block[equals + 1..]) {
        let i = equals + 1 + offset;
        if block[i] == b'\'' {
            in_string = !in_string;
        } else if !in_string {
            let start = i + 1;
            let mut end = start;
            while block.get(end).map_or(false, u8::is_ascii_digit) {
                end += 1;
            }
            if end > start {
                let target_text = std::str::from_utf8(&block[start..end]).unwrap();
                let target = target_text.parse::<usize>().ok();
                if !target.map_or(false, |t| defined.get(t / 64).map_or(false, |w| w >> (t % 64) & 1 == 1)) {
                    return Some(StepParseError::new(format!(
                        "entity {} references undefined entity #{}",
                        String::from_utf8_lossy(&block[..equals]), target_text
                    )));
                }
            }
        }
    }
    None
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

    fn parse_data(records: &str) -> Result<(), StepParseError> {
        let data = format!(
            "ISO-10303-21;HEADER;ENDSEC;DATA;{}ENDSEC;END-ISO-10303-21;",
            records
        );
        StepFile::parse(data.as_bytes()).map(|_| ())
    }

    #[test]
    fn flatten_respects_literals_and_comments() {
        let flat = StepFile::strip_flatten(b" A /* remove; ' */ 'two words; /* keep */ it''s' ").unwrap();
        assert_eq!(flat, b"A'two words; /* keep */ it''s'");
        let flat = StepFile::strip_flatten(b"A/\xff /*'*/'two\xfe words''/*keep*/'\tB").unwrap();
        assert_eq!(flat, b"A/?'two? words''/*keep*/'B");
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
        // SI_UNIT is a NAMED_UNIT, which this complex entity omits.
        assert!(parse_data("#1=(LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));").unwrap_err()
            .to_string().contains("invalid DATA record"));
    }

    #[test]
    fn entity_returns_none_for_missing_id() {
        let file = StepFile(Vec::new());
        assert!(file.entity::<crate::ap214::CartesianPoint_>(Id::new(4)).is_none());
    }

    #[test]
    fn sparse_out_of_order_records_keep_their_ids() {
        let file = StepFile::parse(b"ISO-10303-21;HEADER;ENDSEC;DATA;\
            #19=CARTESIAN_POINT('',(3.,7.,11.));\
            #2=VERTEX_POINT('',#19);ENDSEC;END-ISO-10303-21;").unwrap();
        assert_eq!(file.0.len(), 20);
        assert!(matches!(file.0[1], Entity::_EmptySlot));
        let point = file.entity::<crate::ap214::CartesianPoint_>(Id::new(19)).unwrap();
        assert_eq!(point.coordinates.iter().map(|c| c.0).collect::<Vec<_>>(), [3., 7., 11.]);
        let vertex = file.entity::<crate::ap214::VertexPoint_>(Id::new(2)).unwrap();
        assert_eq!(vertex.vertex_geometry.0, 19);
    }

    #[test]
    fn rejects_missing_explicit_references_with_source_and_target() {
        let in_range = parse_data("#1=UNKNOWN(#2);#3=UNKNOWN($);").unwrap_err().to_string();
        assert!(in_range.contains("#1 references undefined entity #2"));

        let out_of_range = parse_data("#7=UNKNOWN(#99);").unwrap_err().to_string();
        assert!(out_of_range.contains("#7 references undefined entity #99"));

        let explicit_zero = parse_data("#7=VERTEX_POINT('',#0);").unwrap_err().to_string();
        assert!(explicit_zero.contains("#7 references undefined entity #0"));
    }

    #[test]
    fn chunk_sizes_do_not_change_results() {
        // A literal and a comment span lines and hold delimiters. IDs are out
        // of order and #3 repeats, so its last record must win.
        let text = b"ISO-10303-21;\r\nHEADER;/* a ' quote\n; and */ENDSEC;\nDATA;\n\
            #3=CARTESIAN_POINT('first',(0.,0.,0.));\n\
            #2 = VERTEX_POINT ( 'it''s; /* not a\ncomment */', #3 ) ;\n\
            #1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n\
            #3=CARTESIAN_POINT('caf\xc3\xa9',(1.,2.,3.));\n\
            #9=NOT_IN_AP214('x');\nENDSEC;\nEND-ISO-10303-21;\n";
        let flat = StepFile::flatten_in_chunks(text, usize::MAX).unwrap();
        let flat_text = std::str::from_utf8(&flat).unwrap();
        assert!(flat_text.starts_with("ISO-10303-21;HEADER;ENDSEC;DATA;#3="));
        assert!(flat_text.contains("#2=VERTEX_POINT('it''s; /* not a\ncomment */',#3);"));
        assert!(flat_text.contains("#3=CARTESIAN_POINT('caf??',(1.,2.,3.));"));
        let step = StepFile::parse_in_chunks(&flat, usize::MAX, usize::MAX).unwrap();
        let point = step.entity::<crate::ap214::CartesianPoint_>(Id::new(3)).unwrap();
        assert_eq!(point.coordinates.iter().map(|c| c.0).collect::<Vec<_>>(), [1., 2., 3.]);
        assert_eq!(step.0.len(), 10);
        let entities = format!("{:?}", step);

        for bytes in 1..=text.len() {
            assert_eq!(StepFile::flatten_in_chunks(text, bytes).unwrap(), flat);
            for records in 1..=6 {
                let step = StepFile::parse_in_chunks(&flat, bytes, records).unwrap();
                assert_eq!(format!("{:?}", step), entities);
            }
        }
    }

    #[test]
    fn chunked_parsing_reports_the_first_error_in_file_order() {
        // Both invalid records are shadowed by later duplicate IDs.
        let invalid = b"ISO-10303-21;HEADER;ENDSEC;DATA;#1=lowercase(1);#2=NOT_IN_AP214(#1);\
            #1=NOT_IN_AP214(1);#3=lowercase(3);#3=NOT_IN_AP214(3);ENDSEC;END-ISO-10303-21;";
        let undefined = b"ISO-10303-21;HEADER;ENDSEC;DATA;#1=NOT_IN_AP214(1);\
            #2=NOT_IN_AP214(#8);#3=NOT_IN_AP214(#9);ENDSEC;END-ISO-10303-21;";
        for bytes in 1..=invalid.len() {
            for records in 1..=4 {
                let e = StepFile::parse_in_chunks(invalid, bytes, records).unwrap_err();
                assert_eq!(e.to_string(), "STEP parse error: invalid DATA record: #1=lowercase(1);");
                let e = StepFile::parse_in_chunks(undefined, bytes, records).unwrap_err();
                assert!(e.to_string().ends_with("#2 references undefined entity #8"));
            }
        }
        let unterminated = b"#1=A('x\n;');\n/* open\n#2=B('y');\n";
        for bytes in 1..=unterminated.len() {
            assert!(StepFile::flatten_in_chunks(unterminated, bytes).unwrap_err().to_string()
                .contains("unterminated comment"));
        }
    }

    #[test]
    fn reference_validation_handles_literals_forward_refs_nulls_and_fallbacks() {
        parse_data(
            "#1=UNKNOWN('literal #404 and it''s still #405',#2,$);\
             #2=ANOTHER_UNKNOWN('ok');"
        ).unwrap();
    }
}
