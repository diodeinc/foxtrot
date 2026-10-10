use nom::{
    branch::{alt},
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::{map, map_res, opt},
    error::*,
    sequence::{delimited, preceded, tuple},
    multi::{separated_list0},
};
use memchr::{memchr, memchr3};
use arrayvec::ArrayVec;

use crate::{id::{Id, HasId}, ap214::{Entity, superclasses_of}};

////////////////////////////////////////////////////////////////////////////////

pub type IResult<'a, U> = nom::IResult<&'a str, U, Error<&'a str>>;

/// Helper function to generate a `nom` error result
fn nom_err<'a, U>(s: &'a str, kind: nom::error::ErrorKind) -> IResult<'a, U> {
    Err(nom::Err::Error(Error::new(s, kind)))
}

/// Helper function to generate a `nom` error result with the `Alt` tag
pub fn nom_alt_err<'a, U>(s: &'a str) -> IResult<'a, U> {
    nom_err(s, ErrorKind::Alt)
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Logical(pub Option<bool>);

impl HasId for Logical {
    fn append_ids(&self, _v: &mut Vec<usize>) { /* Nothing to do here */ }
}

////////////////////////////////////////////////////////////////////////////////

pub(crate) trait Parse<'a> {
    fn parse(s: &'a str) -> IResult<'a, Self> where Self: Sized;
}

impl Parse<'_> for f64 {
    fn parse(s: &str) -> IResult<Self> {
        match fast_float2::parse_partial::<f64, _>(s) {
            Err(_) => nom_err(s, ErrorKind::Float),
            Ok((x, n)) => Ok((&s[n..], x)),
        }
    }
}

impl Parse<'_> for i64 {
    fn parse(s: &str) -> IResult<Self> {
        map_res(tuple((opt(char('-')), digit1)),
            |(sign, digits)| -> Result<i64, <i64 as std::str::FromStr>::Err> {
                let num = str::parse::<i64>(digits)?;
                if sign.is_some() {
                    Ok(-num)
                } else {
                    Ok(num)
                }
            })(s)
    }
}
impl<'a> Parse<'a> for &'a str {
    fn parse(s: &'a str) -> IResult<'a, &'a str> {
        if let Some(rest) = s.strip_prefix('$') {
            return Ok((rest, ""));
        }
        if !s.starts_with('\'') {
            return nom_err(s, ErrorKind::Char);
        }

        let bytes = s.as_bytes();
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                } else {
                    // Keep doubled apostrophes in their ISO 10303-21 encoded
                    // representation: borrowed strings cannot be unescaped
                    // without changing the generated entity field types.
                    return Ok((&s[i + 1..], &s[1..i]));
                }
            } else {
                i += 1;
            }
        }
        nom_err(s, ErrorKind::Char)
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for Vec<T> {
    fn parse(s: &'a str) -> IResult<'a, Vec<T>> {
        delimited(char('('), separated_list0(char(','), T::parse), char(')'))(s)
    }
}
impl<'a, T: Parse<'a>, const CAP: usize> Parse<'a> for ArrayVec<T, CAP> {
    fn parse(s: &'a str) -> IResult<'a, ArrayVec<T, CAP>> {
        let (mut s, _) = char('(')(s)?;
        let mut out = ArrayVec::new();
        // Based on nom's separated_list0
        let (s_, o) = match T::parse(s) {
            Err(nom::Err::Error(_)) => return Ok((s, out)),
            e => e?,
        };
        s = s_;
        out.push(o);

        loop {
            let (s_, _) = match char(',')(s) {
                Err(nom::Err::Error(_)) => break,
                e => e?,
            };
            s = s_;
            let (s_, o) = match T::parse(s) {
                Err(nom::Err::Error(_)) => break,
                e => e?,
            };
            s = s_;
            // Some exporters write more items than the bound allows, such
            // as four-component directions. Keep the first ones, as OCCT
            // does, rather than reject the file.
            let _ = out.try_push(o);
        }
        let (s, _) = char(')')(s)?;
        Ok((s, out))
    }
}
impl<'a, T: Parse<'a>> Parse<'a> for Option<T> {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map(char('$'), |_| None),
            map(T::parse, |v| Some(v))))(s)
    }
}
impl<'a> Parse<'a> for Logical {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map(tag(".T."), |_| Logical(Some(true))),
            map(tag(".F."), |_| Logical(Some(false))),
            map(tag(".U."), |_| Logical(None)),
        ))(s)
    }
}
impl<'a> Parse<'a> for bool {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        alt((
            map(tag(".T."), |_| true),
            map(tag(".F."), |_| false),
        ))(s)
    }
}
impl<'a, T> Parse<'a> for Id<T> {
    // References are the most common parameter, so this avoids nom's
    // combinator overhead; it accepts the same `#digits` and `$` forms.
    fn parse(s: &str) -> IResult<Self> {
        if let Some(rest) = s.strip_prefix('#') {
            let n = rest.bytes().take_while(u8::is_ascii_digit).count();
            if let Ok(i) = rest[..n].parse() {
                return Ok((&rest[n..], Id::new(i)));
            }
        } else if let Some(rest) = s.strip_prefix('$') {
            // NUL id deserializes to 0
            return Ok((rest, Id::empty()));
        }
        nom_alt_err(s)
    }
}

////////////////////////////////////////////////////////////////////////////////

pub(crate) trait ParseFromChunks<'a> {
    fn parse_chunks(s: &[&'a str]) -> IResult<'a, Self> where Self: Sized;
}

impl<'a, T: ParseFromChunks<'a>> Parse<'a> for T {
    fn parse(s: &'a str) -> IResult<'a, Self> {
        T::parse_chunks(&[s])
    }
}

// Simple struct so we can use param_from_chunks::<Derived> to parse a '*' or
// an entity reference (#NNN), optionally followed by a comma. Some STEP files
// use explicit DimensionalExponents references instead of '*' in complex
// entities like NAMED_UNIT.
pub struct Derived;
impl<'a> Parse<'a> for Derived {
    fn parse(s: &str) -> IResult<Self> {
        alt((
            map(char('*'), |_| Derived),
            map(preceded(char('#'), digit1), |_: &str| Derived),
        ))(s)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Parse a single attribute from a parameter list, consuming the trailing
/// comma (if this is midway through the list) or close parens (at the end)
///
/// The input is in the form of &str slices plus the index of the current slice,
/// for cases where we're splicing together multiple sets of arguments to build
/// a complete Entity.
fn check_str<'a>(s: &'a str, i: &mut usize, strs: &[&'a str]) -> &'a str {
    if s.is_empty() {
        *i += 1;
        strs.get(*i).unwrap_or(&"")
    } else {
        s
    }
}
pub(crate) fn param_from_chunks<'a, T: Parse<'a>>(
    last: bool, s: &'a str,
    i: &mut usize, strs: &[&'a str]) -> IResult<'a, T>
{
    let s = check_str(s, i, strs);
    let (s, out) = T::parse(s)?;
    let s = check_str(s, i, strs);
    let (s, _) = char(if last { ')'} else { ',' })(s)?;
    Ok((check_str(s, i, strs), out))
}

pub(crate) fn parse_enum_tag(s: &str) -> IResult<&str> {
    delimited(char('.'),
              nom::bytes::complete::take_while(
                  |c: char| c == '_' ||
                            c.is_ascii_uppercase() ||
                            c.is_ascii_digit()),
              char('.'))(s)
}

////////////////////////////////////////////////////////////////////////////////

/// Returns the text before the first `(`, like nom's `take_until("(")`
/// without building a substring searcher for every entity.
pub(crate) fn take_until_paren(s: &str) -> IResult<'_, &str> {
    match memchr(b'(', s.as_bytes()) {
        Some(i) => Ok((&s[i..], &s[..i])),
        None => nom_err(s, ErrorKind::TakeUntil),
    }
}

pub(crate) fn parse_entity_decl(s: &str) -> IResult<'_, (usize, Entity<'_>)> {
    map(tuple((Id::<()>::parse, char('='), Entity::parse)),
        |(i, _, e)| (i.0, e))(s)
}

pub(crate) fn parse_entity_fallback(s: &str) -> IResult<'_, (usize, Entity<'_>)> {
    map(Id::<()>::parse, |i| (i.0, Entity::_FailedToParse))(s)
}

pub(crate) fn parse_complex_mapping(s: &str) -> IResult<Entity> {
    // Sub-entities as (name, name plus open parens, argument string), used
    // to figure out the tree and construct it. A repeated name keeps its last
    // occurrence. There are only a few parts, so lookups are linear scans.
    let mut parts: Vec<(&str, &str, &str)> = Vec::new();
    let bstr = s.as_bytes();
    let mut depth = 0;
    let mut index = 0;
    let mut args_start = 0;
    let mut name: &str = "";
    let mut name_tag: &str = "";
    loop {
        let next = match memchr3(b'(', b')', b'\'', &bstr[index..]) {
            Some(i) => i,
            None => return nom_err(s, ErrorKind::Alt),
        };
        match bstr[index + next] {
            b'(' => {
                if depth == 1 {
                    let name_slice = &bstr[index..(index + next)];
                    name = std::str::from_utf8(name_slice)
                        .expect("Could not convert back to name");
                    args_start = index + next + 1;
                    let name_tag_slice = &bstr[index..(index + next + 1)];
                    name_tag = std::str::from_utf8(name_tag_slice)
                        .expect("Could not convert tag back to name");
                }
                depth += 1;
            },
            b')' => {
                depth -= 1;
                if depth == 1 {
                    let arg_slice = &bstr[args_start..(index + next)];
                    let args = std::str::from_utf8(arg_slice)
                        .expect("Could not convert args");
                    match parts.iter_mut().find(|p| p.0 == name) {
                        Some(p) => *p = (name, name_tag, args),
                        None => parts.push((name, name_tag, args)),
                    }
                } else if depth == 0 {
                    break;
                }
            },
            b'\'' => {
                // TODO: handle escaped quotes
                let j = match memchr(b'\'', &bstr[(index + next + 1)..]) {
                    Some(j) => j,
                    None => return nom_err(s, ErrorKind::Char),
                };
                index += j + 1;
            }
            c => panic!("Invalid char: {}", c),
        }
        index += next + 1;
    }
    // Filter out the list of subclasses to those which aren't a parent of
    // another item in the set; these are our potential leafs.
    let supers: Vec<&[&str]> = parts.iter().map(|p| superclasses_of(p.0)).collect();
    let mut potential_leafs: Vec<(&str, &str)> = parts.iter()
        .filter(|p| !supers.iter().any(|s| s.contains(&p.0)))
        // Eliminate any leaf with no arguments, since they're just addding
        // bonus constraints (which we don't handle anyways)
        .filter(|p| p.2 != "")
        .map(|p| (p.0, p.1))
        .collect();

    // Sort potential leafs so that ComplexEntity is deterministic and we can
    // match against it later
    potential_leafs.sort();

    // At this point, we'll build up argument strings by splicing together bits
    // of arguments from the existing string (to make lifetimes happy), then
    // parse into leaf entities.
    let mut leaf_entities = Vec::with_capacity(potential_leafs.len());
    for (leaf, leaf_tag) in potential_leafs.into_iter() {
        let mut chain = vec![leaf];
        loop {
            let sup = superclasses_of(chain.last().unwrap());
            match sup.len() {
                0 => break,
                1 => chain.push(sup[0]),
                _ => return nom_err(s, ErrorKind::LengthValue), // TODO: error
            }
        }
        let mut new_decl: Vec<&str> = vec![leaf_tag];
        for c in chain.iter().rev() {
            // Records are parsed in parallel, so a missing supertype must be
            // an error rather than a panic that masks an earlier error.
            let args = match parts.iter().find(|p| p.0 == *c) {
                Some(p) => p.2,
                None => return nom_err(s, ErrorKind::Verify),
            };
            if !args.is_empty() {
                new_decl.push(args);
                new_decl.push(if *c == leaf { &")" } else { &"," });
            }
        }
        leaf_entities.push(Entity::parse_chunks(&new_decl)?.1)
    }
    // At this point, we assume that there's nothing left to parse, so we
    // return an empty string for the 'remaining' text
    if leaf_entities.len() == 1 {
        Ok(("", leaf_entities.pop().unwrap()))
    } else {
        Ok(("", Entity::ComplexEntity(leaf_entities)))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_uses_iso_unknown_literal() {
        assert_eq!(Logical::parse(".U."), Ok(("", Logical(None))));
        assert!(Logical::parse(".UNKNOWN.").is_err());
    }

    #[test]
    fn strings_allow_doubled_apostrophes() {
        assert_eq!(<&str>::parse("'Don''t panic',next"),
                   Ok((",next", "Don''t panic")));
        assert!(<&str>::parse("'unterminated").is_err());
    }

    #[test]
    fn bounded_lists_keep_their_first_items() {
        // Some exporters write four-component directions.
        let (rest, ratios) = ArrayVec::<f64, 3>::parse("(0.,0.,1.,0.),next").unwrap();
        assert_eq!((rest, ratios.as_slice()), (",next", &[0., 0., 1.][..]));
    }

    #[test]
    fn test_parse_entity_decl() {
        parse_entity_decl("#3=SHAPE_DEFINITION_REPRESENTATION(#4,#10);").unwrap();
        parse_entity_decl("#38463=ADVANCED_FACE('',(#38464),#38475,.F.);").unwrap();
        parse_entity_decl("#395359=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-007),#395356,'distance_accuracy_value','confusion accuracy');").unwrap();
        parse_entity_decl("#1632=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));").unwrap();
    }
}
