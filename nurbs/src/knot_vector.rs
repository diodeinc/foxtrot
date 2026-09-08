use std::convert::TryInto;

use smallvec::{smallvec, SmallVec};
use std::mem::swap;

use crate::VecF;

#[derive(Debug, Clone)]
pub struct KnotVector {
    /// Knot positions
    U: VecF,

    /// Degree of the knot vector
    p: usize,
}

impl KnotVector {
    /// Constructs a new knot vector of over
    pub fn from_multiplicities(p: usize, knots: &[f64], multiplicities: &[usize]) -> Self {
        assert!(knots.len() == multiplicities.len());
        let U = knots.iter().zip(multiplicities.iter())
            .flat_map(|(k, m)| std::iter::repeat(*k).take(*m))
            .collect();
        Self { U, p }
    }

    /// For basis functions of order `p + 1`, finds the span in the knot vector
    /// that is relevant for position `u`.
    ///
    /// ALGORITHM A2.1
    pub fn find_span(&self, u: f64) -> usize {
        // U is [u_0, u_1, ... u_m]
        let m = self.len() - 1;
        let n = m - (self.p + 1); // max basis index

        if u >= self[n + 1] {
            return n;
        } else if u <= self[self.p] {
            return self.p;
        }
        let mut low = self.p;
        let mut high = n + 1;
        let mut mid = (low + high) / 2;
        while u < self[mid] || u >= self[mid + 1] {
            if u < self[mid] {
                high = mid;
            } else {
                low = mid;
            }
            mid = (low + high) / 2;
        }
        mid
    }

    /// Nonempty active spans incident to a parameter, including both sides
    /// of an interior knot but not the inactive exterior knot intervals.
    pub fn spans_at(&self, u: f64) -> impl Iterator<Item = usize> {
        let right = self.find_span(u);
        let left = if self[right] == u {
            (self.p..right).rev().find(|&i| self[i] < u)
        } else { None };
        std::iter::once(right).chain(left)
    }

    pub fn degree(&self) -> usize {
        self.p
    }
    pub fn len(&self) -> usize {
        self.U.len()
    }
    pub fn min_t(&self) -> f64 {
        self[self.p]
    }
    pub fn max_t(&self) -> f64 {
        self[self.len() - 1 - self.p]
    }

    /// Computes non-vanishing basis functions of order `p + 1` at point `u`.
    ///
    /// ALGORITHM A2.2
    pub fn basis_funs(&self, u: f64) -> VecF {
        let i = self.find_span(u);
        self.basis_funs_for_span(i, u)
    }

    // Inner implementation of basis_funs
    pub fn basis_funs_for_span(&self, i: usize, u: f64) -> VecF {
        match self.p {
            1 => self.basis_positions::<1>(i, u),
            2 => self.basis_positions::<2>(i, u),
            3 => self.basis_positions::<3>(i, u),
            _ => self.basis_positions::<0>(i, u),
        }
    }

    fn basis_positions<const P: usize>(&self, i: usize, u: f64) -> VecF {
        let p = if P == 0 { self.p } else { P };
        let mut N: VecF = smallvec![0.0; p + 1];

        let mut left: VecF = smallvec![0.0; p + 1];
        let mut right: VecF = smallvec![0.0; p + 1];
        N[0] = 1.0;
        for j in 1..=p {
            left[j] = u - self[i + 1 - j];
            right[j] = self[i + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                let temp: f64 = N[r] / (right[r + 1] + left[j - r]);
                N[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            N[j] = saved;
        }
        N
    }

    /// Computes the derivatives (up to and including the `nth` derivative) of non-vanishing
    /// basis functions of order `p + 1` at point `u`.
    ///
    /// ALGORITHM A2.3
    /// Rows are contiguous: ders[k * (p + 1) + j] is the kth derivative
    /// of the function `N_{i-p+j, p}` at `u`.
    pub fn basis_funs_derivs(&self, u: f64, n: usize) -> SmallVec<[f64; 24]> {
        let i = self.find_span(u);
        self.basis_funs_derivs_for_span(i, u, n)
    }

    pub fn basis_funs_derivs_for_span(&self, i: usize, u: f64, n: usize) -> SmallVec<[f64; 24]> {
        // Position-only queries need no derivative triangle or recurrence scratch.
        if n == 0 {
            return self.basis_funs_for_span(i, u).into_iter().collect();
        }
        match (self.p, n) {
            (1, 1) => self.basis_derivatives::<1, 1>(i, u, n),
            (2, 1) => self.basis_derivatives::<2, 1>(i, u, n),
            (2, 2) => self.basis_derivatives::<2, 2>(i, u, n),
            (3, 1) => self.basis_derivatives::<3, 1>(i, u, n),
            (3, 2) => self.basis_derivatives::<3, 2>(i, u, n),
            _ => self.basis_derivatives::<0, 0>(i, u, n),
        }
    }

    fn basis_derivatives<const P: usize, const N: usize>(&self, i: usize, u: f64, n: usize) -> SmallVec<[f64; 24]> {
        let p = if P == 0 { self.p } else { P };
        let n = if N == 0 { n } else { N };
        // Keep common degrees and inverse-projection derivatives inline; higher
        // degrees and derivative orders spill without changing the algorithm.
        // The square basis table is contiguous, including when it spills.
        let width = p + 1;
        let mut scratch: SmallVec<[f64; 96]> = smallvec![0.0; width * (width + 4)];
        let (ndu, scratch) = scratch.split_at_mut(width * width);
        let (left, scratch) = scratch.split_at_mut(width);
        let (right, scratch) = scratch.split_at_mut(width);
        let (a0, a1) = scratch.split_at_mut(width);
        let a = [a0, a1];
        let knots = self.U.as_slice();

        let mut result: SmallVec<[f64; 24]> = smallvec![0.0; width * (n + 1)];
        let ders = result.as_mut_slice();

        ndu[0] = 1.0;
        for j in 1..=p {
            left[j] = u - knots[i + 1 - j];
            right[j] = knots[i + j] - u;
            let mut saved = 0.0;
            for r in 0..j {
                ndu[j * width + r] = right[r + 1] + left[j - r];
                let temp = ndu[r * width + j - 1] / ndu[j * width + r];

                ndu[r * width + j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j * width + j] = saved;
        }
        for j in 0..=p {
            ders[j] = ndu[j * width + p];
        }
        for r in 0..=p {
            let mut s1 = 0;
            let mut s2 = 1;
            a[0][0] = 1.0;
            for k in 1..=n {
                let aus = |i: i32| -> usize {
                    i.try_into().expect("Could not convert to usize")
                };
                let mut d = 0.0;
                let rk = (r as i32) - (k as i32);
                let pk = (p as i32) - (k as i32);
                if r >= k {
                    a[s2][0] = a[s1][0] / ndu[aus(pk + 1) * width + rk as usize];
                    d = a[s2][0] * ndu[aus(rk) * width + aus(pk)];
                }
                let j1 = aus(if rk >= -1 { 1 } else { -rk });
                let j2 = aus(if r as i32 - 1 <= pk as i32 {
                    k as i32 - 1
                } else {
                    p as i32 - r as i32
                });

                for j in j1..=j2 {
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[aus(pk + 1) * width + aus(rk + j as i32)];
                    d += a[s2][j] * ndu[aus(rk + j as i32) * width + aus(pk)];
                }
                if r as i32 <= pk {
                    a[s2][k] = -a[s1][k - 1] / ndu[aus(pk + 1) * width + r];
                    d += a[s2][k] * ndu[r * width + aus(pk)];
                }
                ders[k * width + r] = d;
                swap(&mut s1, &mut s2);
            }
        }

        let mut r = p;
        for k in 1..=n {
            for j in 0..=p {
                ders[k * width + j] *= r as f64;
            }
            r *= p - k;
        }
        result
    }
}

impl std::ops::Index<usize> for KnotVector {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.U[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_reproduce_affine_curves_at_low_and_high_degree() {
        for (degree, order) in [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
            (3, 0), (3, 1), (3, 2), (9, 0), (9, 4)] {
            let knots = KnotVector::from_multiplicities(degree, &[0., 1.], &[degree + 1; 2]);
            for u in [0., 0.125, 0.5, 0.875, 1.] {
                for (k, basis) in knots.basis_funs_derivs(u, order).chunks_exact(degree + 1).enumerate() {
                    let constant: f64 = basis.iter().sum();
                    let affine: f64 = basis.iter().enumerate()
                        .map(|(j, value)| value * j as f64 / degree as f64).sum();
                    assert!((constant - if k == 0 { 1. } else { 0. }).abs() < 1e-9);
                    let expected = match k { 0 => u, 1 => 1., _ => 0. };
                    assert!((affine - expected).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn incident_spans_skip_repetitions_and_inactive_exterior_intervals() {
        let knots = KnotVector::from_multiplicities(3, &[0., 0.5, 1.], &[4, 3, 4]);
        assert_eq!(knots.spans_at(0.).collect::<Vec<_>>(), [3]);
        assert_eq!(knots.spans_at(0.25).collect::<Vec<_>>(), [3]);
        assert_eq!(knots.spans_at(0.5).collect::<Vec<_>>(), [6, 3]);
        assert_eq!(knots.spans_at(1.).collect::<Vec<_>>(), [6]);
        let knots = KnotVector::from_multiplicities(3, &[-3., -2., -1., 0., 0.5, 1., 2., 3., 4.], &[1; 9]);
        assert_eq!(knots.spans_at(0.).collect::<Vec<_>>(), [3]);
        assert_eq!(knots.spans_at(1.).collect::<Vec<_>>(), [4]);
    }
}
