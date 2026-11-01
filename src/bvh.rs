//! The tree and the closest-triangle query, in the order that decides which face wins an exact tie.
//!
//! Every behaviour below was found by black-box probes of cubvh (README, "Clean-room record"):
//! - build: four children per node, a leaf at 8 triangles or fewer. A node's range is split at its median along the
//!   axis of largest centroid variance (libstdc++ nth_element order, `select.rs`), then each half again at its own
//!   median and axis. The children are, in order: first quarter, third quarter, second quarter, fourth quarter.
//! - centroid: ((a + b) + c) / 3 in float32. Variance per axis: float32 sums in the range's current order, mean =
//!   sum * (1 / n), variance = sum of squared deviations; the first axis with the largest variance wins.
//! - query: a stack. A leaf updates the best when a triangle's squared distance is <= the best (so the last equal
//!   triangle wins). An inner node orders its children by box distance with a five-comparator sorting network
//!   (pairs 0-1, 2-3, 0-2, 1-3, 1-2; swap when the second is <= the first) and pushes them farthest first, each only if
//!   its box distance is <= the best at that moment. The best starts at 1000^2; a point with no triangle within it
//!   gets distance 0 and face 0.

use crate::select::nth_element;
use crate::simd;
use crate::tri::{self, V3};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const LEAF_SIZE: usize = 8;
pub const MAX_DIST2: f32 = 1.0e6;

#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    /// leaf: first triangle in `tris`; inner: first child in `nodes` (children are consecutive)
    first: u32,
    /// leaf: number of triangles; inner: 0 (four children)
    count: u32,
}

/// An inner node for the query: the boxes of its four children ([axis][child]) and the children (LEAF_BIT set: an
/// index into `leaves`, otherwise into `inner`).
#[derive(Clone, Copy)]
struct Inner {
    lo: [[f32; 4]; 3],
    hi: [[f32; 4]; 3],
    child: [u32; 4],
}

const LEAF_BIT: u32 = 1 << 31;

pub struct Bvh {
    inner: Vec<Inner>,
    /// per leaf: first triangle (position in `ids`; its 9 * count floats start at packed[9 * first]) and count
    leaves: Vec<(u32, u32)>,
    /// the triangles leaf by leaf, component-major inside a leaf: a.x of every triangle, then a.y, ..., then c.z
    packed: Vec<f32>,
    /// input face index of each triangle position
    ids: Vec<u32>,
    root: u32,
    /// triangle 0 of the input (for a point with no triangle in range)
    tri0: [V3; 3],
    simd: bool,
}

fn centroid(t: &[V3; 3]) -> V3 {
    [(t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0, (t[0][2] + t[1][2] + t[2][2]) / 3.0]
}

/// Axis of the largest variance of the centroids of `ids` (in their current order).
fn split_axis(ids: &[u32], cent: &[V3]) -> usize {
    let mut sum = [0.0f32; 3];
    for &i in ids {
        let c = cent[i as usize];
        for k in 0..3 {
            sum[k] += c[k];
        }
    }
    let inv = 1.0f32 / ids.len() as f32;
    let mean = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
    let mut var = [0.0f32; 3];
    for &i in ids {
        let c = cent[i as usize];
        for k in 0..3 {
            let d = c[k] - mean[k];
            var[k] += d * d;
        }
    }
    let mut axis = 0;
    for k in 1..3 {
        if var[k] > var[axis] {
            axis = k;
        }
    }
    axis
}

/// Median split of `ids` along its variance axis; returns the split position.
fn split(ids: &mut [u32], cent: &[V3]) -> usize {
    let axis = split_axis(ids, cent);
    let m = ids.len() / 2;
    nth_element(ids, m, &|a: &u32, b: &u32| cent[*a as usize][axis] < cent[*b as usize][axis]);
    m
}

fn bounds(t: &[V3; 3]) -> (V3, V3) {
    let mut lo = t[0];
    let mut hi = t[0];
    for v in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    (lo, hi)
}

/// A subtree built on its own (for parallel builds): nodes with local indices, root first.
struct Sub {
    nodes: Vec<Node>,
}

/// Build the subtree over ids[lo..hi] (positions in the final triangle order). Node 0 of the result is its root.
fn build_sub(ids: &mut [u32], base: usize, tris: &[[V3; 3]], cent: &[V3], threads: usize) -> Sub {
    let mut nodes = vec![Node { lo: [0.0; 3], hi: [0.0; 3], first: 0, count: 0 }];
    build_into(&mut nodes, 0, ids, base, tris, cent, threads);
    Sub { nodes }
}

fn build_into(nodes: &mut Vec<Node>, at: usize, ids: &mut [u32], base: usize, tris: &[[V3; 3]], cent: &[V3], threads: usize) {
    let n = ids.len();
    if n <= LEAF_SIZE {
        let (mut lo, mut hi) = bounds(&tris[ids[0] as usize]);
        for &i in &ids[1..] {
            let (l, h) = bounds(&tris[i as usize]);
            for k in 0..3 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
            }
        }
        nodes[at] = Node { lo, hi, first: base as u32, count: n as u32 };
        return;
    }
    let m = split(ids, cent);
    let m1 = split(&mut ids[..m], cent);
    let m2 = m + split(&mut ids[m..], cent);
    // children in the order: first, third, second, fourth quarter
    let ranges = [(0, m1), (m, m2), (m1, m), (m2, n)];
    let first = nodes.len();
    for _ in 0..4 {
        nodes.push(Node { lo: [0.0; 3], hi: [0.0; 3], first: 0, count: 0 });
    }
    if threads >= 4 && n > 65536 {
        // the four quarters are disjoint: build them on their own threads, then splice
        let (q01, q23) = ids.split_at_mut(m);
        let (qa, qb) = q01.split_at_mut(m1);
        let (qc, qd) = q23.split_at_mut(m2 - m);
        let t = threads / 4;
        let subs: Vec<Sub> = std::thread::scope(|s| {
            let h = [
                s.spawn(|| build_sub(qa, base, tris, cent, t)),
                s.spawn(|| build_sub(qc, base + m, tris, cent, t)),
                s.spawn(|| build_sub(qb, base + m1, tris, cent, t)),
                s.spawn(|| build_sub(qd, base + m2, tris, cent, t)),
            ];
            h.into_iter().map(|j| j.join().unwrap()).collect()
        });
        for (k, sub) in subs.into_iter().enumerate() {
            let off = nodes.len() - 1; // local node i (i >= 1) goes to off + i; local 0 goes to first + k
            let mut root = sub.nodes[0];
            if root.count == 0 {
                root.first = (root.first as usize + off) as u32;
            }
            nodes[first + k] = root;
            for nd in &sub.nodes[1..] {
                let mut nd = *nd;
                if nd.count == 0 {
                    nd.first = (nd.first as usize + off) as u32;
                }
                nodes.push(nd);
            }
        }
    } else {
        for (k, &(s, e)) in ranges.iter().enumerate() {
            build_into(nodes, first + k, &mut ids[s..e], base + s, tris, cent, 1);
        }
    }
    let mut lo = nodes[first].lo;
    let mut hi = nodes[first].hi;
    for k in 1..4 {
        for c in 0..3 {
            lo[c] = lo[c].min(nodes[first + k].lo[c]);
            hi[c] = hi[c].max(nodes[first + k].hi[c]);
        }
    }
    nodes[at] = Node { lo, hi, first: first as u32, count: 0 };
}

/// Squared distance from p to a box: per axis max(lo - p, p - hi, 0), summed as d0^2 + (d1^2 + d2^2) with the
/// same fused multiply-adds as the triangle code.
#[inline(always)]
fn box_d2(lo: V3, hi: V3, p: V3) -> f32 {
    let mut d = [0.0f32; 3];
    for k in 0..3 {
        d[k] = (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0);
    }
    tri::dot_r(d, d)
}

/// The query structures from the built tree: inner nodes with their children's boxes, leaves, packed triangles.
fn convert(nodes: &[Node], at: usize, tris: &[[V3; 3]], inner: &mut Vec<Inner>, leaves: &mut Vec<(u32, u32)>, packed: &mut [f32]) -> u32 {
    let n = nodes[at];
    if n.count > 0 {
        let (first, count) = (n.first as usize, n.count as usize);
        for lane in 0..count {
            let t = &tris[first + lane];
            for v in 0..3 {
                for k in 0..3 {
                    packed[9 * first + (v * 3 + k) * count + lane] = t[v][k];
                }
            }
        }
        leaves.push((n.first, n.count));
        return (leaves.len() - 1) as u32 | LEAF_BIT;
    }
    let me = inner.len();
    inner.push(Inner { lo: [[0.0; 4]; 3], hi: [[0.0; 4]; 3], child: [0; 4] });
    let c = n.first as usize;
    let mut node = Inner { lo: [[0.0; 4]; 3], hi: [[0.0; 4]; 3], child: [0; 4] };
    for j in 0..4 {
        for k in 0..3 {
            node.lo[k][j] = nodes[c + j].lo[k];
            node.hi[k][j] = nodes[c + j].hi[k];
        }
        node.child[j] = convert(nodes, c + j, tris, inner, leaves, packed);
    }
    inner[me] = node;
    me as u32
}

impl Bvh {
    pub fn build(vertices: &[V3], faces: &[[u32; 3]], threads: usize) -> Bvh {
        let nf = faces.len();
        let tris_in: Vec<[V3; 3]> = faces.iter().map(|f| [vertices[f[0] as usize], vertices[f[1] as usize], vertices[f[2] as usize]]).collect();
        let tri0 = if nf > 0 { tris_in[0] } else { [[0.0; 3]; 3] };
        let mut ids: Vec<u32> = (0..nf as u32).collect();
        let mut nodes = vec![Node { lo: [0.0; 3], hi: [0.0; 3], first: 0, count: 0 }];
        if nf > 0 {
            let cent: Vec<V3> = tris_in.iter().map(centroid).collect();
            build_into(&mut nodes, 0, &mut ids, 0, &tris_in, &cent, threads.max(1));
        }
        let tris: Vec<[V3; 3]> = ids.iter().map(|&i| tris_in[i as usize]).collect();
        drop(tris_in);
        let (mut inner, mut leaves, mut packed) = (Vec::new(), Vec::new(), vec![0.0f32; 9 * nf]);
        let root = if nf > 0 { convert(&nodes, 0, &tris, &mut inner, &mut leaves, &mut packed) } else { 0 };
        Bvh { inner, leaves, packed, ids, root, tri0, simd: simd::available() && std::env::var("TRIANGLEBVH_SIMD").map_or(true, |v| v != "0") }
    }

    fn triangle(&self, leaf: usize, lane: usize) -> [V3; 3] {
        let (first, count) = self.leaves[leaf];
        let base = 9 * first as usize;
        let c = count as usize;
        let g = |k: usize| self.packed[base + k * c + lane];
        [[g(0), g(1), g(2)], [g(3), g(4), g(5)], [g(6), g(7), g(8)]]
    }

    /// (squared distance, face, leaf, lane) of the closest triangle, or None when none is within MAX_DIST2.
    fn closest(&self, p: V3) -> Option<(f32, u32, usize, usize)> {
        if self.ids.is_empty() {
            return None;
        }
        let mut best = MAX_DIST2;
        let mut found: Option<(u32, usize, usize)> = None;
        // (node, its box distance when pushed); an inner node popped with a box distance above the best is skipped:
        // its children's boxes lie inside its box, so none of them could be pushed (box distances only grow)
        let mut stack = [(0u32, 0.0f32); 256];
        stack[0] = (self.root, 0.0);
        let mut sp = 1usize;
        let mut d8 = [0.0f32; 8];
        while sp > 0 {
            sp -= 1;
            let (id, bd) = stack[sp];
            if id & LEAF_BIT != 0 {
                let leaf = (id & !LEAF_BIT) as usize;
                let (first, count) = self.leaves[leaf];
                let (first, count) = (first as usize, count as usize);
                if self.simd {
                    // SAFETY: simd is set only when the CPU has AVX2 and FMA; the leaf holds 9 * count floats
                    #[cfg(target_arch = "x86_64")]
                    unsafe { simd::dist2_8(self.packed.as_ptr().add(9 * first), count, p, &mut d8) };
                } else {
                    for lane in 0..count {
                        let t = self.triangle(leaf, lane);
                        d8[lane] = tri::dist2(p, t[0], t[1], t[2]);
                    }
                }
                for lane in 0..count {
                    if d8[lane] <= best {
                        best = d8[lane];
                        found = Some((self.ids[first + lane], leaf, lane));
                    }
                }
                continue;
            }
            if bd > best {
                continue;
            }
            let n = &self.inner[id as usize];
            let mut d = [0.0f32; 4];
            if self.simd {
                // SAFETY: as above
                #[cfg(target_arch = "x86_64")]
                unsafe { d = simd::box_d2_4(&n.lo, &n.hi, p) };
            } else {
                for j in 0..4 {
                    d[j] = box_d2([n.lo[0][j], n.lo[1][j], n.lo[2][j]], [n.hi[0][j], n.hi[1][j], n.hi[2][j]], p);
                }
            }
            let mut o = n.child;
            for (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
                if d[j] <= d[i] {
                    d.swap(i, j);
                    o.swap(i, j);
                }
            }
            for k in (0..4).rev() {
                if d[k] <= best {
                    assert!(sp < stack.len(), "tree deeper than the query stack");
                    stack[sp] = (o[k], d[k]);
                    sp += 1;
                }
            }
        }
        found.map(|(id, leaf, lane)| (best, id, leaf, lane))
    }

    /// Unsigned distance, face index and (optionally) barycentric coordinates of the closest point, per query point.
    pub fn unsigned_distance(&self, pts: &[V3], dist: &mut [f32], face: &mut [i64], mut uvw: Option<&mut [V3]>, threads: usize) {
        let n = pts.len();
        let next = AtomicUsize::new(0);
        const BLOCK: usize = 4096;
        let dist_p = SendPtr(dist.as_mut_ptr());
        let face_p = SendPtr(face.as_mut_ptr());
        let uvw_p = SendPtr(uvw.as_mut().map_or(std::ptr::null_mut(), |u| u.as_mut_ptr()));
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                let next = &next;
                let (dist_p, face_p, uvw_p) = (&dist_p, &face_p, &uvw_p);
                s.spawn(move || loop {
                    let start = next.fetch_add(BLOCK, Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    for i in start..(start + BLOCK).min(n) {
                        let p = pts[i];
                        let (d, f, t) = match self.closest(p) {
                            Some((d2, id, leaf, lane)) => (d2.sqrt(), id as i64, self.triangle(leaf, lane)),
                            None => (0.0, 0, self.tri0),
                        };
                        // SAFETY: each index i is written by exactly one thread (blocks are claimed atomically)
                        unsafe {
                            *dist_p.0.add(i) = d;
                            *face_p.0.add(i) = f;
                            if !uvw_p.0.is_null() {
                                *uvw_p.0.add(i) = tri::closest_uvw(p, t[0], t[1], t[2]);
                            }
                        }
                    }
                });
            }
        });
    }

    /// Every face whose squared distance to p equals the minimum over all faces (analysis of exact ties).
    pub fn ties(&self, p: V3) -> Vec<u32> {
        let mut all = Vec::with_capacity(self.ids.len());
        for (leaf, &(first, count)) in self.leaves.iter().enumerate() {
            for lane in 0..count as usize {
                let t = self.triangle(leaf, lane);
                all.push((tri::dist2(p, t[0], t[1], t[2]), self.ids[first as usize + lane]));
            }
        }
        let best = all.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        let mut out: Vec<u32> = all.into_iter().filter(|x| x.0 == best).map(|x| x.1).collect();
        out.sort_unstable();
        out
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Host memory held by the tree, in bytes.
    pub fn bytes(&self) -> usize {
        self.inner.capacity() * std::mem::size_of::<Inner>() + self.leaves.capacity() * 8 + self.packed.capacity() * 4 + self.ids.capacity() * 4
    }
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
