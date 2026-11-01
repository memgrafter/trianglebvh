//! Point-triangle distance and closest point, with every float32 operation and its rounding written out.
//!
//! Rust never fuses a * b + c by itself, so each fused multiply-add here is an explicit `mul_add` (one rounding), and
//! every other operation rounds on its own. The operation order was fitted bit for bit to black-box probes of cubvh
//! (see the README, "Clean-room record"); the methods are published ones:
//! - distance: outside the triangle, the minimum over the three edges of the point-segment distance; inside, the
//!   distance to the plane (the "unsigned distance to a triangle" of I. Quilez, and Ericson, Real-Time Collision
//!   Detection, 5.1.2);
//! - closest point: projection on the plane, or the closest point on the nearest edge;
//! - barycentric coordinates: Ericson 3.4 (Cramer's rule on the edge dot products).

pub type V3 = [f32; 3];

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[inline(always)]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// x0*y0 + (x1*y1 + x2*y2), the inner sum fused on x1*y1, the outer on x0*y0.
#[inline(always)]
pub fn dot_r(x: V3, y: V3) -> f32 {
    fma(x[0], y[0], fma(x[1], y[1], x[2] * y[2]))
}

/// x0*y0 + (x1*y1 + x2*y2), the inner sum fused on x2*y2, the outer on x0*y0.
#[inline(always)]
pub fn dot_q(x: V3, y: V3) -> f32 {
    fma(x[0], y[0], fma(x[2], y[2], x[1] * y[1]))
}

/// Cross product; each component l_i*r_j - l_j*r_i fused on the first product.
#[inline(always)]
pub fn cross(l: V3, r: V3) -> V3 {
    [
        fma(l[1], r[2], -(l[2] * r[1])),
        fma(l[2], r[0], -(l[0] * r[2])),
        fma(l[0], r[1], -(l[1] * r[0])),
    ]
}

/// t < 0 -> 0, t > 1 -> 1, otherwise t (a NaN passes through).
#[inline(always)]
fn clamp01(t: f32) -> f32 {
    if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t
    }
}

/// min keeping the first argument unless the second is smaller (a NaN first argument is kept, a NaN second dropped).
#[inline(always)]
fn min_first(x: f32, y: f32) -> f32 {
    if y < x {
        y
    } else {
        x
    }
}

/// Squared distance from p to segment e0 -> e1.
#[inline(always)]
fn segment_d2(p: V3, e0: V3, e1: V3) -> f32 {
    let e = sub(e1, e0);
    let pe = sub(p, e0);
    let t = clamp01(dot_r(e, pe) / dot_r(e, e));
    let x = [fma(e[0], t, -pe[0]), fma(e[1], t, -pe[1]), fma(e[2], t, -pe[2])];
    dot_r(x, x)
}

/// The normal used by both the inside test and the plane distance: (b - a) x (a - c).
#[inline(always)]
fn normal(a: V3, b: V3, c: V3) -> V3 {
    cross(sub(b, a), sub(a, c))
}

/// Is the projection of p inside the triangle: none of the three edge tests has its sign bit set (so -0.0 counts as
/// outside).
#[inline(always)]
pub fn inside(p: V3, a: V3, b: V3, c: V3) -> bool {
    let ba = sub(b, a);
    let cb = sub(c, b);
    let ac = sub(a, c);
    let nor = cross(ba, ac);
    let s0 = dot_r(cross(ba, nor), sub(p, a));
    let s1 = dot_r(cross(cb, nor), sub(p, b));
    let s2 = dot_r(cross(ac, nor), sub(p, c));
    !(s0.is_sign_negative() || s1.is_sign_negative() || s2.is_sign_negative())
}

/// Squared distance from p to the triangle (a NaN for some degenerate triangles: such a triangle never wins).
#[inline(always)]
pub fn dist2(p: V3, a: V3, b: V3, c: V3) -> f32 {
    if inside(p, a, b, c) {
        let nor = normal(a, b, c);
        let h = dot_q(nor, sub(p, a));
        (h * h) / dot_q(nor, nor)
    } else {
        min_first(min_first(segment_d2(p, a, b), segment_d2(p, b, c)), segment_d2(p, c, a))
    }
}

/// Barycentric coordinates (u, v, w) of q: v and w by Cramer's rule, u = 1 - v - w in double precision.
#[inline(always)]
pub fn barycentric(q: V3, a: V3, b: V3, c: V3) -> V3 {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(q, a);
    let d00 = dot_r(v0, v0);
    let d01 = dot_r(v0, v1);
    let d11 = dot_r(v1, v1);
    let d20 = dot_r(v2, v0);
    let d21 = dot_r(v2, v1);
    let den = fma(d00, d11, -(d01 * d01));
    let v = fma(d11, d20, -(d01 * d21)) / den;
    let w = fma(d00, d21, -(d01 * d20)) / den;
    let u = ((1.0f64 - v as f64) - w as f64) as f32;
    [u, v, w]
}

/// Closest point on the triangle to p, as barycentric coordinates: the projection q0 on the plane when q0 is inside
/// (by the signs of the three sub-triangle areas), otherwise the closest point on the nearest edge to q0 (edges a->b,
/// b->c, c->a; the first wins a tie).
pub fn closest_uvw(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let nor = normal(a, b, c);
    let nn = dot_r(nor, nor).sqrt();
    let nh = [nor[0] / nn, nor[1] / nn, nor[2] / nn];
    let hh = dot_r(nh, sub(p, a));
    let q0 = [fma(-nh[0], hh, p[0]), fma(-nh[1], hh, p[1]), fma(-nh[2], hh, p[2])];
    // inside: the three sub-triangle areas (q0, b, c), (q0, c, a), (q0, a, b) along the normal have one sign
    let ua = dot_r(nor, cross(sub(b, q0), sub(c, q0)));
    let va = dot_r(nor, cross(sub(c, q0), sub(a, q0)));
    let wa = dot_r(nor, cross(sub(a, q0), sub(b, q0)));
    if (ua <= 0.0 && va <= 0.0 && wa <= 0.0) || (ua >= 0.0 && va >= 0.0 && wa >= 0.0) {
        return barycentric(q0, a, b, c);
    }
    let mut best = f32::INFINITY;
    let mut q = q0;
    // the squared edge length is summed in a different order for the first edge than for the other two (fitted)
    for (k, (e0, e1)) in [(a, b), (b, c), (c, a)].into_iter().enumerate() {
        let e = sub(e1, e0);
        let pe = sub(q0, e0);
        let ee = if k == 0 { dot_r(e, e) } else { dot_q(e, e) };
        let t = clamp01(dot_r(e, pe) / ee);
        let qe = [fma(e[0], t, e0[0]), fma(e[1], t, e0[1]), fma(e[2], t, e0[2])];
        let dd = sub(q0, qe);
        let d = dot_r(dd, dd);
        if d < best {
            best = d;
            q = qe;
        }
    }
    barycentric(q, a, b, c)
}
