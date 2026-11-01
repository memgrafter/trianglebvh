//! The squared distance of `tri::dist2` for the up-to-8 triangles of a leaf at once (AVX2 + FMA, one triangle per
//! lane). Every lane performs exactly the operations of the scalar code, each with the same single rounding (vector
//! add, multiply, divide and fused multiply-add are correctly rounded per lane), so the results are bit-identical;
//! `tests::simd_equals_scalar` checks it. Used only when the CPU has AVX2 and FMA (checked at run time).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn cross(l: [__m256; 3], r: [__m256; 3]) -> [__m256; 3] {
    [
        _mm256_fmsub_ps(l[1], r[2], _mm256_mul_ps(l[2], r[1])),
        _mm256_fmsub_ps(l[2], r[0], _mm256_mul_ps(l[0], r[2])),
        _mm256_fmsub_ps(l[0], r[1], _mm256_mul_ps(l[1], r[0])),
    ]
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_r(x: [__m256; 3], y: [__m256; 3]) -> __m256 {
    _mm256_fmadd_ps(x[0], y[0], _mm256_fmadd_ps(x[1], y[1], _mm256_mul_ps(x[2], y[2])))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_q(x: [__m256; 3], y: [__m256; 3]) -> __m256 {
    _mm256_fmadd_ps(x[0], y[0], _mm256_fmadd_ps(x[2], y[2], _mm256_mul_ps(x[1], y[1])))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sub(a: [__m256; 3], b: [__m256; 3]) -> [__m256; 3] {
    [_mm256_sub_ps(a[0], b[0]), _mm256_sub_ps(a[1], b[1]), _mm256_sub_ps(a[2], b[2])]
}

/// Squared distance of segment e0 -> e1 with e = e1 - e0 and pe = p - e0 (as tri::segment_d2).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn segment(e: [__m256; 3], pe: [__m256; 3]) -> __m256 {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let mut t = _mm256_div_ps(dot_r(e, pe), dot_r(e, e));
    t = _mm256_blendv_ps(t, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(t, zero));
    t = _mm256_blendv_ps(t, one, _mm256_cmp_ps::<_CMP_GT_OQ>(t, one));
    let x = [_mm256_fmsub_ps(e[0], t, pe[0]), _mm256_fmsub_ps(e[1], t, pe[1]), _mm256_fmsub_ps(e[2], t, pe[2])];
    dot_r(x, x)
}

/// min keeping the first argument unless the second is smaller.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn min_first(x: __m256, y: __m256) -> __m256 {
    _mm256_blendv_ps(x, y, _mm256_cmp_ps::<_CMP_LT_OQ>(y, x))
}

/// Squared distances from p to the `count` (1 to 8) triangles of a packed leaf: component k (a.x a.y a.z b.x ... c.z)
/// of lane i at leaf[k * count + i]. Lanes from `count` on are not read (masked loads) and their output is undefined.
///
/// # Safety
/// The CPU must support AVX2 and FMA; leaf must hold 9 * count floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dist2_8(leaf: *const f32, count: usize, p: [f32; 3], out: &mut [f32; 8]) {
    const IDX: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(count as i32), _mm256_loadu_si256(IDX.as_ptr() as *const __m256i));
    let ld = |k: usize| _mm256_maskload_ps(leaf.add(k * count), mask);
    let a = [ld(0), ld(1), ld(2)];
    let b = [ld(3), ld(4), ld(5)];
    let c = [ld(6), ld(7), ld(8)];
    let pv = [_mm256_set1_ps(p[0]), _mm256_set1_ps(p[1]), _mm256_set1_ps(p[2])];
    let ba = sub(b, a);
    let cb = sub(c, b);
    let ac = sub(a, c);
    let pa = sub(pv, a);
    let pb = sub(pv, b);
    let pc = sub(pv, c);
    let nor = cross(ba, ac);
    let s0 = dot_r(cross(ba, nor), pa);
    let s1 = dot_r(cross(cb, nor), pb);
    let s2 = dot_r(cross(ac, nor), pc);
    let neg = _mm256_or_ps(_mm256_or_ps(s0, s1), s2);
    // sign bit of any test set -> outside
    let outside = _mm256_castsi256_ps(_mm256_srai_epi32::<31>(_mm256_castps_si256(neg)));
    let h = dot_q(nor, pa);
    let plane = _mm256_div_ps(_mm256_mul_ps(h, h), dot_q(nor, nor));
    let edges = min_first(min_first(segment(ba, pa), segment(cb, pb)), segment(ac, pc));
    _mm256_storeu_ps(out.as_mut_ptr(), _mm256_blendv_ps(plane, edges, outside));
}

pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::tri;

    #[test]
    fn box_simd_equals_scalar() {
        if !available() {
            return;
        }
        let mut x: u64 = 11;
        let mut rnd = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for round in 0..50000 {
            let mut lo = [[0.0f32; 4]; 3];
            let mut hi = [[0.0f32; 4]; 3];
            for k in 0..3 {
                for j in 0..4 {
                    let (a, b) = (rnd(), rnd());
                    lo[k][j] = a.min(b);
                    hi[k][j] = a.max(b);
                }
            }
            let mut p = [rnd(), rnd(), rnd()];
            if round % 97 == 0 {
                p[round % 3] = f32::NAN;
            }
            let got = unsafe { box_d2_4(&lo, &hi, p) };
            for j in 0..4 {
                let mut d = [0.0f32; 3];
                for k in 0..3 {
                    d[k] = (lo[k][j] - p[k]).max(p[k] - hi[k][j]).max(0.0);
                }
                let want = tri::dot_r(d, d);
                assert!(want.to_bits() == got[j].to_bits(), "box {j}: {want} vs {}", got[j]);
            }
        }
    }

    #[test]
    fn simd_equals_scalar() {
        if !available() {
            return;
        }
        let mut x: u64 = 7;
        let mut rnd = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for round in 0..20000 {
            let count = 1 + round % 8;
            let mut leaf = vec![f32::NAN; 9 * count];
            let mut tris = [[[0.0f32; 3]; 3]; 8];
            for lane in 0..8 {
                for v in 0..3 {
                    for k in 0..3 {
                        let mut val = rnd();
                        if round % 7 == 0 && v == 2 {
                            val = tris[lane][0][k] + (tris[lane][1][k] - tris[lane][0][k]) * 0.5; // collinear
                        }
                        if round % 11 == 0 && v == 1 {
                            val = tris[lane][0][k]; // repeated vertex
                        }
                        tris[lane][v][k] = val;
                        if lane < count {
                            leaf[(v * 3 + k) * count + lane] = val;
                        }
                    }
                }
            }
            let p = [rnd() * 2.0, rnd() * 2.0, rnd() * 2.0];
            let mut out = [0.0f32; 8];
            unsafe { dist2_8(leaf.as_ptr(), count, p, &mut out) };
            for lane in 0..count {
                let s = tri::dist2(p, tris[lane][0], tris[lane][1], tris[lane][2]);
                assert!(s.to_bits() == out[lane].to_bits() || (s.is_nan() && out[lane].is_nan()), "lane {lane}: {s} vs {}", out[lane]);
            }
        }
    }
}

/// Squared distances from p to the four boxes of an inner node (lo/hi: [axis][child]), as bvh::box_d2.
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn box_d2_4(lo: &[[f32; 4]; 3], hi: &[[f32; 4]; 3], p: [f32; 3]) -> [f32; 4] {
    let zero = _mm_setzero_ps();
    let mut d = [zero; 3];
    for k in 0..3 {
        let pk = _mm_set1_ps(p[k]);
        let a = _mm_sub_ps(_mm_loadu_ps(lo[k].as_ptr()), pk);
        let b = _mm_sub_ps(pk, _mm_loadu_ps(hi[k].as_ptr()));
        // a.max(b) (IEEE maxNum: a NaN operand loses), then .max(0) (a NaN gives 0)
        let m = _mm_blendv_ps(_mm_max_ps(a, b), a, _mm_cmpunord_ps(b, b));
        d[k] = _mm_max_ps(m, zero);
    }
    let r = _mm_fmadd_ps(d[0], d[0], _mm_fmadd_ps(d[1], d[1], _mm_mul_ps(d[2], d[2])));
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), r);
    out
}
