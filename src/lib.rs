//! C interface for Python (ctypes). All arrays are contiguous, row-major, float32 / int32 / int64 as named.

pub mod bvh;
pub mod select;
pub mod simd;
pub mod tri;

use bvh::Bvh;

/// Build a tree. vertices: nv x 3 float32; faces: nf x 3 int32 or int64 (face_bytes = 4 or 8). Returns null if a face
/// index is outside [0, nv).
///
/// # Safety
/// The pointers must point to arrays of the given sizes.
#[no_mangle]
pub unsafe extern "C" fn tb_build(vertices: *const f32, nv: usize, faces: *const u8, face_bytes: usize, nf: usize, threads: usize) -> *mut Bvh {
    let verts = std::slice::from_raw_parts(vertices as *const [f32; 3], nv);
    let mut f = Vec::with_capacity(nf);
    for i in 0..nf {
        let mut tri = [0u32; 3];
        for k in 0..3 {
            let j = 3 * i + k;
            let idx: i64 = if face_bytes == 8 { *(faces as *const i64).add(j) } else { *(faces as *const i32).add(j) as i64 };
            if idx < 0 || idx as usize >= nv || idx > u32::MAX as i64 {
                return std::ptr::null_mut();
            }
            tri[k] = idx as u32;
        }
        f.push(tri);
    }
    Box::into_raw(Box::new(Bvh::build(verts, &f, threads)))
}

/// 1 when the AVX2 + FMA code runs on this CPU (the results are the same either way).
#[no_mangle]
pub extern "C" fn tb_simd() -> i32 {
    simd::available() as i32
}

/// # Safety
/// `bvh` must come from tb_build and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn tb_free(bvh: *mut Bvh) {
    if !bvh.is_null() {
        drop(Box::from_raw(bvh));
    }
}

/// # Safety
/// `bvh` must come from tb_build.
#[no_mangle]
pub unsafe extern "C" fn tb_bytes(bvh: *const Bvh) -> usize {
    (*bvh).bytes()
}

/// points: n x 3 float32 -> dist: n float32, face: n int64, uvw: n x 3 float32 (may be null).
///
/// # Safety
/// The pointers must point to arrays of the given sizes; `bvh` must come from tb_build.
#[no_mangle]
pub unsafe extern "C" fn tb_unsigned_distance(bvh: *const Bvh, points: *const f32, n: usize, dist: *mut f32, face: *mut i64, uvw: *mut f32, threads: usize) {
    let pts = std::slice::from_raw_parts(points as *const [f32; 3], n);
    let d = std::slice::from_raw_parts_mut(dist, n);
    let f = std::slice::from_raw_parts_mut(face, n);
    let w = if uvw.is_null() { None } else { Some(std::slice::from_raw_parts_mut(uvw as *mut [f32; 3], n)) };
    (*bvh).unsigned_distance(pts, d, f, w, threads);
}

/// For each point, up to `k` faces tied at the minimum squared distance: out[i*k..] (padded with -1); counts[i] = all ties.
///
/// # Safety
/// The pointers must point to arrays of the given sizes; `bvh` must come from tb_build.
#[no_mangle]
pub unsafe extern "C" fn tb_ties(bvh: *const Bvh, points: *const f32, n: usize, k: usize, out: *mut i64, counts: *mut i64) {
    let pts = std::slice::from_raw_parts(points as *const [f32; 3], n);
    for (i, p) in pts.iter().enumerate() {
        let t = (*bvh).ties(*p);
        *counts.add(i) = t.len() as i64;
        for j in 0..k {
            *out.add(i * k + j) = if j < t.len() { t[j] as i64 } else { -1 };
        }
    }
}

/// Analysis: squared distance per (triangle, point) pair: tris n x 9 (a, b, c), points n x 3 -> out n.
///
/// # Safety
/// The pointers must point to arrays of the given sizes.
#[no_mangle]
pub unsafe extern "C" fn tb_dist2_pairs(tris: *const f32, points: *const f32, n: usize, out: *mut f32) {
    let t = std::slice::from_raw_parts(tris as *const [[f32; 3]; 3], n);
    let p = std::slice::from_raw_parts(points as *const [f32; 3], n);
    for i in 0..n {
        *out.add(i) = tri::dist2(p[i], t[i][0], t[i][1], t[i][2]);
    }
}

/// Test: the order std::nth_element-compatible selection leaves: perm (0..n) is reordered by keys[perm[i]] < ...
///
/// # Safety
/// The pointers must point to arrays of n elements.
#[no_mangle]
pub unsafe extern "C" fn tb_nth_element_test(keys: *const f32, n: usize, nth: usize, perm: *mut u32) {
    let k = std::slice::from_raw_parts(keys, n);
    let p = std::slice::from_raw_parts_mut(perm, n);
    for (i, x) in p.iter_mut().enumerate() {
        *x = i as u32;
    }
    select::nth_element(p, nth, &|a: &u32, b: &u32| k[*a as usize] < k[*b as usize]);
}
