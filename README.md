# trianglebvh

A clean-room replacement for the cubvh BVH that CuMesh builds into itself (`cumesh._cubvh`) and that TRELLIS.2's GLB
export (`o_voxel.postprocess.to_glb`) uses for the remesh distance queries and the texture bake's attribute lookup. The
cubvh version that CuMesh pins carries the NVIDIA Source Code License of instant-ngp next to its MIT licence; that
licence allows non-commercial use only, so a commercial pipeline cannot use it.

trianglebvh gives the two calls the pipeline makes, with bit-identical results:

- `cuBVH(vertices, triangles)`: vertices `[N, 3]` float32, triangles `[M, 3]` int32 or int64 (torch tensors or numpy
  arrays). As cubvh, it asserts more than 8 triangles.
- `unsigned_distance(points, return_uvw=False)`: returns `(distance [..], face_id [..] int64, uvw [.., 3] or None)` on
  the device cubvh would use (CUDA when available). `uvw` are the barycentric coordinates of the closest point on
  that face.

`ray_trace` and `signed_distance` are not implemented: they raise an error, so a caller never gets a silently different
result. The work runs on the CPU (Rust, no GPU memory); `TRIANGLEBVH_THREADS` sets the threads per call (default 8).

## Use

```python
import cumesh
import trianglebvh; trianglebvh.install()   # cumesh.cuBVH -> trianglebvh.cuBVH; any call into cubvh then raises
import o_voxel                              # to_glb now uses trianglebvh
```

Install: `pip install --no-build-isolation .` (needs a Rust toolchain; the crate has no dependencies, so the build
does not use the network). The library uses AVX2 and FMA when the CPU has them and the same arithmetic in scalar
code otherwise; the results are the same bits either way.

## Behaviour (found by black-box probes; see "Clean-room record")

Distance of a point p to a triangle (a, b, c), in float32:
- Inside test: with the normal n = (b - a) x (a - c), the projection of p is inside when none of
  ((b - a) x n) . (p - a), ((c - b) x n) . (p - b), ((a - c) x n) . (p - c) has its sign bit set (so -0.0 counts as outside).
- Inside: squared distance h * h / (n . n) with h = n . (p - a). Outside: the minimum over the edges a->b, b->c, c->a of
  the point-segment squared distance |e t - (p - e0)|^2, t = clamp((e . (p - e0)) / (e . e), 0, 1); the clamp lets a
  NaN through, and the minimum keeps a NaN first argument (a triangle whose distance is NaN never wins).
- Every dot product is x0 y0 + (x1 y1 + x2 y2) with fused multiply-adds; the inner sum is fused on x1 y1 in most
  places and on x2 y2 in some (`src/tri.rs` names each one). Every cross product component is one fused multiply-add.
- The distance is the correctly rounded square root of the minimum.

Closest point (uvw), for the winning face only:
- q0 = p - n_hat (n_hat . (p - a)), with n_hat = n / |n| and one fused multiply-add per component.
- If the three sub-triangle areas n . ((b - q0) x (c - q0)), n . ((c - q0) x (a - q0)), n . ((a - q0) x (b - q0)) have one
  sign, q = q0. Otherwise q is the closest point to q0 on the nearest edge (a->b, b->c, c->a; the first wins a tie).
- uvw: v and w by Cramer's rule on the edge dot products (Ericson 3.4); u = 1 - v - w computed in double precision.

Tree and query order (they decide which face wins an exact tie, for example at a vertex shared by a fan):
- Four children per node; a leaf holds at most 8 triangles. A node's range is split at its median along the axis
  of largest centroid variance, then each half again along its own axis; the children are the first, third, second
  and fourth quarter, in that order. The median selection leaves the elements in the order GNU libstdc++'s
  std::nth_element does (checked against libstdc++ itself: `tools/check_select.py`).
- Centroid ((a + b) + c) / 3; variance from float32 sums in the range's current order, mean = sum * (1 / n).
- The query is a stack. A leaf sets the best when a triangle's squared distance is <= the best (the last equal one
  wins). An inner node sorts its children by box distance with a five-comparator network and pushes those with box
  distance <= the best, farthest first. The best starts at 1000^2: a point farther than 1000 from every triangle gets
  distance 0 and face 0.

## Method

The probes (`tools/probe_cubvh.py`) ran cubvh through CuMesh's Python API on test meshes and recorded distances, face
ids and uvw. Each behaviour above was then fitted bit for bit on the CPU with exact float32 arithmetic in numpy
(`tools/fp32.py`): for every expression, the candidate orders of the operations and of the fused multiply-adds were
compared with the recorded bits until one matched every probe. The tree was fitted from exact ties: identical
triangles, triangles sharing one vertex with exact (dyadic) coordinates, and icospheres.

## Files

- `src/tri.rs`: distance, inside test, closest point and barycentric coordinates (the float32 rules).
- `src/bvh.rs`: the tree and the query. `src/select.rs`: the median selection. `src/simd.rs`: the same arithmetic
  for 8 triangles (AVX2 + FMA) and 4 boxes at once.
- `trianglebvh/__init__.py`: the Python class and `install()`.
- `tools/probe_cubvh.py`: black-box probes of cubvh (evaluation only). `tools/check_probe.py`: compares trianglebvh
  with recorded probe outputs. `tools/fp32.py`, `tools/fit_distance.py`: the fitting tools.
- `tools/nth_oracle.cpp`, `tools/check_select.py`: the selection order against libstdc++.
- `tools/export_compare.py`: the equality test on real TRELLIS.2 meshes. `tools/bench.py`: CPU benchmark.
- `docs/COMPARISON.md`: how trianglebvh was compared with cubvh, and the results.

## Clean-room record

Rule: nobody working on this repository opened, read, searched or copied the source code of cubvh (any version, any
fork, the CuMesh submodule, an installed package, a wheel or an sdist), of instant-ngp or tiny-cuda-nn, or of
nvdiffrast or nvdiffrec.

What was read:
- cubvh's public README (API: `cuBVH(vertices, faces)`, `unsigned_distance(points, return_uvw)` returning distances,
  face ids and the barycentric `uvw` of the closest point; "the max distance is hard-coded to 100": the probes show
  1000 for the pinned version).
- CuMesh's own MIT code outside the cubvh submodule: `cumesh/bvh.py` (the Python class: argument conversion, the >8
  triangles assertion, output dtypes), `cumesh/remeshing.py` and `setup.py` (which files form `cumesh._cubvh`, the
  compiler flags).
- TRELLIS.2's MIT code that calls it: `o-voxel/o_voxel/postprocess.py` (`to_glb`).
- Published methods: point-triangle distance (I. Quilez, "distance to triangle"; Ericson, Real-Time Collision
  Detection, 5.1.5), barycentric coordinates (Ericson 3.4), introselect (Musser 1997), Hoare partition.
- Hypotheses for the float order came from general compiler knowledge (fixed-size sums evaluated as x0 + (x1 + x2),
  a * b + c contracted to one fused multiply-add); each was kept only where the probes confirmed it bit for bit.

How the behaviour was found: black-box runs of cubvh through CuMesh's Python API in an evaluation environment, on
test meshes (`tools/probe_cubvh.py`): single triangles of many shapes (well-shaped, tiny, slivers, on a grid, far from
the origin), degenerate triangles, 64,000-triangle grids, fans, duplicate triangles, icospheres, exact-tie meshes,
points near edges, far points, NaN and empty input. The recorded outputs are evaluation data and are not in this
repository. The selection order was checked against GNU libstdc++'s `std::nth_element` as a black box
(`tools/nth_oracle.cpp`).

## Proof of equality

See [docs/COMPARISON.md](docs/COMPARISON.md).

## Licence

MIT (`LICENSE`).
