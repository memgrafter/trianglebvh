# How trianglebvh was compared with cubvh

The question: does a TRELLIS.2 GLB exported with trianglebvh equal the same GLB exported with cubvh? cubvh's licence
allows evaluation, so cubvh ran only as a black box, through CuMesh's Python API (one Linux machine, RTX 3090 and RTX 3090 Ti,
CuMesh 12289e10 with cubvh ce92267a, 2026-11-01). Its source was never read (see "Clean-room record" in the README).

## Scope of the claim

What this shows: for the two calls TRELLIS.2's `to_glb` makes (`cuBVH(vertices, faces)` and
`unsigned_distance(points, return_uvw)`), trianglebvh returns the same float32 distances, the same face ids (also when
several faces are at exactly the same distance) and the same uvw on the tested inputs, and the exported GLB files are
byte-identical.

What this does NOT show:
- `ray_trace` and `signed_distance`: not implemented (they raise an error), so not compared.
- Every possible input. The uvw of the closest point differs from cubvh's in a small share of cases on degenerate
  and sliver triangles of the synthetic probes (table below); no difference was found on real meshes.
- Speed: trianglebvh runs on the CPU and is slower per query (table below).

## Levels

1. **Synthetic probes** (`tools/probe_cubvh.py`, checked with `tools/check_probe.py`), 3.9 million queries:

| probe | queries | distance | face id | uvw |
|---|---|---|---|---|
| single triangles (well-shaped, tiny, slivers, on a 1/1024 grid, far from the origin) | 1,164,000 | 100% | 100% | 98.09% |
| near edges (well-shaped triangles, points 1e-8 to 1e-4 of an edge length from an edge) | 192,000 | 100% | 100% | 100% |
| 64,000-triangle grid | 1,728,000 | 100% | 100% | 99.9997% |
| repeat of a grid (run to run) | 456,192 | 100% | 100% | 99.998% |
| exact ties: fans, duplicates, a plane mesh | 150,016 | 100% | 100% | 100% |
| tree order: 9 to 129 identical triangles, duplicate pairs, icospheres | 120,620 | 100% | 100% | 100% |
| exact ties on 300 random meshes (octant) | 2,400 | 100% | 100% | 98.79% |
| degenerate triangles (collinear, repeated vertices) | 32,340 | 100% | 100% | 95.01% |
| far points (900 to 1800), NaN, infinity, empty | 1,836 | 100% | 100% | 99.8% |

   The uvw differences are all on slivers (a vertex within 1e-4 of the opposite edge), degenerate triangles, query
   points placed exactly on an edge or vertex, or points 1e5 or more away. cubvh was deterministic in every repeat.

2. **Real queries** (`tools/export_compare.py --mode dual`): TRELLIS.2 decoded meshes, the export as run.py makes it;
   every `unsigned_distance` call ran on both BVHs and the bits were compared. torch, iron-box-b, brine-jar: 0 of
   231 million distances and 0 face ids differ. (uvw: see level 3; the last uvw rule was fitted after these runs, and
   all 4,367 uvw that differed on torch now match.)

3. **GLB files** (`tools/export_compare.py --mode glb`): one decoded mesh, exported twice in one process, with cubvh
   and with trianglebvh (library SHA-256 4fe2855a...):

| mesh | BVH faces | queries | GLB | to_glb s cubvh / ours | BVH s (build + queries) cubvh / ours |
|---|---|---|---|---|---|
| iron-box-b | 9.1M | 74.2M | byte-identical | 61.0 / 62.7 | 3.24 / 5.55 |
| licence-tag | | 38.3M | byte-identical | 51.5 / 42.0 | 1.99 / 3.63 |
| company-tally | | 39.2M | byte-identical | 77.0 / 69.9 | 1.49 / 3.07 |
| brine-jar | 16.0M | 142.7M | byte-identical | 48.7 / 48.9 | 6.29 / 13.41 |

4. **Production runs** (2026-11-01; the full TRELLIS.2 runner, gold image to GLB, seed 42, trianglebvh 0.1.0
   with library SHA-256 4fe2855a, made with `tools/proof_table.py`):
   - Against existing production GLBs (made with cubvh) of the same gold, seed and card model: 21 of 21 byte-identical.
     RTX 3090: 10 objects (decoded meshes 0.6M to 40.7M faces). RTX 3090 Ti: 11 objects (1.3M to 30.3M faces).
   - Pairs, the same runner with cubvh and with trianglebvh, where that card has no production GLB: RTX 3090 5
     objects (22.9M to 29.0M faces), RTX 3090 Ti 1: 6 of 6 byte-identical.

| card | objects | GLB byte-identical | BVH queries s, ours / cubvh | export s, ours - cubvh |
|---|---|---|---|---|
| RTX 3090 (vs production) | 10 | 10 | 0.95-11.66 / 0.04-0.64 | -10.6 to +1.1 (275 W vs earlier runs) |
| RTX 3090 Ti (vs production) | 11 | 11 | 1.11-10.95 / 0.04-0.76 | -4.9 to +4.2 (275 W, iron-box 250 W, vs earlier runs) |
| RTX 3090 (pairs, 250 W) | 5 | 5 | 9.32-10.94 / 0.59-0.67 | +6.1 to +9.6 |
| RTX 3090 Ti (pair) | 1 | 1 | 9.49 / 0.68 | +4.7 (275 W vs 250 W) |

   Only the RTX 3090 pairs ran at the same power limit and in the same conditions; there the whole prop took 4.7 to
   10.0 s longer with trianglebvh (one run each).

## Speed and memory

- cubvh builds its tree on the CPU (brine-jar: 5.6 s) and queries on the GPU (0.6 s for 143M queries). trianglebvh
  builds on the CPU (1.7 s, 8 threads) and queries on the CPU (11.7 s, 8 threads, AVX2 + FMA). Net, per export: +1.6
  to +7 s; the whole `to_glb` was within its run-to-run spread on these meshes.
- CPU query rate (`tools/bench.py`, brine-jar's 16.0M faces, points near the surface, an i7-14700KF shared with
  other work): 2.23 / 4.17 / 5.68 / 6.76 million queries per second at 4 / 8 / 12 / 16 threads; same result bits at
  every thread count.
- Memory: trianglebvh uses no GPU memory; cubvh's tree stays on the GPU through the export. Host memory for the tree:
  0.85 GiB at 16.0M faces.

## Reproduce

```bash
python -I tools/probe_cubvh.py PROBE_DIR            # evaluation environment with CuMesh (runs cubvh)
python -I tools/check_probe.py PROBE_DIR            # anywhere (CPU)
python -I tools/export_compare.py --mesh X.pt --out DIR --mode glb --lean PATH/cumesh_lean_remesh.py
g++ -O2 -o /tmp/nth_oracle tools/nth_oracle.cpp && python -I tools/check_select.py /tmp/nth_oracle
```

The recorded cubvh outputs (probe data, GLBs) are evaluation data and are not in this repository.
