"""CPU benchmark of trianglebvh (no GPU, no cubvh): build time and query rate on a real mesh.

    python -I tools/bench.py MESH (.pt decoded mesh with "vertices"/"faces", or .npz with v/f) [--points N] [--threads T ...]
Queries are mesh vertices moved by a random offset of about one voxel (1/1024), like the remesh and bake queries.
Prints the build seconds, host memory, and per thread count the query seconds and a SHA-256 of the results (the same
for every thread count).
"""
import argparse, hashlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--points", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, nargs="+", default=[8])
    a = ap.parse_args()
    import torch
    import trianglebvh
    if a.mesh.endswith(".npz"):
        z = np.load(a.mesh); v, f = z["v"], z["f"]
    else:
        d = torch.load(a.mesh, weights_only=False); v, f = d["vertices"].float().numpy(), d["faces"].int().numpy()
    rng = np.random.default_rng(0)
    pts = v[rng.integers(0, len(v), a.points)] + rng.normal(0, 1 / 1024, (a.points, 3)).astype(np.float32)
    pts = torch.from_numpy(pts.astype(np.float32))
    os.environ["TRIANGLEBVH_THREADS"] = str(max(a.threads))
    t = time.time(); bvh = trianglebvh.cuBVH(v, f); tb = time.time() - t
    print(f"faces {len(f)}  build {tb:.2f} s  host {bvh.host_bytes() / 1024 ** 3:.2f} GiB", flush=True)
    for th in a.threads:
        os.environ["TRIANGLEBVH_THREADS"] = str(th)
        t = time.time(); dd, ff, ww = bvh.unsigned_distance(pts, return_uvw=True); tq = time.time() - t
        h = hashlib.sha256(dd.cpu().numpy().tobytes() + ff.cpu().numpy().tobytes() + ww.cpu().numpy().tobytes()).hexdigest()[:16]
        print(f"threads {th:3d}  {a.points} queries {tq:.2f} s  ({a.points / tq / 1e6:.2f} M/s)  sha {h}", flush=True)


if __name__ == "__main__":
    main()
