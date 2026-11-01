"""Fit the float32 operation order of the point-triangle query to probe data (CPU, numpy, exact float32 emulation).
Each candidate is a published closest-point method (Ericson, Real-Time Collision Detection, 5.1.5 and 3.4) written
with explicit rounding; the script reports the share of samples whose distance and uvw match bit for bit.

    python -I tools/fit_distance.py PROBE_DIR/single.pt [more.pt ...]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp32 as F

f32 = np.float32


def load(paths):
    import torch
    A, B, C, P, D, W = [], [], [], [], [], []
    for path in paths:
        for c in torch.load(path):
            if "tri" not in c:
                continue
            n = c["pts"].shape[0]
            t = c["tri"].numpy().astype(f32)
            A.append(np.repeat(t[None, 0], n, 0)); B.append(np.repeat(t[None, 1], n, 0)); C.append(np.repeat(t[None, 2], n, 0))
            P.append(c["pts"].numpy().astype(f32)); D.append(c["d"].numpy()); W.append(c["w"].numpy())
    return [np.concatenate(x) for x in (A, B, C, P, D, W)]


# ---- vector helpers with a selectable dot-product order -------------------------------------------------------------
def vsub(a, b): return F.sub(a, b)


DOTS = {
    "plain": lambda a, b: F.add(F.add(F.mul(a[:, 0], b[:, 0]), F.mul(a[:, 1], b[:, 1])), F.mul(a[:, 2], b[:, 2])),
    "fma_l": lambda a, b: F.fma(a[:, 2], b[:, 2], F.fma(a[:, 1], b[:, 1], F.mul(a[:, 0], b[:, 0]))),
    "fma_r": lambda a, b: F.fma(a[:, 0], b[:, 0], F.fma(a[:, 1], b[:, 1], F.mul(a[:, 2], b[:, 2]))),
    "fma_m": lambda a, b: F.fma(a[:, 0], b[:, 0], F.fma(a[:, 2], b[:, 2], F.mul(a[:, 1], b[:, 1]))),
    "plain_r": lambda a, b: F.add(F.mul(a[:, 0], b[:, 0]), F.add(F.mul(a[:, 1], b[:, 1]), F.mul(a[:, 2], b[:, 2]))),
    "fma_l1": lambda a, b: F.add(F.fma(a[:, 0], b[:, 0], F.mul(a[:, 1], b[:, 1])), F.mul(a[:, 2], b[:, 2])),
    "fma_l2": lambda a, b: F.fma(a[:, 2], b[:, 2], F.add(F.mul(a[:, 0], b[:, 0]), F.mul(a[:, 1], b[:, 1]))),
    "fma_l3": lambda a, b: F.fma(a[:, 2], b[:, 2], F.fma(a[:, 0], b[:, 0], F.mul(a[:, 1], b[:, 1]))),
    "fma_r2": lambda a, b: F.fma(a[:, 0], b[:, 0], F.add(F.mul(a[:, 1], b[:, 1]), F.mul(a[:, 2], b[:, 2]))),
    "fma_r3": lambda a, b: F.fma(a[:, 0], b[:, 0], F.fma(a[:, 1], b[:, 1], F.mul(a[:, 2], b[:, 2]))),
}


def ulp_diff(x, y):
    xi = x.astype(f32).view(np.int32).astype(np.int64); yi = y.astype(f32).view(np.int32).astype(np.int64)
    xi = np.where(xi < 0, -(xi & 0x7FFFFFFF), xi); yi = np.where(yi < 0, -(yi & 0x7FFFFFFF), yi)
    return np.abs(xi - yi)


def report(name, d, w, D, W, mask):
    dm = (d.view(np.uint32) == D.view(np.uint32))[mask]
    wm = (w.view(np.uint32) == W.view(np.uint32)).all(1)[mask]
    ud = ulp_diff(d, D)[mask]
    print(f"{name:40s} n={mask.sum():8d} d exact {dm.mean():.6f} (ulp<=1 {(ud <= 1).mean():.4f}, max {ud.max()})  "
          f"uvw exact {wm.mean():.6f}", flush=True)
    return dm.mean(), wm.mean()


def main():
    A, B, C, P, D, W = load(sys.argv[1:])
    print("samples", len(P))
    interior = (W > 0).all(1)
    on_vertex = (W == 1).any(1)
    print("interior", interior.sum(), "vertex", on_vertex.sum())
    # distance from the reported uvw: q = u a + v b + w c, d = |p - q|
    for dn, dot in DOTS.items():
        for qn in ("uvw", "a+v*ab+w*ac"):
            u, v, w = W[:, 0:1], W[:, 1:2], W[:, 2:3]
            if qn == "uvw":
                q = F.add(F.add(F.mul(u, A), F.mul(v, B)), F.mul(w, C))
            else:
                q = F.add(F.add(A, F.mul(v, F.sub(B, A))), F.mul(w, F.sub(C, A)))
            pq = F.sub(P, q)
            d = F.sqrt(dot(pq, pq))
            report(f"d=|p-q(uvw)| q={qn} dot={dn}", d, W, D, W, np.ones(len(P), bool))


if __name__ == "__main__":
    main()
