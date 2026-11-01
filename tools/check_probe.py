"""Compare trianglebvh with recorded cubvh probe outputs (tools/probe_cubvh.py), bit for bit. CPU only.

    python -I tools/check_probe.py PROBE_DIR [case ...]
Prints, per case: queries, and the share with bit-identical distance, face index and uvw.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import trianglebvh
from probe_cubvh import with_decoys


def bits(x):
    return x.contiguous().view(torch.int32) if x.dtype == torch.float32 else x


def compare(c):
    if "tri" in c:
        v, f = with_decoys(c["tri"], torch.tensor([[0, 1, 2]], dtype=torch.int32))
    else:
        v, f = c["v"], c["faces"]
    bvh = trianglebvh.cuBVH(v, f)
    d, fi, w = bvh.unsigned_distance(c["pts"].cpu(), return_uvw=c["w"] is not None)
    d, fi = d.cpu(), fi.cpu()
    ok_d = bits(d) == bits(c["d"])
    ok_f = fi == c["f"]
    ok_w = (bits(w.cpu()) == bits(c["w"])).all(1) if w is not None else torch.ones_like(ok_f)
    return ok_d, ok_f, ok_w


def main():
    root = sys.argv[1]
    names = sys.argv[2:] or sorted(n[:-3] for n in os.listdir(root) if n.endswith(".pt"))
    for name in names:
        cases = torch.load(os.path.join(root, f"{name}.pt"))
        tot = dict(n=0, d=0, f=0, w=0, all=0)
        per = []
        for c in cases:
            od, of, ow = compare(c)
            n = od.numel()
            tot["n"] += n; tot["d"] += int(od.sum()); tot["f"] += int(of.sum()); tot["w"] += int(ow.sum())
            tot["all"] += int((od & of & ow).sum())
            if n and not bool((od & of & ow).all()):
                per.append((c.get("name", ""), n, int(od.sum()), int(of.sum()), int(ow.sum())))
        n = max(tot["n"], 1)
        print(f"{name:12s} queries {tot['n']:9d}  distance {tot['d'] / n:.6f}  face {tot['f'] / n:.6f}  uvw {tot['w'] / n:.6f}  "
              f"all {tot['all'] / n:.6f}", flush=True)
        for row in per[:12]:
            print("   not identical:", row)


if __name__ == "__main__":
    main()
