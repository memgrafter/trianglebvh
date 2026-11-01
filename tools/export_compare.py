"""Equality test on real TRELLIS.2 meshes: o_voxel.postprocess.to_glb (run.py's export settings, uvrast bake) on a
saved decoded mesh, with cubvh and with trianglebvh. Needs a TRELLIS.2 environment with CuMesh (evaluation: it runs
cubvh) and a GPU.

    python -I tools/export_compare.py --mesh X.pt --out DIR [--mode dual|glb] [--texture-size 4096]

--mode dual: one export; every unsigned_distance call runs on both BVHs, the bits are compared per call, and cubvh's
result is returned (so every call sees the same input). Mismatching queries go to DIR/mismatch-<call>.npz.
--mode glb: two exports in one process, cubvh then trianglebvh; GLB md5 of each, seconds and peaks per stage.
"""
import argparse, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = dict(decimation_target=500000, remesh=True, remesh_band=1, remesh_project=0)   # as run.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="dual", choices=["dual", "glb"])
    ap.add_argument("--texture-size", type=int, default=4096)
    ap.add_argument("--save-input", action="store_true", help="dual: save each BVH's vertices and faces (bvh-input-<k>.npz)")
    ap.add_argument("--lean", default="", help="path of run.py's cumesh_lean_remesh.py (as production)")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    import numpy as np, torch, uvrast
    uvrast.install()
    import cumesh, o_voxel
    import trianglebvh
    pp = o_voxel.postprocess
    if a.lean:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cumesh_lean_remesh", a.lean)
        lean = importlib.util.module_from_spec(spec); spec.loader.exec_module(lean)
        lean.install(cumesh, swap=True)
    init = cumesh.CuMesh.init
    def init_empty_cache(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return init(self, *args, **kwargs)
    cumesh.CuMesh.init = init_empty_cache
    CubvhClass = cumesh.cuBVH
    os.makedirs(a.out, exist_ok=True)
    d = torch.load(a.mesh, weights_only=False)
    rec = dict(mesh=a.mesh, export=EXPORT, texture_size=a.texture_size, trianglebvh=trianglebvh.library_sha256(),
               threads=trianglebvh.threads(), gpu=torch.cuda.get_device_name(0), runs=[])
    calls = []

    def bits(x):
        return x.contiguous().view(torch.int32) if x.dtype == torch.float32 else x

    class Dual:
        def __init__(self, v, f):
            torch.cuda.synchronize(); t = time.time()
            self.c = CubvhClass(v, f); torch.cuda.synchronize(); tc = time.time() - t
            t = time.time(); self.o = trianglebvh.cuBVH(v, f); to = time.time() - t
            calls.append(dict(call="build", faces=int(f.shape[0]), cubvh_s=round(tc, 3), ours_s=round(to, 3),
                              ours_host_gb=round(self.o.host_bytes() / 1024 ** 3, 3)))
            if a.save_input:
                np.savez(os.path.join(a.out, f"bvh-input-{len(calls) - 1}.npz"), v=v.detach().cpu().numpy(), f=f.detach().cpu().numpy())

        def unsigned_distance(self, pts, return_uvw=False):
            torch.cuda.synchronize(); t = time.time()
            rc = self.c.unsigned_distance(pts, return_uvw=return_uvw); torch.cuda.synchronize(); tc = time.time() - t
            t = time.time(); ro = self.o.unsigned_distance(pts, return_uvw=return_uvw); torch.cuda.synchronize(); to = time.time() - t
            od = bits(rc[0]) == bits(ro[0]); of = rc[1] == ro[1]
            ow = (bits(rc[2]) == bits(ro[2])).all(-1) if return_uvw else torch.ones_like(of)
            bad = ~(od & of & ow)
            k = len(calls)
            row = dict(call=k, n=int(pts.numel() // 3), uvw=return_uvw, cubvh_s=round(tc, 3), ours_s=round(to, 3),
                       d_diff=int((~od).sum()), face_diff=int((~of).sum()), uvw_diff=int((~ow).sum()), any_diff=int(bad.sum()))
            if row["any_diff"]:
                ix = bad.reshape(-1).nonzero().reshape(-1)[:200000]
                P = pts.reshape(-1, 3)[ix]
                np.savez_compressed(os.path.join(a.out, f"mismatch-{k}.npz"), pts=P.cpu().numpy(),
                                    d_c=rc[0].reshape(-1)[ix].cpu().numpy(), d_o=ro[0].reshape(-1)[ix].cpu().numpy(),
                                    f_c=rc[1].reshape(-1)[ix].cpu().numpy(), f_o=ro[1].reshape(-1)[ix].cpu().numpy(),
                                    w_c=rc[2].reshape(-1, 3)[ix].cpu().numpy() if return_uvw else np.zeros(0),
                                    w_o=ro[2].reshape(-1, 3)[ix].cpu().numpy() if return_uvw else np.zeros(0))
            calls.append(row)
            print(json.dumps(row), flush=True)
            return rc

    def export(tag):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        glb = pp.to_glb(vertices=d["vertices"], faces=d["faces"], attr_volume=d["attrs"].cuda(), coords=d["coords"].cuda(),
                        attr_layout=d["layout"], voxel_size=d["voxel_size"], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        verbose=False, texture_size=a.texture_size, **EXPORT)
        torch.cuda.synchronize(); dt = time.time() - t
        path = os.path.join(a.out, f"{tag}.glb")
        glb.export(path)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        run = dict(tag=tag, md5=md5, faces=int(glb.faces.shape[0]), to_glb_s=round(dt, 2),
                   torch_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), calls=list(calls))
        calls.clear()
        rec["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "calls"}), flush=True)
        return run

    if a.mode == "dual":
        for mod in (cumesh, cumesh.bvh, cumesh.remeshing):
            mod.cuBVH = Dual
        export("dual")
    else:
        class Timed:
            impl = None
            def __init__(self, v, f):
                torch.cuda.synchronize(); t = time.time(); self.b = Timed.impl(v, f); torch.cuda.synchronize()
                calls.append(dict(call="build", s=round(time.time() - t, 3)))
            def unsigned_distance(self, pts, return_uvw=False):
                torch.cuda.synchronize(); t = time.time(); r = self.b.unsigned_distance(pts, return_uvw=return_uvw); torch.cuda.synchronize()
                calls.append(dict(call="query", n=int(pts.numel() // 3), uvw=return_uvw, s=round(time.time() - t, 3)))
                return r
        for mod in (cumesh, cumesh.bvh, cumesh.remeshing):
            mod.cuBVH = Timed
        Timed.impl = CubvhClass; export("cubvh")
        Timed.impl = trianglebvh.cuBVH; export("ours")
        rec["identical"] = rec["runs"][0]["md5"] == rec["runs"][1]["md5"]
        print("identical", rec["identical"], flush=True)
    json.dump(rec, open(os.path.join(a.out, f"compare-{a.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
