"""Compare GLBs made with trianglebvh against reference GLBs of the same gold image, seed and card model, and the
per-stage time and memory of both runs (from the TRELLIS.2 runner's record.json).

    python -I tools/proof_table.py NEW_OUT REF_OUT [NEW_OUT REF_OUT ...] [--json OUT.json]
Each OUT folder holds <prop>/raw.glb and <prop>/record.json. Prints a Markdown table.
"""
import hashlib, json, os, sys

STAGES = ("glb.bvh.unsigned_distance", "glb.cumesh.remesh", "glb.grid_sample_3d")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def stage_s(rec, name):
    s = rec.get("stages", {}).get(name)
    return round(s["s"], 2) if s else None


def rows(new_out, ref_out):
    for prop in sorted(os.listdir(new_out)):
        new_glb, ref_glb = os.path.join(new_out, prop, "raw.glb"), os.path.join(ref_out, prop, "raw.glb")
        if not (os.path.isfile(new_glb) and os.path.isfile(ref_glb)):
            continue
        new = json.load(open(os.path.join(new_out, prop, "record.json")))
        ref = json.load(open(os.path.join(ref_out, prop, "record.json")))
        same_input = (new.get("gold"), new.get("seed"), new.get("gpu")) == (ref.get("gold"), ref.get("seed"), ref.get("gpu"))
        yield dict(prop=prop, gpu=new.get("gpu"), mesh_faces=new.get("mesh_faces"), same_input=same_input,
                   md5_new=md5(new_glb), md5_ref=md5(ref_glb), bvh_new=(new.get("bvh") or {}).get("commit") or (new.get("bvh") or {}).get("module"),
                   glb_s_new=new.get("glb_s"), glb_s_ref=ref.get("glb_s"),
                   **{f"{k}_new": stage_s(new, k) for k in STAGES}, **{f"{k}_ref": stage_s(ref, k) for k in STAGES},
                   gpu_peak_glb_new=(new.get("glb_mem") or {}).get("gpu_used_peak_gb"),
                   gpu_peak_glb_ref=(ref.get("glb_mem") or {}).get("gpu_used_peak_gb"),
                   rss_peak_new=new.get("rss_peak_gb"), rss_peak_ref=ref.get("rss_peak_gb"))


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
    allr = []
    for new_out, ref_out in zip(args[::2], args[1::2]):
        allr += list(rows(new_out, ref_out))
    print("| object | card | decoded faces | GLB md5 (trianglebvh) | identical | export s new / ref | BVH queries s new / ref | GPU peak in export GB new / ref |")
    print("|---|---|---|---|---|---|---|---|")
    for r in allr:
        same = "yes" if r["md5_new"] == r["md5_ref"] else "NO"
        if not r["same_input"]:
            same += " (input differs)"
        print(f"| {r['prop']} | {r['gpu'].replace('NVIDIA GeForce ', '')} | {r['mesh_faces']:,} | `{r['md5_new']}` | {same} | "
              f"{r['glb_s_new']} / {r['glb_s_ref']} | {r['glb.bvh.unsigned_distance_new']} / {r['glb.bvh.unsigned_distance_ref']} | "
              f"{r['gpu_peak_glb_new']} / {r['gpu_peak_glb_ref']} |")
    n = len(allr); k = sum(r["md5_new"] == r["md5_ref"] for r in allr)
    print(f"\n{k} of {n} GLBs byte-identical")
    if out_json:
        json.dump(allr, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
