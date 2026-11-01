"""trianglebvh: a clean-room replacement for the cubvh BVH that CuMesh builds into itself (cumesh._cubvh).

TRELLIS.2's GLB export (o_voxel.postprocess.to_glb) calls only `cumesh.cuBVH(vertices, faces)` and
`bvh.unsigned_distance(points, return_uvw=...)`. This module gives the same class name with those two calls; the
results are bit-identical to cubvh's on the inputs in docs/COMPARISON.md. The work runs on the CPU (Rust, no GPU
memory); results come back on the device cubvh would use.

    import trianglebvh; trianglebvh.install()   # after `import cumesh`, before to_glb runs
"""
import ctypes, hashlib, os, sys

import numpy as np

__version__ = "0.1.0"
_LIB_NAME = {"darwin": "libtrianglebvh.dylib", "win32": "trianglebvh.dll"}.get(sys.platform, "libtrianglebvh.so")
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIB_NAME)
DEFAULT_THREADS = 8
_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_LIB_PATH)
        lib.tb_build.restype = ctypes.c_void_p
        lib.tb_build.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        lib.tb_free.argtypes = [ctypes.c_void_p]
        lib.tb_bytes.restype = ctypes.c_size_t
        lib.tb_bytes.argtypes = [ctypes.c_void_p]
        lib.tb_simd.restype = ctypes.c_int
        lib.tb_unsigned_distance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_size_t]
        _lib = lib
    return _lib


def threads():
    """Worker threads per call: TRIANGLEBVH_THREADS, default DEFAULT_THREADS (leaves CPU for other work on a shared machine)."""
    return max(1, int(os.environ.get("TRIANGLEBVH_THREADS", DEFAULT_THREADS)))


def _host(x, dtype):
    """A contiguous host numpy array of x (a torch tensor or array-like), without a copy when it is already one."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x), dtype=dtype)


class cuBVH:
    """Same constructor and unsigned_distance as cumesh.cuBVH (cubvh). Other methods are not implemented."""

    def __init__(self, vertices, triangles):
        tri = triangles.detach().cpu().numpy() if hasattr(triangles, "detach") else np.asarray(triangles)
        assert tri.shape[0] > 8, "BVH needs at least 8 triangles."
        v = _host(vertices, np.float32)
        f = np.ascontiguousarray(tri, dtype=np.int64 if tri.dtype.itemsize == 8 else np.int32)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"vertices must be [N, 3] and triangles [M, 3], got {v.shape} and {f.shape}")
        lib = _load()
        self._ptr = lib.tb_build(v.ctypes.data, v.shape[0], f.ctypes.data, f.dtype.itemsize, f.shape[0], threads())
        if not self._ptr:
            raise ValueError("a triangle index is outside the vertex array")
        self.num_triangles = f.shape[0]

    def __del__(self):
        if getattr(self, "_ptr", None):
            _load().tb_free(self._ptr)
            self._ptr = None

    def host_bytes(self):
        return _load().tb_bytes(self._ptr)

    def unsigned_distance(self, positions, return_uvw=False):
        import torch
        positions = positions.float()
        device = positions.device if positions.is_cuda else torch.device("cuda") if torch.cuda.is_available() else positions.device
        prefix = positions.shape[:-1]
        pts = positions.detach().reshape(-1, 3).cpu().contiguous()
        n = pts.shape[0]
        dist = torch.empty(n, dtype=torch.float32)
        face = torch.empty(n, dtype=torch.int64)
        uvw = torch.empty(n, 3, dtype=torch.float32) if return_uvw else None
        if n:
            _load().tb_unsigned_distance(self._ptr, pts.data_ptr(), n, dist.data_ptr(), face.data_ptr(),
                                         uvw.data_ptr() if uvw is not None else None, threads())
        dist, face = dist.view(prefix).to(device), face.view(prefix).to(device)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3).to(device)
        return dist, face, uvw

    def ties(self, points, k=16):
        """Analysis: per point, the faces tied at the minimum distance ([n, k], -1 padded) and their count."""
        lib = _load()
        lib.tb_ties.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
        p = _host(points, np.float32).reshape(-1, 3)
        out = np.empty((p.shape[0], k), np.int64); cnt = np.empty(p.shape[0], np.int64)
        lib.tb_ties(self._ptr, p.ctypes.data, p.shape[0], k, out.ctypes.data, cnt.ctypes.data)
        return out, cnt

    def ray_trace(self, rays_o, rays_d):
        raise NotImplementedError("trianglebvh implements only unsigned_distance (what TRELLIS.2's to_glb calls)")

    def signed_distance(self, positions, return_uvw=False, mode="watertight"):
        raise NotImplementedError("trianglebvh implements only unsigned_distance (what TRELLIS.2's to_glb calls)")


class _NoCubvh:
    """Put in place of cumesh.bvh._backend by install(): any call into cubvh stops the run."""

    def __getattr__(self, name):
        raise RuntimeError(f"cubvh was called ({name}) after trianglebvh.install()")


def library_sha256():
    with open(_LIB_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def install(cumesh=None):
    """Replace cubvh in CuMesh: cumesh.cuBVH, cumesh.bvh.cuBVH and cumesh.remeshing.cuBVH become trianglebvh.cuBVH,
    and cumesh.bvh._backend (the compiled cubvh module) becomes a stub that raises on any use. Returns provenance."""
    if cumesh is None:
        import cumesh
    for mod in (cumesh, cumesh.bvh, cumesh.remeshing):
        mod.cuBVH = cuBVH
    cumesh.bvh._backend = _NoCubvh()
    return dict(version=__version__, module=os.path.abspath(__file__), library=_LIB_PATH, library_sha256=library_sha256(),
                threads=threads(), simd=bool(_load().tb_simd()) and os.environ.get("TRIANGLEBVH_SIMD", "1") != "0")
