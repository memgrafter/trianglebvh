"""Self-contained tests (no cubvh), CPU, torch needed: python -m pytest tests, or python -I tests/test_basic.py."""
import ctypes, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import contextlib
import torch
import trianglebvh


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")


def mesh(n=2000, seed=0):
    g = np.random.default_rng(seed)
    c = g.random((n, 1, 3)).astype(np.float32) - 0.5
    v = (c + (g.random((n, 3, 3)).astype(np.float32) - 0.5) * 0.05).reshape(-1, 3)
    return v, np.arange(3 * n, dtype=np.int32).reshape(n, 3)


def brute_d2(v, f, p):
    lib = trianglebvh._load()
    lib.tb_dist2_pairs.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    t = np.ascontiguousarray(v[f.astype(np.int64)].reshape(-1, 9), np.float32)
    out = np.empty((len(p), len(t)), np.float32)
    for i, q in enumerate(p):
        pp = np.ascontiguousarray(np.repeat(q[None], len(t), 0), np.float32)
        lib.tb_dist2_pairs(t.ctypes.data, pp.ctypes.data, len(t), out[i].ctypes.data)
    return out


def test_minimum_and_ties():
    v, f = mesh()
    p = (np.random.default_rng(1).random((300, 3)).astype(np.float32) - 0.5) * 1.2
    d, fi, w = trianglebvh.cuBVH(torch.from_numpy(v), torch.from_numpy(f)).unsigned_distance(torch.from_numpy(p), return_uvw=True)
    d2 = brute_d2(v, f, p)
    best = np.nanmin(d2, 1)
    assert np.array_equal(d.cpu().numpy(), np.sqrt(best))                       # the minimum, correctly rounded sqrt
    assert np.all(d2[np.arange(len(p)), fi.cpu().numpy()] == best)              # the face is one of the minimal ones
    assert np.allclose(w.cpu().numpy().sum(1), 1, atol=1e-4)


def test_same_bits_any_thread_count_and_simd():
    v, f = mesh(5000, 2)
    p = torch.from_numpy((np.random.default_rng(3).random((20000, 3)).astype(np.float32) - 0.5))
    outs = []
    for th, simd in ((1, "1"), (8, "1"), (3, "0")):
        os.environ["TRIANGLEBVH_THREADS"], os.environ["TRIANGLEBVH_SIMD"] = str(th), simd
        r = trianglebvh.cuBVH(v, f).unsigned_distance(p, return_uvw=True)
        outs.append([x.cpu().numpy().tobytes() for x in r])
    os.environ.pop("TRIANGLEBVH_THREADS"); os.environ.pop("TRIANGLEBVH_SIMD")
    assert outs[0] == outs[1] == outs[2]


def test_shapes_dtypes_and_errors():
    v, f = mesh(50)
    b = trianglebvh.cuBVH(v, f.astype(np.int64))
    d, fi, w = b.unsigned_distance(torch.zeros(4, 5, 3))
    assert d.shape == (4, 5) and d.dtype == torch.float32 and fi.dtype == torch.int64 and w is None
    d, fi, w = b.unsigned_distance(torch.zeros(0, 3), return_uvw=True)
    assert d.shape == (0,) and w.shape == (0, 3)
    with raises(AssertionError):
        trianglebvh.cuBVH(v[:24], f[:8])
    with raises(ValueError):
        trianglebvh.cuBVH(v, f + 10 ** 6)
    with raises(NotImplementedError):
        b.ray_trace(torch.zeros(1, 3), torch.ones(1, 3))
    with raises(NotImplementedError):
        b.signed_distance(torch.zeros(1, 3))


def test_far_point_is_face_0_distance_0():
    v, f = mesh(20)
    d, fi, _ = trianglebvh.cuBVH(v, f).unsigned_distance(torch.tensor([[2000.0, 0, 0]]))
    assert d.item() == 0 and fi.item() == 0


def test_install_replaces_cubvh():
    stub = types.SimpleNamespace(cuBVH=object, bvh=types.SimpleNamespace(cuBVH=object, _backend=object()),
                                 remeshing=types.SimpleNamespace(cuBVH=object))
    info = trianglebvh.install(stub)
    assert stub.cuBVH is stub.bvh.cuBVH is stub.remeshing.cuBVH is trianglebvh.cuBVH
    with raises(RuntimeError):
        stub.bvh._backend.create_cuBVH
    assert len(info["library_sha256"]) == 64


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
