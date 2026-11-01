"""Black-box probes of cubvh's unsigned_distance through its Python API (CuMesh's cumesh.cuBVH), in an evaluation
environment. Writes one .pt file per case with the inputs and cubvh's outputs. The outputs are evaluation data: keep
them out of this repository.

    python -I tools/probe_cubvh.py OUT_DIR [case ...]
"""
import os, sys, time
import torch

CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def run(bvh_cls, verts, faces, pts, uvw=True):
    bvh = bvh_cls(verts, faces)
    d, f, w = bvh.unsigned_distance(pts.cuda(), return_uvw=uvw)
    torch.cuda.synchronize()
    return dict(d=d.cpu(), f=f.cpu(), w=None if w is None else w.cpu())


def decoys(n=8, at=60.0):
    """n tiny far triangles: the BVH needs more than 8 triangles."""
    g = torch.Generator().manual_seed(7)
    c = torch.tensor([at, at, at]) + torch.rand(n, 1, 3, generator=g) * 0.1
    v = (c + torch.rand(n, 3, 3, generator=g) * 0.01).reshape(-1, 3)
    return v, torch.arange(n * 3, dtype=torch.int32).reshape(n, 3)


def with_decoys(v, f):
    dv, df = decoys()
    return torch.cat([v, dv]), torch.cat([f, df + v.shape[0]])


def points_around(tri, n, g, spread):
    """Query points for one triangle: random in a box, vertices, edge points, centroid, points on the plane."""
    lo, hi = tri.min(0).values, tri.max(0).values
    ext = (hi - lo).max().clamp(min=1e-6) * spread
    box = lo - ext + torch.rand(n, 3, generator=g) * (hi - lo + 2 * ext)
    t = torch.rand(n, 1, generator=g)
    a, b, c = tri
    on_edges = torch.cat([a + t * (b - a), b + t * (c - b), c + t * (a - c)])
    bc = torch.rand(n, 3, generator=g); bc = bc / bc.sum(1, keepdim=True)
    on_face = bc @ tri
    nrm = torch.linalg.cross(b - a, c - a); nrm = nrm / nrm.norm().clamp(min=1e-30)
    off_face = on_face + nrm * (torch.rand(n, 1, generator=g) - 0.5) * ext
    return torch.cat([box, on_edges, on_face, off_face, tri, tri.mean(0, keepdim=True)])


@case
def single(bvh_cls, n_tri=3000, n_pts=64):
    """One target triangle per call near the origin (production scale, |x| < 0.5), 8 far decoys."""
    g = torch.Generator().manual_seed(1)
    out = []
    for i in range(n_tri):
        kind = i % 6
        tri = (torch.rand(3, 3, generator=g) - 0.5)
        if kind == 1: tri = tri * 0.002                      # voxel-sized triangle
        if kind == 2: tri[2] = tri[0] + (tri[1] - tri[0]) * 0.3 + torch.rand(3, generator=g) * 1e-4   # sliver
        if kind == 3: tri = (tri * 1024).round() / 1024      # on a grid
        if kind == 4: tri = tri * 1e-5 + 0.3                 # tiny, off origin
        pts = points_around(tri, n_pts, g, spread=1.0) * torch.ones(1)
        v, f = with_decoys(tri, torch.tensor([[0, 1, 2]], dtype=torch.int32))
        r = run(bvh_cls, v, f, pts)
        out.append(dict(tri=tri, pts=pts, **r))
    return out


@case
def degenerate(bvh_cls, n_pts=256):
    """Collinear, two equal vertices, all equal, zero-length edges; both orders of the vertices."""
    g = torch.Generator().manual_seed(2)
    a, b = torch.tensor([0.1, -0.2, 0.05]), torch.tensor([0.3, 0.1, -0.1])
    tris = [torch.stack([a, b, a + 0.5 * (b - a)]), torch.stack([a, a, b]), torch.stack([a, b, b]), torch.stack([a, b, a]),
            torch.stack([a, a, a]), torch.stack([a, b, a + 2.0 * (b - a)]), torch.stack([a, b, a + 1e-7 * (b - a)])]
    out = []
    for tri in tris + [t[[2, 0, 1]] for t in tris] + [t[[1, 0, 2]] for t in tris]:
        pts = points_around(tri, n_pts, g, spread=1.0)
        v, f = with_decoys(tri, torch.tensor([[0, 1, 2]], dtype=torch.int32))
        out.append(dict(tri=tri, pts=pts, **run(bvh_cls, v, f, pts)))
    return out


@case
def grid(bvh_cls, side=40, n_pts=24):
    """side^3 triangles, one per cell of a 1/side grid in [-0.5, 0.5]^3; queries near each, many per call."""
    g = torch.Generator().manual_seed(3)
    cell = 1.0 / side
    ijk = torch.stack(torch.meshgrid(*[torch.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3).float()
    org = ijk * cell - 0.5 + cell * 0.3
    tri = org[:, None] + torch.rand(org.shape[0], 3, 3, generator=g) * cell * 0.4
    m = tri.shape[0]
    q = tri.mean(1)[:, None] + (torch.rand(m, n_pts, 3, generator=g) - 0.5) * cell * 0.5
    pts = torch.cat([q.reshape(-1, 3), tri.reshape(-1, 3)])
    v = tri.reshape(-1, 3); f = torch.arange(m * 3, dtype=torch.int32).reshape(m, 3)
    return [dict(v=v, faces=f, pts=pts, **run(bvh_cls, v, f, pts))]


@case
def ties(bvh_cls, n_pts=2000):
    """Equal distances: duplicate triangles, shared edges and vertices (fans), and points equidistant by symmetry."""
    g = torch.Generator().manual_seed(4)
    out = []
    # fan of k triangles around a centre vertex, queries on the axis above the centre and on the shared edges
    for k in (3, 4, 6, 8, 12, 16, 32):
        ang = torch.arange(k) * 2 * torch.pi / k
        ring = torch.stack([0.2 * ang.cos(), 0.2 * ang.sin(), torch.zeros(k)], 1)
        v = torch.cat([torch.zeros(1, 3), ring])
        f = torch.stack([torch.zeros(k, dtype=torch.int32), torch.arange(1, k + 1, dtype=torch.int32),
                         (torch.arange(1, k + 1, dtype=torch.int32) % k) + 1], 1)
        h = torch.rand(n_pts, 1, generator=g) * 0.3
        axis = torch.cat([torch.zeros(n_pts, 2), h], 1)
        t = torch.rand(n_pts, 1, generator=g)
        e = torch.randint(0, k, (n_pts,), generator=g)
        edge = t * ring[e] + torch.cat([torch.zeros(n_pts, 2), h - 0.15], 1)
        pts = torch.cat([axis, edge, -axis])
        for perm in ("id", "rev", "rand"):
            ff = f if perm == "id" else f.flip(0) if perm == "rev" else f[torch.randperm(k, generator=g)]
            vv, fd = with_decoys(v, ff) if k <= 8 else (v, ff)
            out.append(dict(name=f"fan{k}-{perm}", v=vv, faces=fd, pts=pts, **run(bvh_cls, vv, fd, pts)))
    # the same triangle repeated m times (exact ties everywhere), and m copies far apart in the list
    tri = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    for m in (2, 3, 9, 33):
        v = tri.repeat(m, 1); f = torch.arange(3 * m, dtype=torch.int32).reshape(m, 3)
        pts = points_around(tri, n_pts // 4, g, spread=1.0)
        vv, fd = with_decoys(v, f) if m <= 8 else (v, f)
        out.append(dict(name=f"dup{m}", v=vv, faces=fd, pts=pts, **run(bvh_cls, vv, fd, pts)))
    # a regular grid mesh (two triangles per square), queries above vertices, edges, faces
    n = 32
    xs = torch.linspace(-0.4, 0.4, n + 1)
    gv = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    v = torch.cat([gv, torch.zeros(gv.shape[0], 1)], 1)
    idx = lambda i, j: i * (n + 1) + j
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij"); ii, jj = ii.reshape(-1), jj.reshape(-1)
    f = torch.cat([torch.stack([idx(ii, jj), idx(ii + 1, jj), idx(ii + 1, jj + 1)], 1),
                   torch.stack([idx(ii, jj), idx(ii + 1, jj + 1), idx(ii, jj + 1)], 1)]).int()
    vx = v[torch.randint(0, v.shape[0], (n_pts,), generator=g)] + torch.tensor([0, 0, 0.05])
    mid = (v[f[:, 0]] + v[f[:, 1]]) / 2
    ex = mid[torch.randint(0, f.shape[0], (n_pts,), generator=g)] + torch.tensor([0, 0, 0.03])
    pts = torch.cat([vx, ex, torch.rand(n_pts, 3, generator=g) - 0.5])
    for perm in ("id", "rand"):
        ff = f if perm == "id" else f[torch.randperm(f.shape[0], generator=g)]
        out.append(dict(name=f"plane-{perm}", v=v, faces=ff, pts=pts, **run(bvh_cls, v, ff, pts)))
    return out


@case
def far(bvh_cls):
    """Points far from the mesh (the README says the max distance is hard-coded to 100), NaN and inf, empty input."""
    v, f = with_decoys(torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), torch.tensor([[0, 1, 2]], dtype=torch.int32))
    r = torch.tensor([1, 10, 50, 99, 99.99, 100, 100.01, 101, 150, 200, 1e3, 1e5, 1e10])
    pts = torch.cat([torch.stack([-r, -r, -r], 1), torch.stack([-r, torch.zeros_like(r), torch.zeros_like(r)], 1),
                     torch.tensor([[float("nan"), 0, 0], [float("inf"), 0, 0], [-float("inf"), 0, 0]])])
    out = [dict(name="far", v=v, faces=f, pts=pts, **run(bvh_cls, v, f, pts))]
    e = run(bvh_cls, v, f, torch.zeros(0, 3))
    out.append(dict(name="empty", v=v, faces=f, pts=torch.zeros(0, 3), **e))
    out.append(dict(name="nouvw", v=v, faces=f, pts=pts[:5], **run(bvh_cls, v, f, pts[:5], uvw=False)))
    return out


@case
def repeat(bvh_cls):
    """Run-to-run: the same grid call three times (determinism)."""
    return [grid(bvh_cls, side=24, n_pts=8)[0] for _ in range(3)]


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    v = [list(map(float, x)) for x in v]
    for _ in range(level):
        mid, nf = {}, []
        def m(i, j):
            k = (min(i, j), max(i, j))
            if k not in mid:
                mid[k] = len(v); v.append([(v[i][c] + v[j][c]) / 2 for c in range(3)])
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    v = torch.tensor(v); v = v / v.norm(dim=1, keepdim=True) * 0.4
    return v.float(), torch.tensor(f, dtype=torch.int32)


@case
def tiebuild(bvh_cls):
    """Tie winners that depend on the tree: N identical triangles; two identical triangles at chosen indices among
    distinct ones; an icosphere (vertex-nearest queries tie), in input order and permuted."""
    g = torch.Generator().manual_seed(5)
    out = []
    tri = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    q = points_around(tri, 16, g, spread=1.0)
    for n in range(9, 130):
        v = tri.repeat(n, 1); f = torch.arange(3 * n, dtype=torch.int32).reshape(n, 3)
        out.append(dict(name=f"dupN{n}", v=v, faces=f, pts=q, **run(bvh_cls, v, f, q)))
    # 62 distinct small triangles on a 4x4x4 grid (one cell left for the target) + 2 copies of the target at i < j
    cells = torch.stack(torch.meshgrid(*[torch.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3).float()
    dec = [(c * 0.2 - 0.3)[None] + tri * 0.5 for c in cells[1:63]]
    tgt = tri * 0.5 + (cells[0] * 0.2 - 0.3)
    qt = points_around(tgt, 8, g, spread=0.5)
    for i, j in [(0, 1), (0, 63), (5, 40), (31, 32), (62, 63), (10, 11), (1, 62), (20, 50), (3, 7), (48, 49)]:
        rows, k = [], 0
        for slot in range(64):
            rows.append(tgt if slot in (i, j) else dec[k]); k += slot not in (i, j)
        v = torch.cat(rows); f = torch.arange(3 * 64, dtype=torch.int32).reshape(64, 3)
        out.append(dict(name=f"pair{i}-{j}", v=v, faces=f, pts=qt, **run(bvh_cls, v, f, qt)))
    for level in (2, 3, 4):
        v, f = icosphere(level)
        nq = 4000
        d = torch.randn(nq, 3, generator=g); d = d / d.norm(dim=1, keepdim=True)
        r = 0.4 + torch.rand(nq, 1, generator=g) * 0.3
        qv = v[torch.randint(0, v.shape[0], (nq,), generator=g)] * (1 + torch.rand(nq, 1, generator=g) * 0.5)
        pts = torch.cat([d * r, qv, -qv * 0.2])
        for perm in range(3):
            ff = f if perm == 0 else f[torch.randperm(f.shape[0], generator=g)]
            out.append(dict(name=f"ico{level}-p{perm}", v=v, faces=ff, pts=pts, **run(bvh_cls, v, ff, pts)))
    return out


@case
def octant(bvh_cls, configs=300):
    """Exact ties among every triangle: each triangle has the origin as its first vertex and the other two on a 1/64
    lattice in the positive octant; a query in the negative octant is nearest to the origin on every triangle, with
    exact arithmetic, so all distances are equal and the winner shows the tree's visiting order."""
    g = torch.Generator().manual_seed(6)
    out = []
    pts = -torch.tensor([[1, 1, 1], [1, 2, 3], [3, 1, 2], [2, 3, 1], [8, 1, 1], [1, 8, 1], [1, 1, 8], [5, 5, 5]], dtype=torch.float32) / 64
    for k in range(configs):
        n = int(torch.randint(9, 400, (1,), generator=g))
        step = [1, 2, 4, 8][k % 4]          # coarser lattices give more equal centroid coordinates
        bc = torch.randint(0, 32 // step + 1, (n, 2, 3), generator=g).float() * step / 64
        bc[:, :, 0] += 1 / 64                # keep b and c off the origin
        v = torch.cat([torch.zeros(n, 1, 3), bc], 1).reshape(-1, 3)
        f = torch.arange(3 * n, dtype=torch.int32).reshape(n, 3)
        out.append(dict(name=f"oct{k}", v=v, faces=f, pts=pts, **run(bvh_cls, v, f, pts)))
    return out


@case
def nearedge(bvh_cls, n_tri=3000, n_pts=64):
    """Well-shaped triangles; points whose projection is within 1e-8 to 1e-4 (times the edge length) of an edge, on
    either side, plus a small offset along the normal (the case that decides projection versus edge for uvw)."""
    g = torch.Generator().manual_seed(8)
    out = []
    for i in range(n_tri):
        while True:
            tri = torch.rand(3, 3, generator=g) - 0.5
            if i % 3 == 1: tri = tri * 0.002 + (torch.rand(3, generator=g) - 0.5) * 0.8
            e = [tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2]]
            nrm = torch.linalg.cross(e[0], -e[2])
            area = nrm.norm() / 2
            if area > 0.2 * max(x.norm() for x in e) ** 2: break     # no slivers
        nrm = nrm / nrm.norm()
        k = torch.randint(0, 3, (n_pts,), generator=g)
        t = torch.rand(n_pts, 1, generator=g)
        e0 = tri[k]; ev = torch.stack(e)[k]
        perp = torch.linalg.cross(ev, nrm.expand_as(ev))
        perp = perp / perp.norm(dim=1, keepdim=True) * ev.norm(dim=1, keepdim=True)
        delta = 10 ** (-8 + 4 * torch.rand(n_pts, 1, generator=g)) * torch.where(torch.rand(n_pts, 1, generator=g) < 0.5, -1.0, 1.0)
        h = (torch.rand(n_pts, 1, generator=g) - 0.5) * 2e-3 * ev.norm(dim=1, keepdim=True)
        pts = e0 + t * ev + delta * perp + h * nrm
        v, f = with_decoys(tri, torch.tensor([[0, 1, 2]], dtype=torch.int32))
        out.append(dict(tri=tri, pts=pts, **run(bvh_cls, v, f, pts)))
    return out


@case
def far2(bvh_cls):
    """Where the far-point cut-off starts: points on the -x axis and the diagonal at 900 to 1800 from the triangle."""
    v, f = with_decoys(torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), torch.tensor([[0, 1, 2]], dtype=torch.int32))
    r = torch.linspace(900, 1800, 901)
    pts = torch.cat([torch.stack([-r, 0 * r, 0 * r], 1), torch.stack([-r, -r, -r], 1) / 3 ** 0.5])
    return [dict(name="far2", v=v, faces=f, pts=pts, **run(bvh_cls, v, f, pts))]


def main():
    out_dir = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    os.makedirs(out_dir, exist_ok=True)
    import cumesh
    for name in names:
        t = time.time()
        res = CASES[name](cumesh.cuBVH)
        torch.save(res, os.path.join(out_dir, f"{name}.pt"))
        print(f"{name}: {len(res)} calls, {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
