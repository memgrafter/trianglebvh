"""Exact float32 arithmetic in numpy, for fitting and checking the float32 operation order of the distance query on
the CPU. Each function returns the correctly rounded float32 result of the exact operation (as CUDA's __fadd_rn,
__fmul_rn, __fmaf_rn, __fdiv_rn, __fsqrt_rn)."""
import numpy as np

f32 = np.float32


def add(a, b): return (np.asarray(a, f32) + np.asarray(b, f32)).astype(f32)
def sub(a, b): return (np.asarray(a, f32) - np.asarray(b, f32)).astype(f32)
def mul(a, b): return (np.asarray(a, f32) * np.asarray(b, f32)).astype(f32)
def div(a, b): return (np.asarray(a, f32) / np.asarray(b, f32)).astype(f32)
def sqrt(a): return np.sqrt(np.asarray(a, f32)).astype(f32)


def fma(a, b, c):
    """round32(a * b + c) with one rounding. a * b is exact in float64 (48 bits); the float64 sum's rounding error is
    found exactly (TwoSum) and decides a float32 tie that the float64 rounding created."""
    a, b, c = (np.asarray(x, f32).astype(np.float64) for x in (a, b, c))
    p = a * b
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)
    r = s.astype(f32)
    # is s exactly halfway between two float32 values? then the sign of e decides
    up = np.nextafter(r, f32(np.inf)); dn = np.nextafter(r, f32(-np.inf))
    rd = r.astype(np.float64)
    half_hi = (up.astype(np.float64) + rd) / 2
    half_lo = (dn.astype(np.float64) + rd) / 2
    r = np.where((s == half_hi) & (e > 0), up, r)   # cast rounded down to r (even) but the exact value is above
    r = np.where((s == half_lo) & (e < 0), dn, r)
    return r.astype(f32)


def selftest(n=200000, seed=0):
    from fractions import Fraction
    g = np.random.default_rng(seed)
    a, b, c = (g.standard_normal(n).astype(f32) * f32(2.0) ** g.integers(-10, 10, n).astype(f32) for _ in range(3))
    r = fma(a, b, c)
    for i in range(0, n, max(1, n // 2000)):
        exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        lo = np.float32(float(exact))
        cands = [np.nextafter(lo, f32(-np.inf)), lo, np.nextafter(lo, f32(np.inf))]
        best = min(cands, key=lambda x: (abs(Fraction(float(x)) - exact), int(np.frombuffer(x.tobytes(), np.uint32)[0]) & 1))
        assert best == r[i], (a[i], b[i], c[i], best, r[i])
    return True



class E:
    """A float32 array whose product stays pending until an add or subtract consumes it, as LLVM's FMA contraction
    (nvcc's default -fmad=true) does: add(mul(a, b), c) -> fma(a, b, c), tried on the first operand first;
    sub(mul(a, b), c) -> fma(a, b, -c); sub(c, mul(a, b)) -> fma(-a, b, c). With contract=False every operation
    rounds on its own."""
    contract = True

    def __init__(self, v=None, pm=None):
        self.v, self.pm = v, pm

    @staticmethod
    def of(x):
        return x if isinstance(x, E) else E(np.asarray(x, f32))

    def val(self):
        if self.v is None:
            self.v = mul(*self.pm)
        return self.v

    def __mul__(self, o):
        o = E.of(o)
        return E(pm=(self.val(), o.val())) if E.contract else E(mul(self.val(), o.val()))
    __rmul__ = __mul__

    def __add__(self, o):
        o = E.of(o)
        if self.pm is not None and self.v is None:
            return E(fma(*self.pm, o.val()))
        if o.pm is not None and o.v is None:
            return E(fma(*o.pm, self.val()))
        return E(add(self.val(), o.val()))

    def __radd__(self, o):
        return E.of(o) + self

    def __sub__(self, o):
        o = E.of(o)
        if self.pm is not None and self.v is None:
            return E(fma(*self.pm, -o.val()))
        if o.pm is not None and o.v is None:
            return E(fma(-o.pm[0], o.pm[1], self.val()))
        return E(sub(self.val(), o.val()))

    def __rsub__(self, o):
        return E.of(o) - self

    def __neg__(self):
        if self.pm is not None and self.v is None:
            return E(pm=(-self.pm[0], self.pm[1]))
        return E(-self.val())

    def __truediv__(self, o):
        return E(div(self.val(), E.of(o).val()))

    def __rtruediv__(self, o):
        return E.of(o) / self


def esqrt(x): return E(sqrt(E.of(x).val()))


class V:
    """A 3-vector of E with Eigen's fixed-size operation order: dot = x0 + (x1 + x2); cross as Eigen's cross()."""
    def __init__(self, x, y, z): self.c = [E.of(x), E.of(y), E.of(z)]

    @staticmethod
    def of(a): return a if isinstance(a, V) else V(a[..., 0], a[..., 1], a[..., 2])

    def __add__(s, o): o = V.of(o); return V(*[a + b for a, b in zip(s.c, o.c)])
    def __sub__(s, o): o = V.of(o); return V(*[a - b for a, b in zip(s.c, o.c)])
    def __neg__(s): return V(*[-a for a in s.c])
    def scale(s, k): k = E.of(k); return V(*[a * k for a in s.c])
    def dot(s, o): o = V.of(o); x = [a * b for a, b in zip(s.c, o.c)]; return x[0] + (x[1] + x[2])
    def sqnorm(s): return s.dot(s)
    def norm(s): return esqrt(s.sqnorm())
    def cross(s, o):
        o = V.of(o); l, r = s.c, o.c
        return V(l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2], l[0] * r[1] - l[1] * r[0])
    def arr(s): return np.stack([a.val() for a in s.c], -1)


if __name__ == "__main__":
    print("fma32 selftest", selftest())
