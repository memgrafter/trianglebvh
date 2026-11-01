"""Compare trianglebvh's selection order with GNU libstdc++'s std::nth_element (tools/nth_oracle.cpp, built with g++).

    g++ -O2 -o /tmp/nth_oracle tools/nth_oracle.cpp && python -I tools/check_select.py /tmp/nth_oracle
"""
import ctypes, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trianglebvh


def cases(rng):
    for n in list(range(1, 70)) + [100, 257, 1000, 4097, 20000]:
        for kind in ("random", "few", "sorted", "reverse", "equal", "organ", "killer"):
            if kind == "random": k = rng.random(n)
            elif kind == "few": k = rng.integers(0, 4, n)
            elif kind == "sorted": k = np.arange(n)
            elif kind == "reverse": k = np.arange(n)[::-1]
            elif kind == "equal": k = np.zeros(n)
            elif kind == "organ": k = np.minimum(np.arange(n), n - np.arange(n))
            else:   # a median-of-3 killer (Musser 1997) drives introselect to its depth limit
                m = n // 2 * 2; k = np.zeros(n)
                for i in range(1, m // 2 + 1):
                    if i % 2: k[i - 1] = i; k[i] = m // 2 + i
                    k[m // 2 + i - 1] = 2 * i
            for nth in sorted({0, n // 2, n - 1, n // 3}):
                yield np.asarray(k, np.float32), nth


def main():
    lib = trianglebvh._load()
    lib.tb_nth_element_test.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    allc = list(cases(np.random.default_rng(0)))
    text = "".join(f"{len(k)} {nth}\n" + " ".join(repr(float(x)) for x in k) + "\n" for k, nth in allc)
    out = subprocess.run([sys.argv[1]], input=text, capture_output=True, text=True, check=True).stdout.split("\n")
    bad = 0
    for (k, nth), line in zip(allc, out):
        ref = np.array(line.split(), np.uint32)
        p = np.empty(len(k), np.uint32)
        lib.tb_nth_element_test(k.ctypes.data, len(k), nth, p.ctypes.data)
        if not np.array_equal(p, ref):
            bad += 1
            if bad <= 5: print("differs: n", len(k), "nth", nth, "first", p[:10], ref[:10])
    print(f"{len(allc)} cases, {bad} differ")


if __name__ == "__main__":
    main()
