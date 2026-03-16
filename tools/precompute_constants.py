import math
import numpy as np

RNS_Q = [12289, 40961, 65537, 114689, 163841]
Q     = 12289
T     = 16
N     = 1024
DELTA = Q // T
K     = len(RNS_Q)

def modinv(a, m):
    return pow(int(a), int(m) - 2, int(m))

def compute_garner():
    print("=" * 60)
    print("GARNER CONSTANTS")
    print("=" * 60)
    invs = [0]
    prod = RNS_Q[0]
    for k in range(1, K):
        inv_k = modinv(prod % RNS_Q[k], RNS_Q[k])
        invs.append(inv_k)
        assert (prod % RNS_Q[k] * inv_k) % RNS_Q[k] == 1
        print(f"lane {k}: q={RNS_Q[k]}  inv={inv_k}  check OK")
        prod *= RNS_Q[k]
    print("\n__constant__ uint64_t GARNER_INV[RNS_K] = {")
    for v in invs:
        print(f"    {v}ULL,")
    print("};")
    S = 38
    print("\n__constant__ uint64_t BARRETT_K[RNS_K] = {")
    for q in RNS_Q:
        print(f"    {(1<<S)//q}ULL,")
    print("};")
    return invs

def compute_chebyshev(degree=27):
    from numpy.polynomial import chebyshev as C
    print(f"\nCHEBYSHEV degree={degree}")
    xmax = Q / 2.0
    xs   = np.linspace(-xmax, xmax, 100000)
    ys_r = np.round(xs / DELTA) % T
    ys   = np.where(ys_r > T/2, ys_r - T, ys_r).astype("float64")
    cs   = C.chebfit(xs / xmax, ys, degree)
    max_e = np.max(np.abs(ys - C.chebval(xs / xmax, cs)))
    status = "OK" if max_e < DELTA/2 else "INSUFFICIENT"
    print(f"max_err={max_e:.4f}  threshold={DELTA//2}  status={status}")
    cs_int = [int(round(c * (1 << 20))) for c in cs]
    print(f"\n__constant__ int32_t CHEB_COEFFS[{degree+1}] = {{")
    for i, c in enumerate(cs_int):
        print(f"    {c:12d},  /* c{i} */")
    print("};")
    return cs_int

def analyze():
    print("\nBUDGET ANALYSIS")
    b = int(math.log2(Q / (2 * math.sqrt(N))))
    c = 1 + math.ceil(math.log2(27))
    print(f"initial budget ~{b} bits  circuit depth={c}  margin={b-c}")

if __name__ == "__main__":
    compute_garner()
    compute_chebyshev(27)
    analyze()
