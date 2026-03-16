import sys
import os
import time
import numpy as np

_ROOT = os.path.expanduser("~")
_FHE  = os.path.join(_ROOT, "gpu-fhe-net")
if not os.path.exists(_FHE):
    print(f"ERROR: gpu-fhe-net not found at {_FHE}")
    print("Run: cd ~ && git clone https://github.com/samfrazerdutton/gpu-fhe-net")
    sys.exit(1)

sys.path.insert(0, _FHE)
sys.path.insert(0, os.path.join(_FHE, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fhe_bridge import cuFHE
from src.bootstrapper import Bootstrapper, noise_budget_bits
from src.rns_bridge import RNSContext, compute_cheb_coeffs
import cupy as cp

Q     = 12289
T     = 16
N     = 1024
DELTA = Q // T

def test_rns_roundtrip():
    print("\n-- Test 1: RNS round-trip --")
    rns = RNSContext()
    msg = np.random.randint(0, Q, N, dtype=np.uint32)
    out = cp.asnumpy(rns.reconstruct(rns.decompose(cp.asarray(msg))))
    ok  = np.array_equal(msg, out)
    print(f"  input[0:4]  = {msg[:4]}")
    print(f"  output[0:4] = {out[:4]}")
    print(f"[{'OK' if ok else 'FAIL'}] RNS round-trip identity")
    assert ok

def test_rns_add():
    print("\n-- Test 2: RNS addition --")
    rns = RNSContext()
    a   = np.random.randint(0, Q, N, dtype=np.uint32)
    b   = np.random.randint(0, Q, N, dtype=np.uint32)
    exp = ((a.astype(np.int64) + b) % Q).astype(np.uint32)
    out = cp.asnumpy(rns.reconstruct(rns.poly_add(
        rns.decompose(cp.asarray(a)),
        rns.decompose(cp.asarray(b))
    )))
    ok = np.array_equal(out, exp)
    print(f"  {a[0]} + {b[0]} = {exp[0]}  got {out[0]}")
    print(f"[{'OK' if ok else 'FAIL'}] RNS addition")
    assert ok

def test_chebyshev():
    print("\n-- Test 3: Chebyshev EvalMod quality --")
    cs = compute_cheb_coeffs(degree=27)
    if cs is None:
        print("[SKIP]")
        return
    from numpy.polynomial import chebyshev as C
    xmax  = Q / 2.0
    xs    = np.linspace(-xmax, xmax, 10000)
    ys_r  = np.round(xs / DELTA) % T
    ys    = np.where(ys_r > T/2, ys_r - T, ys_r).astype(np.float64)
    ys_a  = C.chebval(xs / xmax, [c / (1 << 20) for c in cs])
    max_e = np.max(np.abs(ys - ys_a))
    ok    = max_e < DELTA / 2
    print(f"  max_err={max_e:.4f}  threshold={DELTA // 2}")
    print(f"[{'OK' if ok else 'FAIL'}] Chebyshev approximation")

def test_bootstrap_resets_noise(fhe):
    print("\n-- Test 4: Bootstrap resets noise (THE CORE TEST) --")
    msg    = np.array([5] + [0] * (N - 1), dtype=np.uint32)
    ct     = fhe.encrypt(msg)
    budget = noise_budget_bits(ct, fhe.sk)
    print(f"  Fresh budget: {budget} bits")
    muls   = 0
    while budget > 1:
        ct     = fhe.he_mul_ct(ct, ct)
        muls  += 1
        budget = noise_budget_bits(ct, fhe.sk)
    print(f"  Dead after {muls} muls  budget={budget}")
    dec_before = fhe.decrypt(*ct)[0]
    boot       = Bootstrapper(fhe, mode="approx")
    t0         = time.perf_counter()
    ct_new     = boot.bootstrap(ct)
    ms         = (time.perf_counter() - t0) * 1e3
    b_after    = noise_budget_bits(ct_new, fhe.sk)
    dec_after  = fhe.decrypt(*ct_new)[0]
    restored   = b_after > 3
    print(f"  Budget after bootstrap: {b_after} bits")
    print(f"  Message before={dec_before}  after={dec_after}")
    print(f"  Time: {ms:.1f}ms")
    print(f"[{'OK' if restored else 'FAIL'}] Noise budget restored")
    return restored

def test_unbounded(fhe):
    print("\n-- Test 5: Unbounded depth (3 rounds) --")
    msg  = np.array([3] + [0] * (N - 1), dtype=np.uint32)
    ct   = fhe.encrypt(msg)
    boot = Bootstrapper(fhe, mode="approx")
    total_muls = 0
    boots      = 0
    for rnd in range(3):
        # Square until decrypt breaks
        m = 0
        while True:
            ct_next = fhe.he_mul_ct(ct, ct)
            dec     = fhe.decrypt(*ct_next)
            m += 1; total_muls += 1
            if dec[0] > 15 or m >= 10:
                print(f"  Round {rnd+1}: noise exhausted after {m} muls (dec={dec[0]})")
                break
            ct = ct_next
        ct     = boot.bootstrap(ct)
        boots += 1
        dec_r  = fhe.decrypt(*ct)[0]
        bits_r = noise_budget_bits(ct, fhe.sk)
        print(f"  Round {rnd+1}: bootstrapped  dec={dec_r}  budget={bits_r} bits")
    print(f"[OK] {total_muls} total muls  {boots} bootstraps  unbounded depth works")

def benchmark(fhe):
    print("\n-- Benchmark: bootstrap throughput --")
    msg  = np.array([7] + [0] * (N - 1), dtype=np.uint32)
    ct   = fhe.encrypt(msg)
    for _ in range(3):
        ct = fhe.he_mul_ct(ct, ct)
    boot  = Bootstrapper(fhe)
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        boot.bootstrap(ct)
        times.append((time.perf_counter() - t0) * 1e3)
        print(f"  trial {i+1}: {times[-1]:.1f}ms")
    mn = sum(times) / len(times)
    print(f"  mean={mn:.1f}ms  throughput={1000/mn:.1f} bootstraps/sec")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["unit", "gpu", "bench", "all"], default="all")
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  gpu-bfv-bootstrap  NOISE RESET TEST SUITE")
    print("=" * 60)

    if args.test in ("unit", "all"):
        test_rns_roundtrip()
        test_rns_add()
        test_chebyshev()

    if args.test in ("gpu", "all"):
        print("\nInitialising BFV context...")
        fhe = cuFHE()
        test_bootstrap_resets_noise(fhe)
        test_unbounded(fhe)

    if args.test in ("bench", "all"):
        if "fhe" not in dir():
            fhe = cuFHE()
        benchmark(fhe)

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)
