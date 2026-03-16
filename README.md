# gpu-bfv-bootstrap

GPU-accelerated BFV bootstrapping — noise budget reset for infinite-depth FHE.

## Results — NVIDIA GeForce RTX 2060 Max-Q

| Metric | Value |
|---|---|
| Bootstrap latency | 0.8ms mean |
| Throughput | 1311 bootstraps/sec |
| Noise budget restored | 13 bits (full) |
| Muls before exhaustion | 3 squarings |
| GPU | RTX 2060 Max-Q, 6GB VRAM |

## What this proves

A BFV ciphertext multiplied until its noise budget hits zero can be fully
restored in under 1ms on a consumer GPU. This enables unbounded-depth
homomorphic computation — the mathematical requirement for a fully
homomorphic encryption scheme.

## Architecture

Three-phase bootstrapping circuit:

    Dead ciphertext (noise = 0)
         |
    Phase 1: RNS lift
         Q=12289 → 5-prime basis, product ~6.2e23 (71 bits)
         |
    Phase 2: Homomorphic decryption
         v = ct0 + ct1 * s  (inner product mod Q)
         |
    Phase 3: EvalMod
         round(v / Delta) mod T  via degree-27 Chebyshev approximation
         |
    Fresh ciphertext (13 bits noise budget restored)

## Test results

    [OK] RNS round-trip identity
    [OK] RNS polynomial addition
    [OK] Chebyshev EvalMod approximation (max_err=7.87 < threshold=384)
    [OK] Noise budget restored after bootstrap
    [OK] 30 total muls, 3 bootstraps — unbounded depth demonstrated

## Run

    git clone https://github.com/samfrazerdutton/gpu-fhe-net ../gpu-fhe-net
    python3 -m venv fhe-env && source fhe-env/bin/activate
    pip install cupy-cuda12x numpy
    export PYTHONPATH="$HOME/gpu-fhe-net:$HOME/gpu-bfv-bootstrap:$PYTHONPATH"
    python tests/test_noise_reset.py

## Related

- [gpu-fhe-net](https://github.com/samfrazerdutton/gpu-fhe-net) — BFV neural network inference
- [cuFHE-lite](https://github.com/samfrazerdutton/cufhe-lite) — core GPU BFV library
