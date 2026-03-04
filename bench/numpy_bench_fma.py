import numpy as np
import time

def benchmark_fma(N=1000000, iters=1000):
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    print(f"NumPy FMA Benchmark (N={N}, {iters} iters)")
    
    # Standard: a * b + c (creates temporary)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = a * b + c
    t_std = (time.perf_counter() - t0) * 1e6 / iters
    print(f"Standard (a*b+c):   {t_std:.2f} us")

    # In-place/Out: add(mul(a,b), c, out=out)
    # This is usually the fastest NumPy way
    t0 = time.perf_counter()
    for _ in range(iters):
        np.add(np.multiply(a, b), c, out=out)
    t_opt = (time.perf_counter() - t0) * 1e6 / iters
    print(f"Optimized (np.add): {t_opt:.2f} us")

if __name__ == "__main__":
    benchmark_fma()
