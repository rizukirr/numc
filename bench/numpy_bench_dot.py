import numpy as np
import time

def benchmark_dot(N=1000000, iters=1000):
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    print(f"NumPy Dot Product Benchmark (N={N}, {iters} iters)")
    
    # Standard: np.dot (handles multi-dimensional, but for 1D it's a dot product)
    t0 = time.perf_counter()
    for _ in range(iters):
        res = np.dot(a, b)
    t_dot = (time.perf_counter() - t0) * 1e6 / iters
    print(f"np.dot:  {t_dot:.2f} us")

    # np.vdot: specifically for vectors (dot product)
    t0 = time.perf_counter()
    for _ in range(iters):
        res = np.vdot(a, b)
    t_vdot = (time.perf_counter() - t0) * 1e6 / iters
    print(f"np.vdot: {t_vdot:.2f} us")

if __name__ == "__main__":
    benchmark_dot()
