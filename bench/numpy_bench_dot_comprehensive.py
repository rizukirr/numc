import numpy as np
import time

def bench_1d(N, iters):
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    
    # Warmup
    np.dot(a, b)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(a, b)
    t = (time.perf_counter() - t0) * 1e6 / iters
    
    gflops = (2.0 * N) / (t * 1e-6) / 1e9
    print(f"1D Dot (N={N}): {t:8.2f} us | {gflops:6.2f} GFLOPS")
    return t

def bench_2d(M, K, N, iters):
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    
    np.dot(a, b)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(a, b)
    t = (time.perf_counter() - t0) * 1e6 / iters
    
    gflops = (2.0 * M * K * N) / (t * 1e-6) / 1e9
    print(f"2D Dot ({M}x{K} . {K}x{N}): {t:8.2f} us | {gflops:6.2f} GFLOPS")
    return t

def bench_nd(iters):
    # (10, 100) . (20, 100, 50) -> (10, 20, 50)
    a = np.random.randn(10, 100).astype(np.float32)
    b = np.random.randn(20, 100, 50).astype(np.float32)
    
    np.dot(a, b)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(a, b)
    t = (time.perf_counter() - t0) * 1e6 / iters
    
    ops = 2.0 * 10 * 20 * 100 * 50
    gflops = ops / (t * 1e-6) / 1e9
    print(f"ND Dot (10x100 . 20x100x50): {t:8.2f} us | {gflops:6.2f} GFLOPS")
    return t

if __name__ == "__main__":
    print("--- NumPy Performance Benchmark ---")
    bench_1d(1000, 10000)
    bench_1d(1000000, 100)
    
    bench_2d(128, 128, 128, 1000)
    bench_2d(512, 512, 512, 100)
    bench_2d(1024, 1024, 1024, 10)
    
    bench_nd(100)
