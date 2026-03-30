# numc Roadmap

numc aims to be a production-grade tensor primitive library for machine learning,
deep learning, and LLM inference. This document tracks remaining features needed
to support the full ML/DL/LLM stack.

Current state: ~60 operations covering basic tensor math, element-wise ops,
reductions, matmul, broadcasting, and random number generation.

---

## Phase 1: Classical ML + Basic Neural Networks

These operations unlock logistic regression, decision trees, KNN, SVM, RNNs,
and embedding-based models.

### Activations

| Operation | Signature | Notes |
|---|---|---|
| `numc_tanh` | `(a, out)` + inplace | Hyperbolic tangent. RNN/LSTM gates, policy networks. Same X-macro + SIMD pattern as exp/log. |
| `numc_sigmoid` | `(a, out)` + inplace | Logistic function `1/(1+exp(-x))`. Binary classification, gates. Fused implementation avoids overflow. |

### Shape manipulation

| Operation | Signature | Notes |
|---|---|---|
| `numc_concat` | `(arrays[], n, axis, out)` | Concatenate arrays along an axis. Batch assembly, feature stacking, RL replay buffers. |
| `numc_stack` | `(arrays[], n, axis, out)` | Stack arrays along a NEW axis. Similar to concat but inserts a dimension. |
| `numc_split` | `(a, n_or_indices, axis, out[])` | Split array into sub-arrays. Inverse of concat. |
| `numc_squeeze` | `(a, axis)` | Remove size-1 dimensions. In-place shape change. |
| `numc_unsqueeze` | `(a, axis)` | Insert size-1 dimension. In-place shape change. |
| `numc_tile` | `(a, reps[], out)` | Repeat array along each axis. Broadcasting materialization. |
| `numc_pad` | `(a, pad_widths[], mode, out)` | Pad with zeros/constant/reflect. Conv input padding, sequence padding. |
| `numc_flip` | `(a, axis, out)` | Reverse elements along axis. Data augmentation. |

### Indexing

| Operation | Signature | Notes |
|---|---|---|
| `numc_gather` | `(a, indices, axis, out)` | Gather elements by index along axis. Embedding lookup, replay buffer sampling. |
| `numc_scatter` | `(a, indices, axis, values, out)` | Scatter values into array by index. Sparse gradient updates, one-hot generalization. |
| `numc_index_select` | `(a, indices, axis, out)` | Select rows/cols by integer array. Batch selection. |
| `numc_masked_select` | `(a, mask, out)` | Select elements where mask is nonzero. Filtering. |
| `numc_nonzero` | `(a, out)` | Return indices of nonzero elements. Sparse operations. |

### Sorting

| Operation | Signature | Notes |
|---|---|---|
| `numc_sort` | `(a, axis, out)` | Sort elements along axis. Decision tree split finding. |
| `numc_argsort` | `(a, axis, out)` | Return indices that sort the array. KNN, ranking. |
| `numc_topk` | `(a, k, axis, values_out, indices_out)` | Top-k values and indices. Beam search, top-k sampling. |

### Reductions

| Operation | Signature | Notes |
|---|---|---|
| `numc_var` | `(a, out)` + axis variant | Variance. Batch normalization, statistics. |
| `numc_std` | `(a, out)` + axis variant | Standard deviation. Normalization, z-score. |
| `numc_cumsum` | `(a, axis, out)` | Cumulative sum. RL discounted returns, CDF. |
| `numc_cumprod` | `(a, axis, out)` | Cumulative product. Probability chains. |
| `numc_prod` | `(a, out)` + axis variant | Product of elements. Shape computation, probability. |
| `numc_norm` | `(a, ord, axis, out)` | L1/L2/Linf norm. Distance metrics, regularization. Can compose from existing but fused is faster. |
| `numc_logsumexp` | `(a, axis, out)` | Numerically stable log(sum(exp(x))). Softmax, log-probabilities. |

### Type conversion

| Operation | Signature | Notes |
|---|---|---|
| `numc_cast` | `(a, dtype, out)` | Convert array to different dtype. Mixed precision, int-to-float for training. |

### Array creation

| Operation | Signature | Notes |
|---|---|---|
| `numc_arange` | `(ctx, start, stop, step, dtype)` | Evenly spaced values. Index generation, positional encoding base. |
| `numc_linspace` | `(ctx, start, stop, n, dtype)` | Linearly spaced values. Visualization, interpolation. |
| `numc_eye` | `(ctx, n, dtype)` | Identity matrix. Initialization, linear algebra. |
| `numc_tril` | `(a, k, out)` | Lower triangular. Causal attention masks. |
| `numc_triu` | `(a, k, out)` | Upper triangular. Masking. |

### Random

| Operation | Signature | Notes |
|---|---|---|
| `numc_randint` | `(ctx, low, high, shape, dtype)` | Random integers. Index sampling, data augmentation. |
| `numc_shuffle` | `(a, axis)` | Shuffle elements along axis. In-place. Data loading, SGD. |
| `numc_choice` | `(ctx, a, n, replace, out)` | Random sampling from array. Bootstrap, stochastic methods. |

### Logic

| Operation | Signature | Notes |
|---|---|---|
| `numc_logical_and` | `(a, b, out)` | Element-wise AND. Mask combination. |
| `numc_logical_or` | `(a, b, out)` | Element-wise OR. Mask combination. |
| `numc_logical_not` | `(a, out)` | Element-wise NOT. Mask inversion. |
| `numc_isnan` | `(a, out)` | NaN detection. Gradient debugging. |
| `numc_isinf` | `(a, out)` | Infinity detection. Overflow handling. |

---

## Phase 2: Deep Learning

These operations enable convolutional networks, modern activations, and
efficient training.

### Activations

| Operation | Signature | Notes |
|---|---|---|
| `numc_gelu` | `(a, out)` + inplace | Gaussian Error Linear Unit. Default activation in transformers (BERT, GPT). Needs erf. |
| `numc_silu` | `(a, out)` + inplace | SiLU/Swish: `x * sigmoid(x)`. Used in Llama, modern architectures. |
| `numc_leaky_relu` | `(a, alpha, out)` | Leaky ReLU. GANs, prevents dead neurons. |

### Fused operations

| Operation | Signature | Notes |
|---|---|---|
| `numc_softmax` | `(a, axis, out)` | Numerically stable softmax (fused max-sub-exp-sum-div). Every classifier, every attention layer. |
| `numc_log_softmax` | `(a, axis, out)` | Fused log-softmax. NLL loss, numerically stable. |
| `numc_cross_entropy` | `(logits, labels, out)` | Fused softmax + cross-entropy. 2x faster than separate ops, better numerical stability. |
| `numc_layer_norm` | `(a, gamma, beta, axis, eps, out)` | Fused layer normalization. 3x faster than composing mean/var/sub/div/mul/add. |
| `numc_rms_norm` | `(a, gamma, eps, out)` | Root mean square normalization. Used in Llama, faster than layer_norm. |

### Convolution

| Operation | Signature | Notes |
|---|---|---|
| `numc_conv1d` | `(input, weight, bias, stride, padding, out)` | 1D convolution. Audio, time series, local attention. |
| `numc_conv2d` | `(input, weight, bias, stride, padding, out)` | 2D convolution. CNNs, image processing. Implement via im2col + matmul. |
| `numc_max_pool2d` | `(input, kernel, stride, out)` | Max pooling. CNN feature extraction. |
| `numc_avg_pool2d` | `(input, kernel, stride, out)` | Average pooling. Global average pooling in modern CNNs. |

### Linear algebra

| Operation | Signature | Notes |
|---|---|---|
| `numc_bmm` | `(a, b, out)` | Batched matrix multiply. Multi-head attention: `(B,H,S,D) @ (B,H,D,S)`. Critical for transformers. |
| `numc_outer` | `(a, b, out)` | Outer product. Rank-1 updates, attention patterns. |
| `numc_trace` | `(a, out)` | Matrix trace. Regularization. |

### Math

| Operation | Signature | Notes |
|---|---|---|
| `numc_sin` | `(a, out)` | Sine. Positional encoding (sinusoidal). |
| `numc_cos` | `(a, out)` | Cosine. Positional encoding. |
| `numc_erf` | `(a, out)` | Error function. GELU activation: `0.5 * x * (1 + erf(x / sqrt(2)))`. |
| `numc_rsqrt` | `(a, out)` | Reciprocal square root: `1/sqrt(x)`. RMS normalization, attention scaling. |
| `numc_reciprocal` | `(a, out)` | Element-wise `1/x`. Normalization divisors. |
| `numc_floor` | `(a, out)` | Floor. Index computation. |
| `numc_ceil` | `(a, out)` | Ceiling. Padding computation. |

### Random

| Operation | Signature | Notes |
|---|---|---|
| `numc_dropout_mask` | `(ctx, shape, p, dtype)` | Generate dropout mask (Bernoulli). Training regularization. |
| `numc_bernoulli` | `(ctx, probs, out)` | Sample from Bernoulli distribution. Stochastic operations. |

---

## Phase 3: Transformers + LLM Inference

These operations enable efficient transformer inference and training.

### Attention

| Operation | Signature | Notes |
|---|---|---|
| `numc_scaled_dot_attention` | `(Q, K, V, mask, out)` | Fused scaled dot-product attention. Can compose from bmm+softmax+mask but fused is 5-10x faster due to memory. |
| `numc_flash_attention` | `(Q, K, V, mask, out)` | Memory-efficient attention (Dao et al.). THE key LLM optimization. Tiles computation to stay in SRAM. |
| `numc_kv_cache_append` | `(cache, new_kv, pos)` | Append to KV cache for autoregressive generation. Avoids recomputation. |

### Positional encoding

| Operation | Signature | Notes |
|---|---|---|
| `numc_rope` | `(a, freqs, out)` | Rotary Position Embedding. Used in Llama, Mistral, modern LLMs. Needs sin/cos + element interleaving. |
| `numc_sinusoidal_pe` | `(ctx, seq_len, dim)` | Generate sinusoidal positional encoding matrix. Original transformer. |

### Sampling / generation

| Operation | Signature | Notes |
|---|---|---|
| `numc_topk` | `(a, k, axis, vals, idxs)` | Top-k values and indices. (Also in Phase 1.) |
| `numc_multinomial` | `(probs, n_samples, out)` | Sample from categorical distribution. Token sampling. |
| `numc_temperature_scale` | `(logits, temp, out)` | Scale logits by temperature. Generation control. Can compose from div_scalar. |

### Quantization

| Operation | Signature | Notes |
|---|---|---|
| `numc_quantize_int8` | `(a, scale, zero_point, out)` | Quantize float to int8. Model compression. |
| `numc_dequantize_int8` | `(a, scale, zero_point, out)` | Dequantize int8 to float. Inference. |
| `numc_matmul_int4` | `(a_fp, b_int4, scales, out)` | Mixed-precision matmul. GPTQ/AWQ-style inference. |

### New dtypes

| Dtype | Notes |
|---|---|
| `NUMC_DTYPE_FLOAT16` | Half precision. 2x memory savings, faster on GPUs. |
| `NUMC_DTYPE_BFLOAT16` | Brain float. Same range as float32, less precision. Training standard for LLMs. |

---

## Phase 4: Full Ecosystem

Advanced features for a complete numerical computing library.

### Linear algebra (dense)

| Operation | Signature | Notes |
|---|---|---|
| `numc_svd` | `(a, U, S, Vt)` | Singular value decomposition. PCA, low-rank approximation, LoRA. |
| `numc_eig` | `(a, eigenvalues, eigenvectors)` | Eigendecomposition. Spectral clustering, PCA. |
| `numc_solve` | `(A, b, out)` | Solve linear system Ax=b. Ridge regression, Gaussian processes. |
| `numc_inv` | `(a, out)` | Matrix inverse. Gaussian distribution, Mahalanobis distance. |
| `numc_det` | `(a, out)` | Determinant. Probability density. |
| `numc_cholesky` | `(a, out)` | Cholesky decomposition. Gaussian processes, positive definite systems. |
| `numc_qr` | `(a, Q, R)` | QR decomposition. Least squares, orthogonalization. |

### Sparse arrays

| Feature | Notes |
|---|---|
| CSR/CSC sparse format | Graph neural networks, recommendation systems, NLP (sparse attention). |
| Sparse matmul | SpMM, SpMV for GNN message passing. |
| Sparse-dense conversion | `to_sparse`, `to_dense`. |

### Autograd (optional layer)

| Feature | Notes |
|---|---|
| Computation graph recording | Track operations for automatic differentiation. |
| Backward pass dispatch | Call per-op backward functions. |
| Gradient accumulation | Handle parameter updates. |

Note: Autograd is a **framework concern**, not a tensor primitive. It could be
a separate library (`numc-autograd`) built on top of numc, similar to how
PyTorch's autograd is built on ATen tensors. Including it here for completeness.

---

## Implementation priority

```
Now:     tanh, sigmoid, concat, cast, sort, argsort, gather, scatter
         (8 ops — unlocks logistic regression, decision trees, KNN,
          SVM, RNNs, embedding models)

Next:    gelu, silu, bmm, softmax (fused), var, std, cumsum,
         sin, cos, arange, eye, tril, pad, randint, shuffle
         (15 ops — unlocks CNNs, transformers, modern activations)

Later:   flash_attention, rope, conv2d, layer_norm, rms_norm,
         topk, multinomial, quantization, bf16/fp16
         (enables LLM inference)

Future:  SVD, solve, eigendecomp, sparse, autograd
         (full numerical computing ecosystem)
```

---

## Design principles

1. **Every operation supports all 10 dtypes** via X-macro generators.
2. **Every operation has SIMD implementations** for AVX2, AVX-512, NEON, SVE, RVV.
3. **Fused operations over composed** when the fused version is significantly faster (softmax, layer_norm, cross_entropy).
4. **No external dependencies** — numc remains pure C with zero BLAS/LAPACK/Python dependency.
5. **Arena-based memory** — all allocations through NumcCtx, no hidden mallocs.
6. **Benchmark against NumPy** — every new operation must match or beat NumPy performance.
