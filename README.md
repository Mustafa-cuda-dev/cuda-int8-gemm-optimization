```
# INT8 Quantized GEMM Kernel for NVIDIA T4 (sm_75)

A production-ready, fused INT8 matrix multiplication kernel optimized for NVIDIA T4 GPUs using Tensor Cores. Implements shared-memory tiling, warp-level synchronization, full boundary handling, and zero register spills.

---

## Repository Structure

```

cuda-int8-gemm-optimization/
├── README.md
├── int8_gemm.cu          # Complete kernel + benchmark + CPU reference
├── LICENSE
└── docs/
└── case-study.md     # Detailed optimization journey

```

---

## Key Features

- Tensor Core acceleration using wmma::mma_sync for INT8 × INT8 → INT32 accumulation.
- Shared memory tiling – 64×64 tiles with zero padding, 8 KB per block, enabling up to 8 blocks per SM.
- Zero register spills – achieved through careful __launch_bounds__ tuning and selective unrolling.
- Fused quantization – scaling, rounding, and saturation performed inline.
- Full boundary handling – works for any matrix dimensions, not just multiples of 64.
- Warp-private writeback – each warp writes its 16×16 fragment independently using __syncwarp().
- Robust host-side error checking – every CUDA API call validated.

---

## Performance

Hardware: NVIDIA T4 (sm_75) on Google Colab
Matrix size: 4096 × 4096 × 4096 (M=N=K=4096)

- Kernel execution time: 20.37 ms (average over 100 runs)
- Sustained INT8 throughput: 6.75 TOPS
- Correctness: PASS (0 mismatches vs CPU reference)
- Register spills: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
- Shared memory per block: 8192 bytes
- Registers per thread: 126

Compiler output (nvcc -O3 -arch=sm_75):
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 126 registers, 8192 bytes smem

```

---

## Compilation & Usage

Prerequisites:
- CUDA Toolkit 11.8 or later
- NVIDIA driver supporting sm_75 (T4)

Build:
```

nvcc -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -o int8_gemm int8_gemm.cu

```

Run:
```

./int8_gemm

```

The benchmark runs a 4096×4096×4096 GEMM, verifies correctness against a CPU reference, and reports execution time and throughput.

---

## How It Works

1. Tiling – each block handles a 64×64 output tile; K dimension in 64‑element steps.
2. Load – global memory loaded into shared memory using vectorized 32‑bit reads with safe fallback for boundaries.
3. Compute – four warps cooperate to execute 16×16×16 WMMA multiplications on Tensor Cores.
4. Writeback – each warp stages its 16×16 fragment to private shared memory, then writes coalesced to global with scaling and rounding.
5. Synchronization – warp‑level __syncwarp() avoids cross‑warp stalls.

---

## Correctness Validation

The kernel is validated against a naive CPU implementation using random int8 inputs in the range [-12, 12]. The test passes if every output element differs by at most 1 (due to rounding differences). The benchmark reports SUCCESS when all checks pass.

---

## Optimization Journey

Multiple rounds of rigorous audits fixed:

- Stride alignment – ensured load_matrix_sync strides are multiples of 16 bytes.
- Register spills – changed __launch_bounds__ from (128,8) to (128,4) and removed excessive unrolling.
- Pointer safety – replaced out‑of‑bounds pointer construction with uintptr_t address computation.
- Bank conflicts – used zero padding (stride 64 bytes) for conflict‑free shared memory.
- Error handling – added comprehensive CUDA error checks for launch and runtime faults.

See docs/case-study.md for the full story.

---

## License

MIT License – see LICENSE file.

---

## Author

Mustafa-cuda-dev  
GitHub: https://github.com/Mustafa-cuda-dev

---

## Acknowledgements

- NVIDIA CUDA Toolkit and WMMA documentation
- Google Colab for free T4 GPU access
- The open‑source CUDA community
```

---

