README.md – INT8 Quantized GEMM Kernel for T4

---

```markdown
# INT8 Quantized GEMM Kernel for NVIDIA T4 (sm_75)

[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-T4-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production‑ready, fused INT8 matrix multiplication (GEMM) kernel optimised for **NVIDIA T4** GPUs (sm_75) using Tensor Cores via CUDA WMMA. The kernel performs per‑tensor symmetric quantisation, implements shared‑memory tiling, warp‑level synchronisation, and full boundary handling – **all with zero register spills**.

---

## 📁 Repository Structure

```

cuda-int8-gemm-optimization/
├── README.md
├── int8_gemm.cu          # Complete kernel + benchmark + CPU reference
├── LICENSE
└── docs/
└── case-study.md     # Detailed optimisation journey

```

---

## • Key Features

- **Tensor Core acceleration** – uses `wmma::mma_sync` for INT8 × INT8 → INT32 accumulation.
- **Shared memory tiling** – 64×64 tiles with zero padding, 8 KB per block, enabling up to 8 blocks per SM.
- **Zero register spills** – achieved through careful `__launch_bounds__` tuning and selective unrolling.
- **Fused quantisation** – scaling, rounding, and saturation are performed inline, eliminating an extra pass.
- **Full boundary handling** – works for any matrix dimensions (not multiples of 64) with correct padding.
- **Warp‑private writeback** – each warp streams its 16×16 accumulator fragment to global memory using `__syncwarp()`.
- **Robust host‑side error checking** – every CUDA API call validated, including asynchronous kernel errors.

---

## 🏆 Performance

**Hardware:** NVIDIA T4 (sm_75) on Google Colab  
**Matrix size:** 4096 × 4096 × 4096 (M=N=K=4096)

| Metric | Value |
|--------|-------|
| **Kernel execution time** | 20.37 ms (average over 100 runs) |
| **Sustained INT8 throughput** | **6.75 TOPS** |
| **Correctness** | PASS (0 mismatches vs CPU reference) |
| **Register spills** | 0 bytes (stack frame, spills, loads) |
| **Shared memory per block** | 8,192 bytes |
| **Registers per thread** | 126 |

**Compiler output (nvcc -O3 -arch=sm_75):**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 126 registers, 8192 bytes smem

```

---

## • Compilation & Usage

### Prerequisites
- CUDA Toolkit 11.8 or later
- NVIDIA driver supporting sm_75 (T4)

### Build
```bash
nvcc -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -o int8_gemm int8_gemm.cu
```

Run

```bash
./int8_gemm
```

The benchmark runs a 4096×4096×4096 GEMM, verifies correctness against a CPU reference, and reports execution time and throughput.

---

📖 How It Works

1. Tiling: Each thread block handles a 64×64 output tile; K dimension is processed in 64‑element steps.
2. Load: Global memory is loaded into shared memory using vectorised 32‑bit accesses (with safe fallback for unaligned/boundary cases).
3. Compute: Four warps per block cooperate to perform 16×16×16 WMMA multiplications using Tensor Cores.
4. Writeback: Each warp stages its 16×16 accumulator fragment to a private shared‑memory segment, then writes to global memory in a coalesced manner with scaling and rounding.
5. Synchronisation: Warp‑level __syncwarp() is used in the writeback phase to avoid cross‑warp stalls.

---

• Correctness Validation

The kernel is validated against a naive CPU implementation using random int8 inputs in the range [-12, 12]. The test passes if every output element differs by at most 1 (due to rounding differences). The benchmark reports [SUCCESS] when all checks pass.

---

• Optimisation Journey

This kernel underwent multiple rounds of rigorous correctness and performance audits. Key fixes include:

· Stride alignment: Ensured wmma::load_matrix_sync strides are multiples of 16 bytes.
· Register spills: Adjusted __launch_bounds__ from (128,8) to (128,4) and removed excessive unrolling to eliminate spills.
· Pointer safety: Replaced out‑of‑bounds pointer construction with uintptr_t‑based address computation.
· Bank conflicts: Used zero padding (stride = 64 bytes) for bank‑conflict‑free shared memory access.
· Error handling: Added comprehensive CUDA error checking for both launch and runtime errors.

See case-study.md for the full optimisation story.

---

• License

This project is licensed under the MIT License – see the LICENSE file for details.

---

• Author

Mustafa-cuda-dev
GitHub • LinkedIn

---

• Acknowledgements

· NVIDIA CUDA Toolkit and WMMA documentation
· Google Colab for providing free T4 GPU access
· The open‑source CUDA community

```

---

