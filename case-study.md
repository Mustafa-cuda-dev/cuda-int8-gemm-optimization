```markdown
# Case Study: INT8 Quantized GEMM Kernel for NVIDIA T4 (sm_75)

**Author:** Mustafa-cuda-dev  
**Repository:** [cuda-int8-gemm-optimization](https://github.com/Mustafa-cuda-dev/cuda-int8-gemm-optimization)  
**Hardware:** NVIDIA T4 (sm_75)  
**Goal:** Build a production‑ready, fused INT8 matrix multiplication kernel using Tensor Cores with zero register spills and full correctness validation.

---

## 1. Project Overview

General Matrix Multiplication (GEMM) is the core computational primitive in deep learning inference. Quantization to INT8 reduces memory bandwidth and increases throughput, but writing an efficient INT8 GEMM kernel for Tensor Cores requires careful management of memory hierarchy, alignment, and register pressure.

This project implements a fused INT8 GEMM with per‑tensor symmetric quantization. The kernel loads 8‑bit inputs, performs integer multiplication via `wmma::mma_sync`, accumulates into INT32, then scales, rounds, and saturates to INT8 in a single pass – eliminating unnecessary global memory round‑trips.

---

## 2. Technical Challenges & Solutions

### 2.1. Alignment Constraints for WMMA
- **Problem:** `wmma::load_matrix_sync` requires the leading dimension (stride) to be a multiple of 16 bytes for INT8 data. Initially, a stride of **84 bytes** (64 + 20 padding) was used, causing unaligned vector loads and silent memory corruption.
- **Solution:** Removed padding entirely. Set `SHMEM_A_STRIDE = TILE_K = 64` and `SHMEM_B_STRIDE = TILE_N = 64`. This satisfies the 16‑byte alignment requirement and reduces shared memory footprint.

### 2.2. Register Spills Destroying Performance
- **Problem:** First compilation showed `240 bytes stack frame, 240 spill stores, 256 spill loads`. Local memory accesses are ~400 ns vs registers at ~0.3 ns – a 1,300× slowdown. This limited throughput to just 6.14 TOPS.
- **Root Cause:** `__launch_bounds__(128, 8)` forced 64 registers per thread to allow 8 blocks per SM, but the kernel logic exceeded this, forcing spills.
- **Solution:** Relaxed launch bounds to `__launch_bounds__(128, 4)`, allowing up to 128 registers per thread. Removed aggressive `#pragma unroll` from memory‑intensive loops (`load_tile` and `k_inst` loop) to reduce live variable counts. This achieved **0 bytes stack frame, 0 spills** – fully register‑resident.

### 2.3. Undefined Behavior from Out‑of‑Bounds Pointer Arithmetic
- **Problem:** Computing `&A[row * K + col]` before verifying `row < M` is undefined behavior. Compilers can legally remove subsequent safety checks.
- **Solution:** Replaced direct pointer construction with `uintptr_t` offset calculations. The kernel now only dereferences the address after confirming `row_in_bounds` and `col + 3 < K`.

### 2.4. Shared Memory Bank Conflicts
- **Problem:** A stride of 84 bytes (21 words) caused a 2‑way bank conflict during vectorized writes because different threads mapped to the same bank.
- **Solution:** Used zero padding (stride = 64 bytes = 16 words). This maps perfectly to 32 banks with no conflicts, maximizing shared memory throughput.

### 2.5. Block‑Level Synchronization Bottleneck
- **Problem:** The writeback phase used `__syncthreads()` to ensure all warp data was ready before global writes. This forced all warps to wait for the slowest warp.
- **Solution:** Replaced with `__syncwarp()` inside the writeback loops. Each warp now independently stages and writes its fragment, allowing faster warps to proceed.

### 2.6. Host‑Side Error Detection
- **Problem:** `cudaGetLastError()` called immediately after launch only catches configuration errors. Runtime faults surfaced later, complicating debugging.
- **Solution:** Added a two‑step validation: `cudaGetLastError()` after launch, then `cudaDeviceSynchronize()`, then a second `cudaGetLastError()` to catch execution faults.

---

## 3. Final Architecture

### Kernel Design
- **Thread Block:** 128 threads (4 warps arranged in a 2×2 grid).
- **Tile Dimensions:** 64×64 (M×N) per block, K step = 64.
- **Shared Memory:** 8 KB total (two 4 KB tiles for A and B). Zero padding.
- **Compute:** Four warps perform four `16×16×16` WMMA multiplications to cover the 64×64 tile.
- **Writeback:** Warp‑private staging buffers store 16×16 fragments; `__syncwarp()` ensures intra‑warp ordering; each warp writes to global memory in a coalesced manner.

### Quantization Pipeline
1. `INT8` A and B are multiplied → `INT32` accumulator.
2. `INT32` value is converted to float and multiplied by `scale`.
3. Rounded to nearest integer.
4. Clamped to `[-128, 127]`.
5. Stored as `INT8` in C.

---

## 4. Benchmark Results

**Setup:** Google Colab T4, CUDA 11.8, Matrix size 4096×4096×4096 (M=N=K=4096), 100 iterations.

| Metric | Value |
|--------|-------|
| **Kernel execution time** | 20.37 ms |
| **Sustained INT8 throughput** | **6.75 TOPS** |
| **Correctness** | PASS (0 mismatches) |
| **Register spills** | 0 bytes |
| **Shared memory per block** | 8,192 bytes |
| **Registers per thread** | 126 |

**Compiler Report (`nvcc -O3 -arch=sm_75`):**
```

0 bytes stack frame
0 bytes spill stores
0 bytes spill loads
Used 126 registers, 8192 bytes smem

```

---

## 5. What This Demonstrates

1.  **Deep CUDA Expertise:**
    - Understanding of WMMA alignment constraints, register allocation, and occupancy trade‑offs.
    - Ability to read `ptxas` output and diagnose micro‑architecture issues (spills, bank conflicts, coalescing).
2.  **Production‑Grade Correctness:**
    - Comprehensive error handling for host and device.
    - Boundary handling for arbitrary input sizes.
    - Pointer safety to avoid undefined behavior.
3.  **Systematic Optimization:**
    - Structured multiple rounds of correctness and performance audits.
    - Data‑driven decisions based on compiler metrics.
4.  **Self‑Sufficiency:**
    - Complete benchmark harness with CPU reference, bypassing dependency on external libraries.

---

## 6. Lessons Learned

- **Occupancy vs. Register Pressure:** Chasing maximum occupancy is counter‑productive if it forces register spills. Sometimes fewer blocks with more registers yield significantly better performance.
- **Unrolling is a Double‑Edged Sword:** While it improves ILP, forced full unrolling increases register lifetime and can cause spills. Selective unrolling is critical.
- **Hardware Specifications are Mandatory:** WMMA stride alignment and shared memory bank layouts are not optional – violating them leads to silent corruption or crashes.
- **A Strong Audit Pipeline Saves Hours:** Running formal correctness and performance audits before manual testing eliminated numerous hard‑to‑find bugs early.

---

## 7. Future Work

- **Double Buffering:** Overlap global‑to‑shared loads with computation to hide memory latency.
- **Data Layout Optimization:** Support for column‑major and strided layouts.
- **Mixed Precision:** Extend to BF16/FP16 with configurable quantization scales.

---

## 8. Conclusion

This project delivered a **fully functional, optimised, and validated** INT8 Quantized GEMM kernel for NVIDIA T4. The final implementation achieves **6.75 TOPS** with **zero register spills**, full correctness, and robust error handling. The journey demonstrates rigorous engineering discipline, from identifying hardware misalignment and undefined behavior to resolving complex register pressure issues – resulting in a production‑ready kernel suitable for high‑performance inference pipelines.
```

---
