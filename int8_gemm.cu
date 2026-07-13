#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

// [HIGH-CONFIDENCE] Optimized INT8 Quantized GEMM kernel specifically tuned for T4 (sm_75) GPUs.
// Reverts to a single-buffer layout (8 KB shared memory per block) to maintain SM occupancy 
// up to 4 blocks/SM (512 threads/SM) to avoid register spills and maintain register-file performance.
// Eliminates alignment violation crashes on load_matrix_sync by enforcing a strict 16-byte multiple 
// stride (64 bytes). Incorporates warp-level synchronization in the writeback phase to bypass costly block-level barriers.
// Selectively restricts loop unrolling on memory transfers and the matrix-instruction step to eliminate register spilling.

using namespace nvcuda;

// Compile Command:
// !nvcc -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -o int8_gemm int8_gemm.cu

// Error handling macro for robust host-side API validation
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Global constants for configuration and dimensions
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
constexpr int WARP_SIZE = 32;
constexpr int WARP_ROWS = 2;
constexpr int WARP_COLS = 2;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Zero padding setup to guarantee 16-byte multiple stride alignment for wmma
constexpr int SHMEM_PAD = 0;
constexpr int SHMEM_A_STRIDE = TILE_K + SHMEM_PAD;   // 64 bytes (aligned to 16-byte multiple)
constexpr int SHMEM_B_STRIDE = TILE_N + SHMEM_PAD;   // 64 bytes (aligned to 16-byte multiple)
constexpr int SHMEM_A_SIZE = TILE_M * SHMEM_A_STRIDE; // 64 * 64 = 4096 bytes
constexpr int SHMEM_B_SIZE = TILE_K * SHMEM_B_STRIDE; // 64 * 64 = 4096 bytes
constexpr int TOTAL_SHMEM_SIZE = SHMEM_A_SIZE + SHMEM_B_SIZE; // 8192 bytes (8 KB)

// Derived loop bounds and mapping steps to avoid hardcoded magic numbers
constexpr int LOAD_ITERATIONS = (TILE_M * TILE_K) / (BLOCK_SIZE * 4); // 8 iterations
constexpr int ROW_DIV = TILE_K / 4;                                  // 16 word blocks
constexpr int COL_MULT = 4;                                          // 4 bytes per word load
constexpr int FRAG_ELEMENTS = WMMA_M * WMMA_N;                       // 256 accumulator staging elements

// Device function to perform coalesced vectorized loading from global memory to shared memory with alignment verification.
// Unrolling is omitted on this helper function to maintain a lower register-pressure footprint.
__device__ __forceinline__ void load_tile(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* shmem_A_dest,
    int8_t* shmem_B_dest,
    const int k_offset,
    const int tid,
    const size_t M,
    const size_t N,
    const size_t K
) {
    for (int i = 0; i < LOAD_ITERATIONS; ++i) {
        const int word_idx = i * BLOCK_SIZE + tid;
        const int row = word_idx / ROW_DIV;
        const int col = (word_idx % ROW_DIV) * COL_MULT;

        const int global_row_A = blockIdx.y * TILE_M + row;
        const int global_col_A = k_offset + col;

        // Secure boundary tracking. Address calculation is calculated via integer offsets 
        // to prevent constructing invalid out-of-bounds pointers, avoiding Undefined Behavior.
        const bool row_in_bounds_A = (global_row_A < static_cast<int>(M));
        const uintptr_t raw_addr_A = reinterpret_cast<uintptr_t>(A) + 
            (static_cast<size_t>(global_row_A) * K + global_col_A) * sizeof(int8_t);
        const bool is_aligned_A = (raw_addr_A & 3) == 0;
        const bool safe_vec_A = row_in_bounds_A && is_aligned_A && (global_col_A + 3 < static_cast<int>(K));

        if (safe_vec_A) {
            const int8_t* ptr_A = reinterpret_cast<const int8_t*>(raw_addr_A);
            const int32_t val = *reinterpret_cast<const int32_t*>(ptr_A);
            *reinterpret_cast<int32_t*>(&shmem_A_dest[row * SHMEM_A_STRIDE + col]) = val;
        } else {
            int8_t bytes_A[4] = {0, 0, 0, 0};
            if (row_in_bounds_A) {
                for (int b = 0; b < 4; ++b) {
                    if (global_col_A + b < static_cast<int>(K)) {
                        bytes_A[b] = A[static_cast<size_t>(global_row_A) * K + (global_col_A + b)];
                    }
                }
            }
            for (int b = 0; b < 4; ++b) {
                shmem_A_dest[row * SHMEM_A_STRIDE + col + b] = bytes_A[b];
            }
        }

        const int global_row_B = k_offset + row;
        const int global_col_B = blockIdx.x * TILE_N + col;

        const bool row_in_bounds_B = (global_row_B < static_cast<int>(K));
        const uintptr_t raw_addr_B = reinterpret_cast<uintptr_t>(B) + 
            (static_cast<size_t>(global_row_B) * N + global_col_B) * sizeof(int8_t);
        const bool is_aligned_B = (raw_addr_B & 3) == 0;
        const bool safe_vec_B = row_in_bounds_B && is_aligned_B && (global_col_B + 3 < static_cast<int>(N));

        if (safe_vec_B) {
            const int8_t* ptr_B = reinterpret_cast<const int8_t*>(raw_addr_B);
            const int32_t val = *reinterpret_cast<const int32_t*>(ptr_B);
            *reinterpret_cast<int32_t*>(&shmem_B_dest[row * SHMEM_B_STRIDE + col]) = val;
        } else {
            int8_t bytes_B[4] = {0, 0, 0, 0};
            if (row_in_bounds_B) {
                for (int b = 0; b < 4; ++b) {
                    if (global_col_B + b < static_cast<int>(N)) {
                        bytes_B[b] = B[static_cast<size_t>(global_row_B) * N + (global_col_B + b)];
                    }
                }
            }
            for (int b = 0; b < 4; ++b) {
                shmem_B_dest[row * SHMEM_B_STRIDE + col + b] = bytes_B[b];
            }
        }
    }
}

// Relaxed launch bounds to target a maximum of 4 blocks per SM (512 threads per SM). 
// This expands the compiler's registers-per-thread ceiling to 128, preventing register spills
// into high-latency local memory and significantly improving overall Tensor Core throughput.
extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, 4) gemm_tensor_core_int8_fused(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float scale,
    int8_t* __restrict__ C,
    const size_t M,
    const size_t N,
    const size_t K
) {
    static_assert(BLOCK_SIZE == 128, "This kernel assumes 128 threads per block (4 warps arranged in a 2x2 grid).");

    // Static sequential Shared Memory buffer layout (8 KB total per block)
    __shared__ alignas(16) int8_t shmem[TOTAL_SHMEM_SIZE];
    int8_t* shmem_A = shmem;
    int8_t* shmem_B = shmem + SHMEM_A_SIZE; // offset by 4096

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_row = warp_id / WARP_ROWS;
    const int warp_col = warp_id % WARP_COLS;

    // Initialize WMMA accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0);
        }
    }

    // Main sequential loading/computing K-stepping loop
    for (int k_offset = 0; k_offset < static_cast<int>(K); k_offset += TILE_K) {
        // Load block elements into shared memory
        load_tile(A, B, shmem_A, shmem_B, k_offset, tid, M, N, K);
        __syncthreads();

        // Tensor Core Multiplication loop runs sequentially (unrolling omitted to reduce register pressure)
        for (int k_inst = 0; k_inst < 4; ++k_inst) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> frag_A[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> frag_B[2];

            wmma::load_matrix_sync(frag_A[0], &shmem_A[(warp_row * 32) * SHMEM_A_STRIDE + k_inst * WMMA_K], SHMEM_A_STRIDE);
            wmma::load_matrix_sync(frag_A[1], &shmem_A[(warp_row * 32 + 16) * SHMEM_A_STRIDE + k_inst * WMMA_K], SHMEM_A_STRIDE);

            wmma::load_matrix_sync(frag_B[0], &shmem_B[(k_inst * WMMA_K) * SHMEM_B_STRIDE + warp_col * 32], SHMEM_B_STRIDE);
            wmma::load_matrix_sync(frag_B[1], &shmem_B[(k_inst * WMMA_K) * SHMEM_B_STRIDE + warp_col * 32 + 16], SHMEM_B_STRIDE);

            wmma::mma_sync(acc[0][0], frag_A[0], frag_B[0], acc[0][0]);
            wmma::mma_sync(acc[0][1], frag_A[0], frag_B[1], acc[0][1]);
            wmma::mma_sync(acc[1][0], frag_A[1], frag_B[0], acc[1][0]);
            wmma::mma_sync(acc[1][1], frag_A[1], frag_B[1], acc[1][1]);
        }

        __syncthreads();
    }

    // High-performance writeback phase utilising warp-level synchronization
    // Overwrite the shmem_A buffer to partition warp-private staging segments (1024 bytes per warp)
    int32_t* warp_shmem_C = reinterpret_cast<int32_t*>(shmem) + warp_id * FRAG_ELEMENTS;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            // Stage wmma fragment values to warp-private shared memory
            wmma::store_matrix_sync(warp_shmem_C, acc[i][j], 16, wmma::mem_row_major);
            
            // Replaced __syncthreads() with __syncwarp() to avoid cross-warp stalls
            __syncwarp();

            // Perform coalesced read-scale-writeback sequence
            const int lane_id = tid % WARP_SIZE;
            #pragma unroll
            for (int elem = 0; elem < 8; ++elem) {
                const int local_idx = elem * 32 + lane_id;
                const int r_frag = local_idx / 16;
                const int c_frag = local_idx % 16;

                const int32_t val = warp_shmem_C[local_idx];
                const float scaled = __int2float_rn(val) * scale;
                const int rounded = __float2int_rn(scaled);
                const int8_t out_val = static_cast<int8_t>(max(-128, min(127, rounded)));

                const size_t global_row = static_cast<size_t>(blockIdx.y) * TILE_M + warp_row * 32 + i * 16 + r_frag;
                const size_t global_col = static_cast<size_t>(blockIdx.x) * TILE_N + warp_col * 32 + j * 16 + c_frag;

                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = out_val;
                }
            }

            __syncwarp();
        }
    }
}

// Host Entry Point API
extern "C" void solution(
    const int8_t* A,
    const int8_t* B,
    const float scale,
    int8_t* C,
    const size_t M,
    const size_t N,
    const size_t K
) {
    const dim3 block_dim(BLOCK_SIZE, 1, 1);
    const dim3 grid_dim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);

    // Call dynamic launch parameter. 8 KB easily fits within default limit configurations.
    gemm_tensor_core_int8_fused<<<grid_dim, block_dim>>>(A, B, scale, C, M, N, K);
}

// Host-Side CPU Verification Reference
static void cpu_quantized_gemm_reference(
    const int8_t* A,
    const int8_t* B,
    const float scale,
    int8_t* C,
    const size_t M,
    const size_t N,
    const size_t K
) {
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            int32_t acc = 0;
            for (size_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[r * K + k]) * static_cast<int32_t>(B[k * N + c]);
            }
            float scaled = static_cast<float>(acc) * scale;
            int rounded = static_cast<int>(std::round(scaled));
            C[r * N + c] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
        }
    }
}

// Executes performance validation and numerical accuracy checks.
extern "C" void run_benchmark(
    const size_t M,
    const size_t N,
    const size_t K,
    const float scale
) {
    std::cout << "[SYSTEM INFO] Initializing Corrected INT8 GEMM Optimization Benchmark (M=" << M << ", N=" << N << ", K=" << K << ")\n";

    const size_t bytes_A = M * K * sizeof(int8_t);
    const size_t bytes_B = K * N * sizeof(int8_t);
    const size_t bytes_C = M * N * sizeof(int8_t);

    // Allocating Page-Locked Host Memory with validation
    int8_t *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    CUDA_CHECK(cudaHostAlloc(&h_A, bytes_A, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, bytes_B, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C_gpu, bytes_C, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C_cpu, bytes_C, cudaHostAllocDefault));

    // Initialize matrices with pseudo-random inputs
    std::mt19937 prng(1337);
    std::uniform_int_distribution<int> dist(-12, 12);
    for (size_t i = 0; i < M * K; ++i) h_A[i] = static_cast<int8_t>(dist(prng));
    for (size_t i = 0; i < K * N; ++i) h_B[i] = static_cast<int8_t>(dist(prng));

    // Allocate Device Memory with validation
    int8_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // Fast H2D copy
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Warmup Iteration with launch and execution validation
    solution(d_A, d_B, scale, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());       // Catch synchronous launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel execution
    CUDA_CHECK(cudaGetLastError());       // Catch asynchronous execution errors

    // Accuracy Verification
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost));
    std::cout << "[VERIFICATION] Computing sequential CPU reference path...\n";
    cpu_quantized_gemm_reference(h_A, h_B, scale, h_C_cpu, M, N, K);

    size_t absolute_mismatches = 0;
    for (size_t i = 0; i < M * N; ++i) {
        if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > 1) { 
            absolute_mismatches++;
        }
    }

    if (absolute_mismatches > 0) {
        std::cerr << "[ERROR] Correctness check failed. Total mismatched elements: " << absolute_mismatches << " / " << M * N << "\n";
    } else {
        std::cout << "[SUCCESS] Numerical validation passed within permissible rounding bounds.\n";
    }

    // Profiling Loop
    const int benchmark_iterations = 100;
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt, 0));
    for (int iter = 0; iter < benchmark_iterations; ++iter) {
        solution(d_A, d_B, scale, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop_evt, 0));
    CUDA_CHECK(cudaDeviceSynchronize()); // Sync device to capture launch errors before completing profiling
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    const float average_ms = elapsed_ms / benchmark_iterations;

    // Sustained performance calculation
    const double total_operations = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double operations_per_sec = (total_operations * 1e-9) / (average_ms * 1e-3); 

    std::cout << "[PERFORMANCE] Average Kernel Execution Time: " << average_ms << " ms\n";
    std::cout << "[PERFORMANCE] Sustained INT8 Throughput: " << operations_per_sec << " GOPS/TOPS\n";

    // Host/Device Resource clean up
    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C_gpu));
    CUDA_CHECK(cudaFreeHost(h_C_cpu));
}

// Standalone benchmark entry point
int main() {
    const size_t M = 4096;
    const size_t N = 4096;
    const size_t K = 4096;
    const float scale = 0.123f;
    run_benchmark(M, N, K, scale);
    return 0;
}

// ============================================================================
// REMAINING UNKNOWNS:
// 1. Host side controller launch overhead context in complex runtime environments.
// 2. Real performance throttling based on systematic GPU thermal variations and power cap controls.
// 3. Dynamic alignment properties of input arrays under standard library memory allocations.
// ============================================================================
