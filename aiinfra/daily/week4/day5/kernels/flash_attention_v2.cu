// flash_attention_v2.cu —— 完整 FlashAttention Forward Kernel（batch + multi-head）
// 编译命令: nvcc -o flash_attention_v2 flash_attention_v2.cu -O3 -arch=sm_120
// 运行命令: ./flash_attention_v2

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int D = 64;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
static_assert(Br % WARPS_PER_BLOCK == 0, "Br must be divisible by WARPS_PER_BLOCK");
constexpr int ROWS_PER_WARP = Br / WARPS_PER_BLOCK;

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void flashAttentionForward(const float* __restrict__ Q, const float* __restrict__ K,
                                      const float* __restrict__ V, float* __restrict__ O, int B, int H, int N, int d) {

    __shared__ float s_Q[Br][D];
    __shared__ float s_K[Bc][D];
    __shared__ float s_V[Bc][D];

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileRow = blockIdx.x * Br;

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;
    int qRowStart = warpId * ROWS_PER_WARP;

    int bhOffset = ((batch * H + head) * N) * d;

// 协作加载 Q tile
    #pragma unroll
    for (int idx = tid; idx < Br * d; idx += THREADS_PER_BLOCK) {
        int r = idx / d;
        int c = idx % d;
        int globalRow = qTileRow + r;
        s_Q[r][c] = (globalRow < N) ? Q[bhOffset + globalRow * d + c] : 0.0f;
    }
    __syncthreads();

    // 每个 warp 维护 ROWS_PER_WARP 个 Q 行的 running 状态
    float m_arr[ROWS_PER_WARP];
    float l_arr[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP][D];

    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        m_arr[i] = -1e30f;
        l_arr[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++) {
            acc[i][j] = 0.0f;
        }
    }

    float scale = 1.0f / sqrtf((float)d);

    // 内层循环：遍历 KV tile
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
// 协作加载 K/V tile
        #pragma unroll
        for (int idx = tid; idx < Bc * d; idx += THREADS_PER_BLOCK) {
            int r = idx / d;
            int c = idx % d;
            int globalRow = kvStart + r;
            s_K[r][c] = (globalRow < N) ? K[bhOffset + globalRow * d + c] : 0.0f;
            s_V[r][c] = (globalRow < N) ? V[bhOffset + globalRow * d + c] : 0.0f;
        }
        __syncthreads();

// 每个 warp 处理 ROWS_PER_WARP 个 Q 行
        #pragma unroll
        for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
            int qi = qRowStart + localRow;
            int globalQi = qTileRow + qi;
            if (qi >= Br || globalQi >= N)
                continue;

            // Step 1: Sij[c] = Qi · Kj[c]^T (每线程算 Bc/32 个)
            float Sij[Bc / 32];
            #pragma unroll
            for (int c = lane; c < Bc; c += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int di = 0; di < d; di++) {
                    dot += s_Q[qi][di] * s_K[c][di];
                }
                Sij[c / 32] = dot * scale;
            }

            // Step 2: 局部 max (warp reduce)
            float localMax = -1e30f;
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++) {
                localMax = fmaxf(localMax, Sij[i]);
            }
            localMax = warpReduceMax(localMax);

            // Step 3: online softmax update
            float m_prev = m_arr[localRow];
            float m_new = fmaxf(m_prev, localMax);
            float scale_old = expf(m_prev - m_new);

            m_arr[localRow] = m_new;
            l_arr[localRow] *= scale_old;
            #pragma unroll
            for (int di = 0; di < d; di++) {
                acc[localRow][di] *= scale_old;
            }

// Step 4: 处理新块
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++) {
                int c = lane + i * 32;
                bool valid = c < Bc && (kvStart + c) < N;
                float s_val = valid ? Sij[i] : -1e30f;
                float p_val = valid ? expf(s_val - m_new) : 0.0f;

                float p_sum = warpReduceSum(p_val);
                if (lane == 0) {
                    l_arr[localRow] += p_sum;
                }

                #pragma unroll
                for (int di = 0; di < d; di++) {
                    float contrib = valid ? p_val * s_V[c][di] : 0.0f;
                    float sum_contrib = warpReduceSum(contrib);
                    if (lane == 0) {
                        acc[localRow][di] += sum_contrib;
                    }
                }
            }

            // 广播 l 和 acc 到 warp 内所有线程
            l_arr[localRow] = __shfl_sync(0xFFFFFFFF, l_arr[localRow], 0);
            #pragma unroll
            for (int di = 0; di < d; di++) {
                acc[localRow][di] = __shfl_sync(0xFFFFFFFF, acc[localRow][di], 0);
            }
        }

        __syncthreads();
    }

// 写回 O
    #pragma unroll
    for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
        int qi = qRowStart + localRow;
        int globalRow = qTileRow + qi;
        if (qi >= Br || globalRow >= N)
            continue;

        float inv_l = 1.0f / l_arr[localRow];
        #pragma unroll
        for (int di = lane; di < d; di += 32) {
            O[bhOffset + globalRow * d + di] = acc[localRow][di] * inv_l;
        }
    }
}

void cpuAttention(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    float* S = (float*)malloc(N * N * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }

        float mx = S[i * N];
        for (int j = 1; j < N; j++)
            mx = fmaxf(mx, S[i * N + j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - mx);
            sm += S[i * N + j];
        }
        for (int j = 0; j < N; j++)
            S[i * N + j] /= sm;

        for (int k = 0; k < d; k++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += S[i * N + j] * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }
    free(S);
}

void initData(float* data, int n) {
    srand(42);
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    float maxDiff = 0.0f;
    for (int i = 0; i < n; i++) {
        maxDiff = fmaxf(maxDiff, fabsf(a[i] - b[i]));
    }
    bool ok = maxDiff < eps;
    printf(" maxDiff = %.2e (%s)\n", maxDiff, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int B = 2, H = 4, d = D;

    printf("=== FlashAttention v2 Forward Kernel ===\n");
    printf("Config: B=%d, H=%d, N=%d, d=%d\n", B, H, N, d);
    printf("Tile: Br=%d, Bc=%d, Threads=%d\n\n", Br, Bc, THREADS_PER_BLOCK);

    size_t totalElems = (size_t)B * H * N * d;
    size_t bytes = totalElems * sizeof(float);

    float* h_Q = (float*)malloc(bytes);
    float* h_K = (float*)malloc(bytes);
    float* h_V = (float*)malloc(bytes);
    float* h_O = (float*)malloc(bytes);
    float* h_O_CPU = (float*)malloc(bytes);

    initData(h_Q, totalElems);
    initData(h_K, totalElems);
    initData(h_V, totalElems);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    dim3 grid((N + Br - 1) / Br, H, B);
    dim3 block(THREADS_PER_BLOCK);

    // warmup
    flashAttentionForward<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, d);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    flashAttentionForward<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost);

    // CPU 验证（只验证第一个 head）
    cpuAttention(h_Q, h_K, h_V, h_O_CPU, N, d);
    printf("[B=0, H=0] First head check:\n");
    checkResult(h_O, h_O_CPU, N * d, 1e-3f);
    printf("GPU Time: %.3f ms\n", ms);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O_CPU);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// ---------- PyTorch C++ Extension Wrapper ----------
#ifdef WITH_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V) {
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto O = at::empty_like(Q);
    dim3 grid(N / 64, B * H);
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    flashAttentionForward<<<grid, block, 0, stream>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), B, H, N, D);
    return O;
}
#endif
