// kv_cache.cu —— 支持多轮对话的 KV Cache CUDA 实现
// 编译命令: nvcc -o kv_cache kv_cache.cu -O3 -arch=sm_120
// 运行命令: ./kv_cache

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// --------------------------------------------------
// KVCache 类
// 存储 layout: (num_layers, batch_size, num_heads, max_seq_len, d_head)
// 为简化，append 用 cudaMemcpy 逐 head 拷贝；生产级会用一个 kernel 完成
// --------------------------------------------------
class KVCache {
  public:
    KVCache(int num_layers, int batch_size, int num_heads, int max_seq_len, int d_head)
        : num_layers_(num_layers), batch_size_(batch_size), num_heads_(num_heads), max_seq_len_(max_seq_len),
          d_head_(d_head) {

        size_per_layer_ = (size_t)batch_size_ * num_heads_ * max_seq_len_ * d_head_ * sizeof(float);
        total_size_ = (size_t)num_layers_ * size_per_layer_;

        cudaMalloc(&k_cache_, total_size_);
        cudaMalloc(&v_cache_, total_size_);
        cudaMemset(k_cache_, 0, total_size_);
        cudaMemset(v_cache_, 0, total_size_);

        // 每个 batch 当前已缓存的序列长度
        seq_lens_ = std::vector<int>(batch_size_, 0);
    }

    ~KVCache() {
        cudaFree(k_cache_);
        cudaFree(v_cache_);
    }

    // 追加新的 K/V 到 cache
    // k_new/v_new shape: (batch_size, num_heads, new_len, d_head)，在 device 上
    void append(int layer_id, const float* k_new, const float* v_new, int new_len) {
        for (int b = 0; b < batch_size_; b++) {
            int start = seq_lens_[b];
            int end = start + new_len;
            if (end > max_seq_len_) {
                printf("Error: seq len %d exceeds max_seq_len %d\n", end, max_seq_len_);
                return;
            }

            // 拷贝 k_new[b, :, :, :] 到 k_cache_[layer_id, b, :, start:end, :]
            for (int h = 0; h < num_heads_; h++) {
                size_t src_offset = ((size_t)b * num_heads_ * new_len * d_head_ + h * new_len * d_head_);
                size_t dst_offset =
                    ((size_t)layer_id * batch_size_ * num_heads_ * max_seq_len_ * d_head_ +
                     b * num_heads_ * max_seq_len_ * d_head_ + h * max_seq_len_ * d_head_ + start * d_head_);
                size_t bytes = (size_t)new_len * d_head_ * sizeof(float);
                cudaMemcpy(k_cache_ + dst_offset, k_new + src_offset, bytes, cudaMemcpyDeviceToDevice);
                cudaMemcpy(v_cache_ + dst_offset, v_new + src_offset, bytes, cudaMemcpyDeviceToDevice);
            }
            seq_lens_[b] = end;
        }
    }

    // 获取某层 cache 指针和各 batch 序列长度
    void get_cache(int layer_id, float** k_ptr, float** v_ptr, std::vector<int>* seq_lens) {
        *k_ptr = k_cache_ + (size_t)layer_id * size_per_layer_ / sizeof(float);
        *v_ptr = v_cache_ + (size_t)layer_id * size_per_layer_ / sizeof(float);
        *seq_lens = seq_lens_;
    }

    int get_seq_len(int batch_id) const {
        return seq_lens_[batch_id];
    }

    void reset() {
        cudaMemset(k_cache_, 0, total_size_);
        cudaMemset(v_cache_, 0, total_size_);
        std::fill(seq_lens_.begin(), seq_lens_.end(), 0);
    }

    void reset_batch(int batch_id) {
        size_t batch_bytes = (size_t)num_heads_ * max_seq_len_ * d_head_ * sizeof(float);
        for (int l = 0; l < num_layers_; l++) {
            size_t offset = ((size_t)l * batch_size_ * num_heads_ * max_seq_len_ * d_head_ +
                             batch_id * num_heads_ * max_seq_len_ * d_head_);
            cudaMemset(k_cache_ + offset, 0, batch_bytes);
            cudaMemset(v_cache_ + offset, 0, batch_bytes);
        }
        seq_lens_[batch_id] = 0;
    }

  private:
    int num_layers_, batch_size_, num_heads_, max_seq_len_, d_head_;
    size_t size_per_layer_, total_size_;
    float* k_cache_;
    float* v_cache_;
    std::vector<int> seq_lens_;
};

// --------------------------------------------------
// 验证：把 cache 内容读回 host，检查追加的数据是否正确落位
// --------------------------------------------------
void initData(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }
}

int main() {
    const int num_layers = 2;
    const int batch_size = 1;
    const int num_heads = 8;
    const int max_seq_len = 1024;
    const int d_head = 64;

    printf("=== KV Cache Test ===\n");
    printf("Config: layers=%d, batch=%d, heads=%d, max_len=%d, d_head=%d\n", num_layers, batch_size, num_heads,
           max_seq_len, d_head);

    KVCache cache(num_layers, batch_size, num_heads, max_seq_len, d_head);

    // Round 1: prompt 长度 10
    int round1_len = 10;
    size_t round1_bytes = (size_t)batch_size * num_heads * round1_len * d_head * sizeof(float);
    float *d_k1, *d_v1;
    cudaMalloc(&d_k1, round1_bytes);
    cudaMalloc(&d_v1, round1_bytes);
    float* h_k1 = (float*)malloc(round1_bytes);
    float* h_v1 = (float*)malloc(round1_bytes);
    initData(h_k1, batch_size * num_heads * round1_len * d_head);
    initData(h_v1, batch_size * num_heads * round1_len * d_head);
    cudaMemcpy(d_k1, h_k1, round1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, h_v1, round1_bytes, cudaMemcpyHostToDevice);

    cache.append(0, d_k1, d_v1, round1_len);
    printf("After Round 1 (len=%d): seq_len=%d\n", round1_len, cache.get_seq_len(0));

    // Round 2: 新增 5 个 tokens
    int round2_len = 5;
    size_t round2_bytes = (size_t)batch_size * num_heads * round2_len * d_head * sizeof(float);
    float *d_k2, *d_v2;
    cudaMalloc(&d_k2, round2_bytes);
    cudaMalloc(&d_v2, round2_bytes);
    float* h_k2 = (float*)malloc(round2_bytes);
    float* h_v2 = (float*)malloc(round2_bytes);
    initData(h_k2, batch_size * num_heads * round2_len * d_head);
    initData(h_v2, batch_size * num_heads * round2_len * d_head);
    cudaMemcpy(d_k2, h_k2, round2_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, h_v2, round2_bytes, cudaMemcpyHostToDevice);

    cache.append(0, d_k2, d_v2, round2_len);
    printf("After Round 2 (len=%d): seq_len=%d\n", round2_len, cache.get_seq_len(0));

    // Round 3: 新增 8 个 tokens
    cache.append(0, d_k2, d_v2, 8);
    printf("After Round 3 (len=8): seq_len=%d\n", cache.get_seq_len(0));

    // 验证总长度
    int expected = round1_len + round2_len + 8;
    if (cache.get_seq_len(0) == expected) {
        printf("PASS: seq_len = %d (expected %d)\n", cache.get_seq_len(0), expected);
    } else {
        printf("FAIL: seq_len = %d (expected %d)\n", cache.get_seq_len(0), expected);
    }

    // 验证数据正确性：读回 layer 0 的 K cache，检查 round1 数据落在 [0:10]
    float *k_ptr, *v_ptr;
    std::vector<int> seq_lens;
    cache.get_cache(0, &k_ptr, &v_ptr, &seq_lens);

    size_t check_bytes = (size_t)num_heads * round1_len * d_head * sizeof(float);
    float* h_check = (float*)malloc(check_bytes);
    // layer 0, batch 0, head 0 起始位置
    cudaMemcpy(h_check, k_ptr, check_bytes, cudaMemcpyDeviceToHost);

    float max_diff = 0.f;
    for (int i = 0; i < num_heads * round1_len * d_head; i++) {
        max_diff = fmaxf(max_diff, fabsf(h_check[i] - h_k1[i]));
    }
    printf("Data verification (Round 1 K in cache): max_diff = %.2e (%s)\n", max_diff,
           max_diff < 1e-5f ? "PASS" : "FAIL");

    // 内存占用统计
    size_t bytes_per_token = (size_t)num_layers * num_heads * d_head * sizeof(float) * 2;
    printf("KV Cache bytes per token: %zu\n", bytes_per_token);
    printf("Max memory usage: %zu MB\n", bytes_per_token * max_seq_len / (1024 * 1024));

    // 类比真实模型：LLaMA-7B (32 层, 32 头, d_head=128, fp16)
    size_t llama_per_token = 2 * 32 * 32 * 128 * 2; // fp16 = 2 bytes
    printf("\n[LLaMA-7B reference] bytes per token: %zu (%.1f KB)\n", llama_per_token, llama_per_token / 1024.0);
    printf("[LLaMA-7B reference] 4096 tokens: %zu MB\n", llama_per_token * 4096 / (1024 * 1024));
    printf("[LLaMA-7B reference] batch=16, 4096 tokens: %zu GB\n", llama_per_token * 4096 * 16 / (1024 * 1024 * 1024));

    free(h_k1);
    free(h_v1);
    free(h_k2);
    free(h_v2);
    free(h_check);
    cudaFree(d_k1);
    cudaFree(d_v1);
    cudaFree(d_k2);
    cudaFree(d_v2);

    return 0;
}
