#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

struct ggml_cuda_top_k_qsa_match {
    const ggml_tensor * scores = nullptr;
    const ggml_tensor * cells = nullptr;
    const ggml_tensor * mask = nullptr;
    ggml_tensor * dst = nullptr;
    int node_count = 0;
};

bool ggml_cuda_match_top_k_qsa(const ggml_cgraph * graph, int index, ggml_cuda_top_k_qsa_match & match);
void ggml_cuda_op_top_k_qsa(ggml_backend_cuda_context & ctx, const ggml_cuda_top_k_qsa_match & match);
