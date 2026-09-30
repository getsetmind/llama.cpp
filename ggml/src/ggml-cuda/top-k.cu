#include "argsort.cuh"
#include "top-k.cuh"

#include <type_traits>
#include <atomic>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

template<typename Input>
static void top_k_cub(ggml_cuda_pool & pool,
                      Input            src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

bool ggml_cuda_match_top_k_qsa(const ggml_cgraph * graph, int index, ggml_cuda_top_k_qsa_match & match) {
#ifdef CUB_TOP_K_AVAILABLE
    static const bool enabled = getenv("GGML_CUDA_QSA_TOP_K") != nullptr && std::atoi(getenv("GGML_CUDA_QSA_TOP_K")) != 0;
    if (!enabled) {
        return false;
    }
    static const bool diagnose = getenv("GGML_CUDA_QSA_TOP_K_DIAG") != nullptr;
    static std::atomic<int> diagnostics { 0 };
    const bool log_match = diagnose && graph->nodes[index]->op == GGML_OP_GET_ROWS &&
            graph->nodes[index]->src[0]->type == GGML_TYPE_F32 && graph->nodes[index]->src[0]->ne[0] <= 128 &&
            diagnostics.fetch_add(1) < 4;
    auto reject = [&](const char * reason, int offset = -1) {
        if (log_match) {
            GGML_LOG_INFO("qsa_match_diag: reject=%s offset=%d\n", reason, offset);
        }
        return false;
    };
    if (log_match) {
        GGML_LOG_INFO("qsa_match_diag: scores=%lld,%lld,%lld cells=%lld\n",
            (long long) graph->nodes[index]->src[0]->ne[0], (long long) graph->nodes[index]->src[0]->ne[1],
            (long long) graph->nodes[index]->src[0]->ne[2], (long long) graph->nodes[index]->src[1]->ne[0]);
        for (int offset = 0; offset < 8 && index + offset < graph->n_nodes; ++offset) {
            const ggml_tensor * node = graph->nodes[index + offset];
            GGML_LOG_INFO("qsa_match_diag: offset=%d op=%s flags=%d uses=%d name=%s\n", offset,
                ggml_op_name(node->op), node->flags, ggml_node_get_use_count(graph, index + offset), node->name);
            auto log_tensor = [offset](const char * role, const ggml_tensor * tensor) {
                if (tensor == nullptr) {
                    return;
                }
                GGML_LOG_INFO("qsa_match_diag: offset=%d role=%s type=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n",
                    offset, role, ggml_type_name(tensor->type),
                    (long long) tensor->ne[0], (long long) tensor->ne[1],
                    (long long) tensor->ne[2], (long long) tensor->ne[3],
                    tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
            };
            log_tensor("node", node);
            log_tensor("src0", node->src[0]);
            log_tensor("src1", node->src[1]);
        }
    }
    if (index + 4 >= graph->n_nodes) {
        return reject("minimum_node_count");
    }
    const std::initializer_list<ggml_op> cast_ops = {
        GGML_OP_GET_ROWS, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY,
        GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_TOP_K,
    };
    const std::initializer_list<ggml_op> reshape_ops = {
        GGML_OP_GET_ROWS, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_TOP_K,
    };
    const std::initializer_list<ggml_op> ready_ops = {
        GGML_OP_GET_ROWS, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_TOP_K,
    };
    const ggml_op mask_op = graph->nodes[index + 3]->op;
    const auto ops = mask_op == GGML_OP_CPY ? cast_ops : mask_op == GGML_OP_RESHAPE ? reshape_ops : ready_ops;
    const int last = index + (int) ops.size() - 1;
    if (last >= graph->n_nodes) {
        return reject("pattern_node_count");
    }
    for (int offset = 0; offset < (int) ops.size(); ++offset) {
        const ggml_tensor * node = graph->nodes[index + offset];
        if (node->op != ops.begin()[offset] || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return reject("operation_or_compute_flag", offset);
        }
        // castのCPYは自身の書込先参照を含むので、外部への参照と区別する
        if (offset < (int) ops.size() - 1 && ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                ggml_node_get_use_count(graph, index + offset) != (node->op == GGML_OP_CPY ? 2 : 1))) {
            return reject("output_or_use_count", offset);
        }
    }
    if (!ggml_check_edges(graph, index, { { 1, 0, 0 }, { 2, 0, 1 } })) {
        return reject("gather_permute_cont_edges");
    }
    const ggml_tensor * gather = graph->nodes[index];
    const ggml_tensor * scores = gather->src[0];
    const ggml_tensor * cells = gather->src[1];
    const ggml_tensor * expanded = graph->nodes[index + 2];
    const ggml_tensor * add = graph->nodes[last - 1];
    ggml_tensor * dst = graph->nodes[last];
    if (add->src[0] != expanded || dst->src[0] != add) {
        return reject("add_topk_edges");
    }
    const ggml_tensor * mask = add->src[1];
    if (ops.size() != ready_ops.size()) {
        const ggml_tensor * reshaped = graph->nodes[last - 2];
        if (mask != reshaped) {
            return reject("mask_reshape_edge");
        }
        mask = reshaped->src[0];
    }
    // maskのcastがCPUへ割り当てられた場合は、転送済みF32をそのまま読む
    if (ops.size() == cast_ops.size()) {
        const ggml_tensor * cast = graph->nodes[index + 3];
        if (mask != cast || cast->type != GGML_TYPE_F32 || cast->src[0]->type != GGML_TYPE_F16) {
            return reject("mask_cast_edge_or_type");
        }
        mask = cast->src[0];
    } else if (mask->type != GGML_TYPE_F32) {
        return reject("mask_type");
    }
    if (scores->type != GGML_TYPE_F32 || cells->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(scores) || !ggml_is_contiguous(cells) || !ggml_is_contiguous(mask) ||
            !ggml_is_contiguous(expanded) || !ggml_is_contiguous(dst)) {
        return reject("type_or_contiguity");
    }
    const int64_t n_tokens = scores->ne[0];
    const int64_t n_blocks = scores->ne[1];
    const int64_t n_streams = scores->ne[2];
    const int64_t n_cells = cells->ne[0];
    // 複数queryでは単体測定で遅くなったため、確認済みの単一queryだけ融合する
    if (n_tokens != 1) {
        return reject("multiple_queries");
    }
    if (scores->ne[3] != 1 || cells->ne[1] != n_streams || ggml_nrows(cells) != n_streams ||
            n_tokens <= 0 || n_blocks <= 0 || n_cells <= 0 || n_cells > INT_MAX ||
            ggml_nelements(mask) != n_cells*n_tokens*n_streams ||
            expanded->ne[0] != n_cells || expanded->ne[1] != n_tokens || expanded->ne[2] != n_streams || expanded->ne[3] != 1 ||
            !ggml_are_same_shape(expanded, add->src[1]) || !ggml_are_same_shape(expanded, add) ||
            dst->ne[1] != n_tokens || dst->ne[2] != n_streams || dst->ne[3] != 1 || dst->ne[0] <= 0 || dst->ne[0] > n_cells) {
        return reject("shape");
    }
    // 展開後のcell軸とtoken軸が入れ替わっていることをstrideで確認する
    const ggml_tensor * permuted = graph->nodes[index + 1];
    if (permuted->ne[0] != n_cells || permuted->ne[1] != n_tokens || permuted->ne[2] != n_streams ||
            permuted->nb[0] != gather->nb[1] || permuted->nb[1] != gather->nb[0] || permuted->nb[2] != gather->nb[2]) {
        return reject("permute_shape_or_stride");
    }
    match = { scores, cells, mask, dst, (int) ops.size() };
    if (log_match) {
        GGML_LOG_INFO("qsa_match_diag: matched nodes=%d\n", match.node_count);
    }
    return true;
#else
    GGML_UNUSED(graph);
    GGML_UNUSED(index);
    GGML_UNUSED(match);
    return false;
#endif
}

#ifdef CUB_TOP_K_AVAILABLE
template<typename Mask>
struct qsa_score_at_cell {
    const float * scores;
    const int32_t * cells;
    const Mask * mask;
    int64_t token_stride;

    __host__ __device__ float operator()(int cell) const {
        const float score = scores[(int64_t) cells[cell]*token_stride];
        if constexpr (std::is_same<Mask, half>::value) {
            return score + __half2float(mask[cell]);
        } else {
            return score + mask[cell];
        }
    }
};

template<typename Mask>
static void top_k_qsa_cuda(ggml_backend_cuda_context & ctx, const ggml_cuda_top_k_qsa_match & match) {
    const int64_t n_tokens = match.scores->ne[0];
    const int64_t n_blocks = match.scores->ne[1];
    const int64_t n_cells = match.cells->ne[0];
    const int64_t width = match.dst->ne[0];
    for (int64_t stream = 0; stream < match.scores->ne[2]; ++stream) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            const int64_t row = stream*n_tokens + token;
            qsa_score_at_cell<Mask> transform {
                (const float *) match.scores->data + stream*n_blocks*n_tokens + token,
                (const int32_t *) match.cells->data + stream*n_cells,
                (const Mask *) match.mask->data + row*n_cells,
                n_tokens,
            };
            auto input = cuda::make_transform_iterator(cuda::make_counting_iterator(0), transform);
            top_k_cub(ctx.pool(), input, (int *) match.dst->data + row*width, (int) n_cells, (int) width, ctx.stream());
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

void ggml_cuda_op_top_k_qsa(ggml_backend_cuda_context & ctx, const ggml_cuda_top_k_qsa_match & match) {
#ifdef CUB_TOP_K_AVAILABLE
    if (match.mask->type == GGML_TYPE_F16) {
        top_k_qsa_cuda<half>(ctx, match);
    } else {
        top_k_qsa_cuda<float>(ctx, match);
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(match);
    GGML_ABORT("QSA top-k requires CUB DeviceTopK");
#endif
}
