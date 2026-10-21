// PQ2_0 mat-vec for the P100 (GP100, sm_60), 1-4 columns, plain 2D. Bit-identical to
// mul_mat_vec_pq2_0_ordered (ggml-cuda.cu), which it replaces there when the runtime device is
// cc == 600; GGML_CUDA_GP100_MMVQ_DISABLE=1 falls back to the dp4a kernel at runtime.
//
// Why: sm_60 has no __dp4a, so the dp4a kernel's byte-multiplies run as four XMAD each, while GP100
// has full-rate HFMA2 even on subnormal operands. Here the 2-bit weight codes unpack straight into
// subnormal half2 operands -- (word >> 2p) & 0x00030003 is the half2 (c_p, c_{p+8}) * 2^-24, since
// two 2-bit codes land in mantissa bits [1:0] and [17:16] -- and the activations arrive pre-scaled
// as half2 (q_p, q_{p+8}) * 2^8 (GGML_CUDA_Q8_1_PQ2_F16, quantize.cu), so one HFMA2 accumulates
// c*q*2^-16 exactly.
//
// Bit-identity to the dp4a kernel: the two fp16 accumulators per (row, block) each hold 8 pairs,
// |sum(c*q)| <= 8*2*128 = 2048 = fp16's exact-integer limit, so sum(c*q) is the same integer the
// dp4a kernel computes; qs is subtracted as the same raw int16 block sum, and the epilog keeps the
// same d2 * d8 association order (scaling d8 by the exact power of two 2^16 preserves rounding).
// Same sums, same float ops, same bits.
//
// Geometry follows the dp4a kernel: one warp keeps that kernel's chunk order for R rows at once
// (lane owns chunks lane + 32*w + 128*i), no shared memory, no block barrier, clamped rows are
// computed but not stored. R is 4 like the dp4a kernel; GGML_CUDA_GP100_MMVQ_R=<2|4|8> overrides it
// for tuning sweeps. Device code compiles for sm_60 only; other archs see NO_DEVICE_CODE stubs.

#include "common.cuh"
#include "quantize.cuh"

#include <cstdlib>

struct block_q8_1_pq2f16 {
    half2 y[16]; // pair p = elements (p, p+8) of each 16-element half, pre-scaled by 2^8
    half2 ds;    // (d, raw int16 sum of q in the bit pattern)
};

template <int ncols, int R>
static __global__ void __launch_bounds__(128) mul_mat_vec_pq2_0_ordered_gp100(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
#if __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
    const int lane = threadIdx.x & 31;
    const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * R;
    if (row0 >= nrows_x) {
        return;
    }
    const int    nchunk    = ncols_x / QK8_1;
    const size_t col_bytes = (size_t) stride_col_y * sizeof(block_q8_1_pq2f16);
    const size_t ds_off    = (size_t) stride_col_y * 2*QK8_1; // payload is 64 B per block, ds packed after the row
    const char * y         = (const char *) vy;

    float s[R][ncols];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float v[R][ncols] = {};
        for (int c = 32*w + lane; c < nchunk; c += 128) {
            uint32_t w0[R], w1[R];
            float    d2[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, nrows_x - 1); // clamped rows are computed but not stored
                const block_pq2_0 * blk = (const block_pq2_0 *) vx + (int64_t) row*stride_row_x + (c >> 2);
                const uint16_t    * qp  = (const uint16_t *) (blk->qs + (c & 3)*8); // blocks are only 2-byte aligned
                d2[r] = blk->d;
                w0[r] = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
                w1[r] = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
            }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const char * yc = y + j*col_bytes;
                    const int4  a01 = *(const int4 *) (yc + c*64 +  0); // pairs 0-3  (w0 codes, elements 0-15)
                    const int4  a23 = *(const int4 *) (yc + c*64 + 16); // pairs 4-7
                    const int4  a45 = *(const int4 *) (yc + c*64 + 32); // pairs 8-11 (w1 codes, elements 16-31)
                    const int4  a67 = *(const int4 *) (yc + c*64 + 48); // pairs 12-15
                    const half2 ds  = *(const half2 *) (yc + ds_off + c*sizeof(half2));
                    const float d8  = __low2float(ds);
                    const int   qs  = __half_as_short(__high2half(ds));
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        uint32_t m[16];
#pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            m[p]   = (w0[r] >> (2*p)) & 0x00030003u;
                            m[8+p] = (w1[r] >> (2*p)) & 0x00030003u;
                        }
                        const half2 * mw = (const half2 *) m;
                        const half2 * ma = (const half2 *) &a01;
                        half2 acc0 = make_half2(0.0f, 0.0f); // pairs 0-7
                        half2 acc1 = make_half2(0.0f, 0.0f); // pairs 8-15
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            acc0 = __hfma2(mw[p], ma[p], acc0);
                        }
                        ma = (const half2 *) &a23;
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            acc0 = __hfma2(mw[4+p], ma[p], acc0);
                        }
                        ma = (const half2 *) &a45;
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            acc1 = __hfma2(mw[8+p], ma[p], acc1);
                        }
                        ma = (const half2 *) &a67;
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            acc1 = __hfma2(mw[12+p], ma[p], acc1);
                        }
                        // 2^16 undoes the 2^-16 HFMA2 scale (exact) before the raw int sum, so (d2*d8)
                        // rounds the same product as in the dp4a kernel
                        v[r][j] += d2[r] * d8 * (((__low2float(acc0) + __high2float(acc0)) +
                                                  (__low2float(acc1) + __high2float(acc1))) * 65536.0f - (float) qs);
                    }
                }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                s[r][j] = w == 0 ? v[r][j] : s[r][j] + v[r][j];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float sum = warp_reduce_sum<WARP_SIZE>(s[r][j]);
            if (lane == 0 && row0 + r < nrows_x) {
                dst[(int64_t) j*stride_col_dst + row0 + r] = sum;
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_PASCAL
}

static int gp100_mmvq_r() {
    int r = 4;
    if (const char * e = getenv("GGML_CUDA_GP100_MMVQ_R")) {
        r = atoi(e);
    }
    if (r != 2 && r != 8) {
        r = 4; // only the swept values instantiate
    }
    return r;
}

bool ggml_cuda_mul_mat_pq2_0_ordered_gp100(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_GP100_MMVQ_DISABLE") != nullptr &&
        atoi(getenv("GGML_CUDA_GP100_MMVQ_DISABLE")) != 0;
    if (disabled) {
        return false;
    }
    // GP100 only: P40 (sm_61) has no full-rate fp16, anything newer has native dp4a
    if (ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS
    cudaStream_t stream = ctx.stream();
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne11*ne10_padded*sizeof(block_q8_1_pq2f16)/QK8_1);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), GGML_CUDA_Q8_1_PQ2_F16, ne10,
        nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float), ne10_padded, ne11, 1, 1, stream);

    const int sx = nb01/ggml_type_size(src0->type), sy = ne10_padded/QK8_1, sd = nb1/sizeof(float);
    const void * x = src0->data;
    const char * yq = src1_q8_1.get();
    float * d = (float *) dst->data;

    const int r = gp100_mmvq_r();
    const dim3 threads(128);
    const dim3 blocks((ne01 + 4*r - 1) / (4*r));
#define GP100_R_CASES(NCOLS)                                                                                                        \
    switch (r) {                                                                                                                    \
        case 2: mul_mat_vec_pq2_0_ordered_gp100<NCOLS, 2><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;  \
        case 8: mul_mat_vec_pq2_0_ordered_gp100<NCOLS, 8><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;  \
        default: mul_mat_vec_pq2_0_ordered_gp100<NCOLS, 4><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break; \
    }
    switch (ne11) {
        case 1: GP100_R_CASES(1); break;
        case 2: GP100_R_CASES(2); break;
        case 3: GP100_R_CASES(3); break;
        case 4: GP100_R_CASES(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef GP100_R_CASES
    CUDA_CHECK(cudaGetLastError());
    return true;
}
