/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <doca_error.h>
#include <doca_gpunetio_dev_verbs_onesided.cuh>

#include "verbs_common.h"

#define KERNEL_DEBUG_TIMES 1
#define ENABLE_DEBUG 0

template <enum doca_gpu_dev_verbs_exec_scope scope>
__global__ void put_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t data_size,
                       uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf,
                       uint32_t dst_buf_mkey, uint32_t *timer) {
    doca_gpu_dev_verbs_ticket_t out_ticket;
    uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

#if KERNEL_DEBUG_TIMES == 1
    unsigned long long step1 = 0, step2 = 0, step3 = 0;
#endif

    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_iters;
         idx += (blockDim.x * gridDim.x)) {
#if KERNEL_DEBUG_TIMES == 1
        step1 = doca_gpu_dev_verbs_query_globaltimer();
#endif

        doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                               DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, scope>(
            qp,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(dst_buf + (data_size * tidx)),
                                    .key = (uint32_t)dst_buf_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(src_buf + (data_size * tidx)),
                                    .key = (uint32_t)src_buf_mkey},
            data_size, &out_ticket);

#if KERNEL_DEBUG_TIMES == 1
        step2 = doca_gpu_dev_verbs_query_globaltimer();
#endif

        if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD) {
            if (doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    doca_gpu_dev_verbs_qp_get_cq_sq(qp), out_ticket) != 0) {
#if ENABLE_DEBUG == 1
                printf("Error CQE!\n");
#endif
            }
        }

        if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP) {
            if (lane_idx == 0) {
                if (doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                        doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                        out_ticket + blockDim.x - 1) != 0) {
#if ENABLE_DEBUG == 1
                    printf("Error CQE!\n");
#endif
                }
            }
        }
#if KERNEL_DEBUG_TIMES == 1
        step3 = doca_gpu_dev_verbs_query_globaltimer();
        timer[idx] = step2 - step1;
        timer[idx + num_iters] = step3 - step2;
#endif

        __syncthreads();

#if KERNEL_DEBUG_TIMES == 1
        // if (threadIdx.x == 0)
        //     printf("iteration %d src_buf %lx size %d dst_buf %lx put %ld ns, poll %ld ns\n", idx,
        //            src_buf, data_size, dst_buf, step2 - step1, step3 - step2);
#endif
    }
}

extern "C" {

doca_error_t gpunetio_verbs_put_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads,
                                   uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
                                   uint8_t *dst_buf, uint32_t dst_buf_mkey,
                                   enum doca_gpu_dev_verbs_exec_scope scope) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    uint32_t *timer = nullptr;
    cudaMalloc(&timer, sizeof(uint32_t) * num_iters * 2);

    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
        put_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, data_size, src_buf, src_buf_mkey, dst_buf, dst_buf_mkey, timer);
    else if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
        put_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, data_size, src_buf, src_buf_mkey, dst_buf, dst_buf_mkey, timer);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

#if KERNEL_DEBUG_TIMES == 1
    uint32_t *host_timer = (uint32_t *)malloc(sizeof(uint32_t) * num_iters * 2);
    cudaMemcpy(host_timer, timer, sizeof(uint32_t) * num_iters * 2, cudaMemcpyDeviceToHost);
    // printf("Kernel timer results, data_size = %d:\n", data_size);
    uint32_t sum_put = 0;
    uint32_t sum_poll = 0;
    for (uint32_t i = 0; i < num_iters; i++) {
        // printf("Iteration %d: put %u ns, poll %u ns\n", i, host_timer[i], host_timer[i + num_iters]);
        sum_put += host_timer[i];
        sum_poll += host_timer[i + num_iters];
    }
    printf("Average time per iteration for size %d: [put %f us, poll %f us, total %f us]\n", data_size, (float)sum_put / num_iters / 1000, (float)sum_poll / num_iters / 1000, (float)(sum_put + sum_poll) / num_iters / 1000);
    cudaFree(timer);
    free(host_timer);
#endif

    return DOCA_SUCCESS;
}
}
