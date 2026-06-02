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

#include "task_queue.cuh"

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

struct GlobalContext {
    struct doca_gpu_dev_verbs_qp *qp;
    uint32_t num_iters;
    uint32_t data_size;
    uint8_t *src_buf;
    uint32_t src_buf_mkey;
    uint8_t *dst_buf;
    uint32_t dst_buf_mkey;
    uint32_t* timer;

    uint32_t num_coroutines;
};

struct ThreadContext {
    uint32_t idx[1024];
    doca_gpu_dev_verbs_ticket_t out_ticket[1024];
};

struct TaskContext {
    // task state
    int state = 0;

    // task context parameters
    ThreadContext thread_ctx;
};

__global__
void initPersistentKernel(CUDAQueue<int>** d_task_queues, int num_coroutines, GlobalContext* gc, TaskContext* tc,
                          struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t page_size,
                          uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
                          uint32_t* timer) {
    // global context
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        gc->qp = qp;
        gc->num_iters = num_iters;
        gc->data_size = page_size;
        gc->src_buf = src_buf;
        gc->src_buf_mkey = src_buf_mkey;
        gc->dst_buf = dst_buf;
        gc->dst_buf_mkey = dst_buf_mkey;
        gc->timer = timer;

        gc->num_coroutines = num_coroutines;
    }

    // task context
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // for each block, only thread-0 does initialization
        // get block id and its corresponding task queue
        int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
        CUDAQueue<int>* io_queue = d_task_queues[block_id];
        // for each task in this block, push task id into the queue
        for (int t = 0; t < gc->num_coroutines; t++) {
            int task_id = block_id * gc->num_coroutines + t;
            bool push_ok = queuePush<int>(io_queue, task_id);
            tc[task_id] = TaskContext();
            // printf("Initialized task %d, push_ok=%d\n", task_id, push_ok);
        }
    }
}

__device__ void do_kernel_thread_segment_0(int task_id, uint32_t& idx) {
    idx = task_id * blockDim.x + threadIdx.x;
}

__device__ void do_kernel_thread_save_context_segment_0(uint32_t idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.idx[tid] = idx;
}

__device__ void do_kernel_thread_restore_context_segment_1(uint32_t& idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    idx = task_ctx.thread_ctx.idx[tid];
}

__device__ void do_kernel_thread_segment_1(struct doca_gpu_dev_verbs_qp *qp, uint32_t data_size,
                       uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
                       doca_gpu_dev_verbs_ticket_t& out_ticket, int task_id) {
    uint32_t tidx = threadIdx.x + (task_id * blockDim.x);
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>(
    qp,
    doca_gpu_dev_verbs_addr{.addr = (uint64_t)(dst_buf + (data_size * tidx)),
                            .key = (uint32_t)dst_buf_mkey},
    doca_gpu_dev_verbs_addr{.addr = (uint64_t)(src_buf + (data_size * tidx)),
                            .key = (uint32_t)src_buf_mkey},
    data_size, &out_ticket);
}

__device__ void do_kernel_thread_save_context_segment_1(doca_gpu_dev_verbs_ticket_t out_ticket, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.out_ticket[tid] = out_ticket;
}

__device__ void do_kernel_thread_restore_context_segment_2(doca_gpu_dev_verbs_ticket_t& out_ticket, uint32_t& idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    out_ticket = task_ctx.thread_ctx.out_ticket[tid];
    idx = task_ctx.thread_ctx.idx[tid];
}

__device__ void do_kernel_thread_segment_2(struct doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_ticket_t& out_ticket, uint32_t& idx, uint32_t num_coroutines) {
    if (doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            doca_gpu_dev_verbs_qp_get_cq_sq(qp), out_ticket) != 0) {
        printf("Error CQE for ticket %lu!\n", out_ticket);
    }
    idx += blockDim.x * gridDim.x * num_coroutines;
}

__device__ void do_kernel_thread_save_context_segment_2(uint32_t idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.idx[tid] = idx;
}


__device__ void do_kernel_thread(GlobalContext* gc, TaskContext* tc, int task_id) {
    TaskContext& task_ctx = tc[task_id];

    uint32_t idx;
    doca_gpu_dev_verbs_ticket_t out_ticket;

    switch (task_ctx.state) {
        case 0: {
            do_kernel_thread_segment_0(task_id, idx);
            do_kernel_thread_save_context_segment_0(idx, task_ctx);
            
            if (threadIdx.x == 0) task_ctx.state = 1;
            // break;
        }
        case 1: {
        case1:
            do_kernel_thread_restore_context_segment_1(idx, task_ctx);
            do_kernel_thread_segment_1(gc->qp, gc->data_size, gc->src_buf, gc->src_buf_mkey, gc->dst_buf, gc->dst_buf_mkey, out_ticket, task_id);
            do_kernel_thread_save_context_segment_1(out_ticket, task_ctx);
            
            if (threadIdx.x == 0) task_ctx.state = 2;
            break;
        }
        case 2: {
            do_kernel_thread_restore_context_segment_2(out_ticket, idx, task_ctx);
            do_kernel_thread_segment_2(gc->qp, out_ticket, idx, gc->num_coroutines);
            do_kernel_thread_save_context_segment_2(idx, task_ctx);

            if (threadIdx.x == 0) {
                task_ctx.state = (idx < gc->num_iters) ? 1 : 3;
            }
            __syncthreads();
            if (task_ctx.state == 1) {
                goto case1;   // loop back to segment 1 for the next request
            }
            break;
        }
    }

}


__global__
void launchPersistentKernelThread(CUDAQueue<int>** d_task_queues, GlobalContext* gc, TaskContext* tc)
{   
    int block_id = blockIdx.x;
    CUDAQueue<int>* task_queue = d_task_queues[block_id];   // find the task queue according to the block id
    
    while (true) {
        __shared__ int task_id;
        __shared__ bool pop_ok;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            pop_ok = queuePop(task_queue, task_id);
            // printf("Popped task_id %d by thread block: %d\n", task_id, blockIdx.x);
        }
        __syncthreads();
        if (!pop_ok) {
            // printf("warning: no task to pop for block %d\n", 
            //        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
            break;
        }

        do_kernel_thread(gc, tc, task_id);
        __syncthreads();

        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            if (tc[task_id].state >= 3) { // task done
                // printf("Task %d completed all states, state = %d.\n", task_id, tc[task_id].state);
            } else {
                queuePush(d_task_queues[blockIdx.x], task_id);
            }
        }
        __syncthreads();
    } // end while

    return ;
}


__device__ void do_kernel_warp_segment_0(int task_id, uint32_t& idx) {
    idx = task_id * blockDim.x + threadIdx.x;
}

__device__ void do_kernel_warp_save_context_segment_0(uint32_t idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.idx[tid] = idx;
}

__device__ void do_kernel_warp_restore_context_segment_1(uint32_t& idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    idx = task_ctx.thread_ctx.idx[tid];
}

__device__ void do_kernel_warp_segment_1(struct doca_gpu_dev_verbs_qp *qp, uint32_t data_size,
                       uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
                       doca_gpu_dev_verbs_ticket_t& out_ticket, int task_id) {
    uint32_t tidx = threadIdx.x + (task_id * blockDim.x);
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP>(
    qp,
    doca_gpu_dev_verbs_addr{.addr = (uint64_t)(dst_buf + (data_size * tidx)),
                            .key = (uint32_t)dst_buf_mkey},
    doca_gpu_dev_verbs_addr{.addr = (uint64_t)(src_buf + (data_size * tidx)),
                            .key = (uint32_t)src_buf_mkey},
    data_size, &out_ticket);
}

__device__ void do_kernel_warp_save_context_segment_1(doca_gpu_dev_verbs_ticket_t out_ticket, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.out_ticket[tid] = out_ticket;
}

__device__ void do_kernel_warp_restore_context_segment_2(doca_gpu_dev_verbs_ticket_t& out_ticket, uint32_t& idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    out_ticket = task_ctx.thread_ctx.out_ticket[tid];
    idx = task_ctx.thread_ctx.idx[tid];
}

__device__ void do_kernel_warp_segment_2(struct doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_ticket_t& out_ticket, uint32_t& idx, uint32_t num_coroutines) {
    uint32_t lane_idx = doca_gpu_dev_verbs_get_lane_id();
    if (lane_idx == 0) {
        if (doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                doca_gpu_dev_verbs_qp_get_cq_sq(qp), out_ticket + blockDim.x - 1) != 0) {
            printf("Error CQE for ticket %lu!\n", out_ticket);
        }
    }
    idx += blockDim.x * gridDim.x * num_coroutines;
}

__device__ void do_kernel_warp_save_context_segment_2(uint32_t idx, TaskContext& task_ctx) {
    auto tid = threadIdx.x;
    task_ctx.thread_ctx.idx[tid] = idx;
}


__device__ void do_kernel_warp(GlobalContext* gc, TaskContext* tc, int task_id) {
    TaskContext& task_ctx = tc[task_id];

    uint32_t idx;
    doca_gpu_dev_verbs_ticket_t out_ticket;

    switch (task_ctx.state) {
        case 0: {
            do_kernel_warp_segment_0(task_id, idx);
            do_kernel_warp_save_context_segment_0(idx, task_ctx);
            
            if (threadIdx.x == 0) task_ctx.state = 1;
            // break;
        }
        case 1: {
        case1:
            do_kernel_warp_restore_context_segment_1(idx, task_ctx);
            do_kernel_warp_segment_1(gc->qp, gc->data_size, gc->src_buf, gc->src_buf_mkey, gc->dst_buf, gc->dst_buf_mkey, out_ticket, task_id);
            do_kernel_warp_save_context_segment_1(out_ticket, task_ctx);
            
            if (threadIdx.x == 0) task_ctx.state = 2;
            break;
        }
        case 2: {
            do_kernel_warp_restore_context_segment_2(out_ticket, idx, task_ctx);
            do_kernel_warp_segment_2(gc->qp, out_ticket, idx, gc->num_coroutines);
            do_kernel_warp_save_context_segment_2(idx, task_ctx);

            if (threadIdx.x == 0) {
                task_ctx.state = (idx < gc->num_iters) ? 1 : 3;
            }
            __syncthreads();
            if (task_ctx.state == 1) {
                goto case1;   // loop back to segment 1 for the next request
            }
            break;
        }
    }

}

__global__
void launchPersistentKernelWarp(CUDAQueue<int>** d_task_queues, GlobalContext* gc, TaskContext* tc)
{   
    int block_id = blockIdx.x;
    CUDAQueue<int>* task_queue = d_task_queues[block_id];   // find the task queue according to the block id
    
    while (true) {
        __shared__ int task_id;
        __shared__ bool pop_ok;
        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            pop_ok = queuePop(task_queue, task_id);
            // printf("Popped task_id %d by thread block: %d\n", task_id, blockIdx.x);
        }
        __syncthreads();
        if (!pop_ok) {
            // printf("warning: no task to pop for block %d\n", 
            //        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
            break;
        }

        do_kernel_warp(gc, tc, task_id);
        __syncthreads();

        if (threadIdx.x + threadIdx.y + threadIdx.z == 0) { // first thread of the block
            if (tc[task_id].state >= 3) { // task done
                // printf("Task %d completed all states, state = %d.\n", task_id, tc[task_id].state);
            } else {
                queuePush(d_task_queues[blockIdx.x], task_id);
            }
        }
        __syncthreads();
    } // end while

    return ;
}


extern "C" {

doca_error_t gpunetio_verbs_put_bw_coro(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads, uint32_t num_coroutines,
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


    const int TASK_NUM = cuda_blocks * num_coroutines;   // total number of tasks (block-level)
    std::vector<CUDAQueue<int>*> h_task_queues;
    for (int i = 0; i < cuda_blocks; i++) {
        CUDAQueue<int>* d_task_queue;
        cudaMalloc((void**)&d_task_queue, sizeof(CUDAQueue<int>));
        queueInit<int>(d_task_queue, num_coroutines);
        h_task_queues.push_back(d_task_queue);
    }
    CUDAQueue<int>** d_task_queues;
    cudaMalloc((void**)&d_task_queues, sizeof(CUDAQueue<int>*) * cuda_blocks);
    cudaMemcpy(d_task_queues, h_task_queues.data(), sizeof(CUDAQueue<int>*) * cuda_blocks, cudaMemcpyHostToDevice);
    
    // 2. task contexts
    GlobalContext* global_ctx;
    cudaMalloc((void**)&global_ctx, sizeof(GlobalContext));
    TaskContext* task_ctx;
    cudaMalloc((void**)&task_ctx, sizeof(TaskContext) * TASK_NUM);

    // 3. init kernel
    initPersistentKernel<<<cuda_blocks, cuda_threads>>>(d_task_queues, num_coroutines, global_ctx, task_ctx,
                                                        qp, num_iters, data_size, src_buf, src_buf_mkey, dst_buf, dst_buf_mkey, timer);
    cudaDeviceSynchronize();

    // 4. launch persistent kernel
    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
        launchPersistentKernelThread<<<cuda_blocks, cuda_threads, 0, stream>>>(
            d_task_queues, global_ctx, task_ctx);
    else if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
        launchPersistentKernelWarp<<<cuda_blocks, cuda_threads, 0, stream>>>(
            d_task_queues, global_ctx, task_ctx);

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
    // printf("Average time per iteration for size %d: [put %f us, poll %f us, total %f us]\n", data_size, (float)sum_put / num_iters / 1000, (float)sum_poll / num_iters / 1000, (float)(sum_put + sum_poll) / num_iters / 1000);
    cudaFree(timer);
    free(host_timer);
#endif

    return DOCA_SUCCESS;
}
}
