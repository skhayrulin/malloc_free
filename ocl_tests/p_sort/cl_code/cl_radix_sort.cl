/*******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2011, 2017 OpenWorm.
 * http://openworm.org
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License
 * which accompanies this distribution, and is available at
 * http://opensource.org/licenses/MIT
 *
 * Contributors:
 *     	OpenWorm - http://openworm.org/people.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PRINTF_ON // this comment because using printf leads to very slow work on Radeon r9 290x on my machine
// don't know why
#elif defined(cl_intel_printf)
#pragma OPENCL EXTENSION cl_intel_printf : enable
#define PRINTF_ON
#endif

#ifdef _DOUBLE_PRECISION
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#define _DOUBLE_PRECISION
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif
#endif

#include "util/radixsort.h"
/** COUNT KERNEL **/

__kernel void count(
    const __global int* input,
    __global int* output,
    __local int* local_histo,
    const int pass,
    const int nkeys)
{
    uint g_id = (uint)get_global_id(0); // gloabal id for ellement in group
    uint l_id = (uint)get_local_id(0); // local id for ellement in group
    uint l_size = (uint)get_local_size(0); // local size of group

    uint group_id = (uint)get_group_id(0); // id of group
    uint n_groups = (uint)get_num_groups(0); // number of groups

    //Set the buckets of each item to 0
    int i;
    for (i = 0; i < BUCK; i++) {
        local_histo[i * l_size + l_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Calculate elements to process per item
    int size = (nkeys / n_groups) / l_size;

    //Calculate where to start on the global array
    int start = g_id * size;
    for (i = 0; i < size; i++) {
        int key = input[i + start];
        //Extract the corresponding radix of the key
        key = ((key >> (pass * RADIX)) & (BUCK - 1));
        //Count the ocurrences in the corresponding bucket
        ++local_histo[key * l_size + l_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (i = 0; i < BUCK; i++) {
        //"from" references the local buckets
        int from = i * l_size + l_id;
        //"to" maps to the global buckets
        int to = i * n_groups + group_id;
        //Map the local data to its global position
        output[l_size * to + l_id] = local_histo[from];
    }
    //printf("\nSTEP - %d, LEN - %d, mid - %d, first start - %d, first end - %d, second start - %d, second end - %d, id - %d, start - %d\n", step, len, mid, first_sub_array_start,first_sub_array_end, second_sub_array_start, second_sub_array_end, id, start);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

/** SCAN KERNEL **/
__kernel void scan(
    __global int* input,
    __global int* output,
    __local int* local_scan,
    __global int* block_sum)
{
    uint g_id = (uint)get_global_id(0);
    uint l_id = (uint)get_local_id(0);
    uint l_size = (uint)get_local_size(0);

    uint group_id = (uint)get_group_id(0);
    uint n_groups = (uint)get_num_groups(0);

    //Store data from global to local memory to operate
    local_scan[2 * l_id] = input[2 * g_id];
    local_scan[2 * l_id + 1] = input[2 * g_id + 1];

    //UP SWEEP
    int d, offset = 1;
    for (d = l_size; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_id < d) {
            int a = offset * (2 * l_id + 1) - 1;
            int b = offset * (2 * l_id + 2) - 1;
            local_scan[b] += local_scan[a];
        }
        offset *= 2;
    }

    if (l_id == 0) {
        //Store the full sum on last item
        if (block_sum != NULL) {
            block_sum[group_id] = local_scan[l_size * 2 - 1];
        }

        //Clear the last element
        local_scan[l_size * 2 - 1] = 0;
    }

    //DOWN SWEEP
    for (d = 1; d < (l_size * 2); d *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_id < d) {
            int a = offset * (2 * l_id + 1) - 1;
            int b = offset * (2 * l_id + 2) - 1;
            int tmp = local_scan[a];
            local_scan[a] = local_scan[b];
            local_scan[b] += tmp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //Write results from Local to Global memory
    output[2 * g_id] = local_scan[2 * l_id];
    output[2 * g_id + 1] = local_scan[2 * l_id + 1];
}

/** COALESCE KERNEL **/
__kernel void coalesce(
    __global int* scan,
    __global int* block_sums)
{

    uint g_id = (uint)get_global_id(0);
    uint group_id = (uint)get_group_id(0);

    int b = block_sums[group_id];

    scan[2 * g_id] += b;
    scan[2 * g_id + 1] += b;

    barrier(CLK_GLOBAL_MEM_FENCE);
}

/** REORDER KERNEL **/
__kernel void reorder(
    __global int* array,
    __global int* histo,
    __global int* output,
    const int pass,
    const int nkeys,
    __local int* local_histo)
{
    uint g_id = (uint)get_global_id(0);
    uint l_id = (uint)get_local_id(0);
    uint l_size = (uint)get_local_size(0);

    uint group_id = (uint)get_group_id(0);
    uint n_groups = (uint)get_num_groups(0);

    //Bring histo to local memory
    int i;
    for (i = 0; i < BUCK; i++) {
        int to = i * n_groups + group_id;
        local_histo[i * l_size + l_id] = histo[l_size * to + l_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Write to global memory in order
    int size = (nkeys / n_groups) / l_size;
    int start = g_id * size;
    for (i = 0; i < size; i++) {
        int key = array[i + start];
        key = (key >> (pass * RADIX)) & (BUCK - 1);
        int pos = local_histo[key * l_size + l_id];
        ++local_histo[key * l_size + l_id];

        output[pos] = key;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
}
