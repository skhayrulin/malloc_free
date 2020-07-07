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
#ifndef OCL_RADIX_SORT_SOLVER_HPP
#define OCL_RADIX_SORT_SOLVER_HPP

#if defined(_WIN32) || defined(_WIN64)
#pragma comment(lib, "opencl.lib") // opencl.lib
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/cl.hpp"
#else
#include <CL/cl.hpp>
#endif
#include "device.h"
#include "radixsort.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>

using std::cout;
using std::endl;
using std::shared_ptr;
// OCL constans block
#define QUEUE_EACH_KERNEL 1

enum LOGGING_MODE {
    FULL,
    NO
};

template <class T = int>
class ocl_radix_sort_solver {
public:
    ocl_radix_sort_solver(std::vector<T>& m, shared_ptr<device> d, LOGGING_MODE log_mode = LOGGING_MODE::NO)
        : model(m)
        , dev(d)
        , log_mode(log_mode)
    {
        auto tmp = WG_SIZE * N_GROUPS;
        size = (m.size() + (tmp - m.size() % tmp));
        array_dataSize = size * sizeof(int);
        try {
            this->initialize_ocl();
            init_model();
        } catch (...) {
            throw;
        }
    }
    std::shared_ptr<device> get_device()
    {
        return this->dev;
    }
    // TODO rename method!!!
    void init_model()
    {
        init_buffers();
        init_kernels();
    }

    ~ocl_radix_sort_solver() { }

    void sort()
    {
        copy_buffer_to_device((void*)(&(model[0])), array_buffer, 0, model.size() * sizeof(int));

        for (int pass = 0; pass < BITS / RADIX; ++pass) {
            run_count(pass);
            run_scan();
            run_blocksum();
            run_coalesce();
            run_reorder(pass);
            auto tmp = array_buffer;
            array_buffer = output_buffer;
            output_buffer = tmp;
        }
        copy_buffer_from_device((void*)(&(model[0])), output_buffer, model.size() * sizeof(int), 0);
    }

private:
    std::vector<T>& model;
    shared_ptr<device> dev;
    std::string msg = dev->name + '\n';
    const std::string cl_program_file = "cl_code//cl_radix_sort.cl";
    LOGGING_MODE log_mode;
    cl::Kernel count;
    cl::Kernel scan;
    cl::Kernel blocksum;
    cl::Kernel coalesce;
    cl::Kernel reorder;
    cl::Kernel shuffle;

    cl::Buffer array_buffer;
    cl::Buffer histo_buffer;
    cl::Buffer scan_buffer;
    cl::Buffer blocksum_buffer;
    cl::Buffer output_buffer;

    cl::Buffer null_buffer;

    cl::CommandQueue queue;
    cl::Program program;

    int size;
    int array_dataSize;

    void init_buffers()
    {
        create_ocl_buffer("array_buffer", array_buffer, CL_MEM_READ_WRITE, sizeof(int) * model.size());
        //Create histo buff
        create_ocl_buffer("histo_buffer", histo_buffer, CL_MEM_READ_WRITE, sizeof(int) * BUCK * N_GROUPS * WG_SIZE);
        //Create scan buff
        create_ocl_buffer("scan_buffer", scan_buffer, CL_MEM_READ_WRITE, sizeof(int) * BUCK * N_GROUPS * WG_SIZE);
        //Create blocksum buff
        create_ocl_buffer("blocksum_buffer", blocksum_buffer, CL_MEM_READ_WRITE, sizeof(int) * N_GROUPS);
        //Create output buff
        create_ocl_buffer("output_buffer", output_buffer, CL_MEM_READ_WRITE, sizeof(int) * model.size());
    }

    void init_kernels()
    {
        create_ocl_kernel("count", count);
        create_ocl_kernel("scan", scan);
        create_ocl_kernel("scan", blocksum);
        create_ocl_kernel("coalesce", coalesce);
        create_ocl_kernel("reorder", reorder);
    }

    void initialize_ocl()
    {
        int err;
        queue = cl::CommandQueue(dev->context, dev->dev, 0, &err);
        if (err != CL_SUCCESS) {
            throw "Failed to create command queue";
        }
        std::ifstream file(cl_program_file);
        if (!file.is_open()) {
            throw "Could not open file with OpenCL program check input arguments oclsourcepath: ";
        }
        std::string programSource(std::istreambuf_iterator<char>(file),
            (std::istreambuf_iterator<char>()));
        // TODO fix this to param
        if (0) {
            programSource = "#define _DOUBLE_PRECISSION\n" + programSource; // not now it needs double extension check on device
        }
        cl::Program::Sources source(
            1, std::make_pair(programSource.c_str(), programSource.length() + 1));
        program = cl::Program(dev->context, source);
#if defined(__APPLE__)
        err = program.build("-g -cl-opt-disable -I .");
#else
#if INTEL_OPENCL_DEBUG
        err = program.build(OPENCL_DEBUG_PROGRAM_PATH + "-g -cl-opt-disable -I .");
#else
        err = program.build("-I .");
#endif
#endif
        if (err != CL_SUCCESS) {
            std::string compilationErrors;
            compilationErrors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev->dev);
            msg += "Compilation failed: " + compilationErrors;
            throw msg;
        }
        std::cout
            << msg
            << "OPENCL program was successfully build. Program file oclsourcepath: "
            << cl_program_file << std::endl;
    }

    void create_ocl_buffer(const char* name, cl::Buffer& b,
        const cl_mem_flags flags, const int size)
    {
        int err;
        b = cl::Buffer(dev->context, flags, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::string error_m = "Buffer creation failed" + std::string(name);
            throw error_m;
        }
    }

    void create_ocl_kernel(const char* name, cl::Kernel& k)
    {
        int err;
        k = cl::Kernel(program, name, &err);
        if (err != CL_SUCCESS) {
            std::string error_m = "Kernel creation failed" + std::string(name);
            throw error_m;
        }
    }

    void copy_buffer_to_device(
        const void* host_b,
        cl::Buffer& ocl_b,
        const size_t offset,
        const size_t size)
    {
        // Actually we should check  size and type
        int err = queue.enqueueWriteBuffer(ocl_b, CL_TRUE, offset, size, host_b);
        if (err != CL_SUCCESS) {
            std::string error_m = "Copy buffer to device is failed error code is " + std::to_string(err);
            throw error_m;
        }
        queue.finish();
    }

    void copy_buffer_from_device(void* host_b, const cl::Buffer& ocl_b,
        const size_t size, size_t offset)
    {
        // Actualy we should check  size and type
        int err = queue.enqueueReadBuffer(ocl_b, CL_TRUE, offset, size, host_b);
        if (err != CL_SUCCESS) {
            std::string error_m = "Copy buffer from device is failed error code is " + std::to_string(err);
            throw error_m;
        }
        queue.finish();
    }

    int run_count(int pass)
    {
        static size_t count_global_work_size = N_GROUPS * WG_SIZE;
        static size_t count_local_work_size = WG_SIZE;
        if (log_mode == LOGGING_MODE::FULL)
            std::cout << "run count kernel --> " << dev->name << std::endl;
        return this->kernel_runner(
            this->count,
            count_global_work_size,
            count_local_work_size,
            0,
            this->array_buffer,
            this->histo_buffer,
            cl::Local(sizeof(int) * BUCK * WG_SIZE),
            pass,
            size);
    }

    int run_scan()
    {
        //Scan fixed args
        static size_t scan_global_work_size = (BUCK * N_GROUPS * WG_SIZE) / 2;
        static size_t scan_local_work_size = scan_global_work_size / N_GROUPS;
        if (log_mode == LOGGING_MODE::FULL)
            std::cout << "run scan kernel --> " << dev->name << std::endl;
        this->kernel_runner(
            this->scan,
            scan_global_work_size,
            scan_local_work_size,
            0,
            this->histo_buffer,
            this->scan_buffer,
            cl::Local(sizeof(int) * BUCK * WG_SIZE),
            blocksum_buffer);
    }

    int run_blocksum()
    {
        //Scan fixed args
        static size_t blocksum_global_work_size = N_GROUPS / 2;
        static size_t blocksum_local_work_size = N_GROUPS / 2;
        if (log_mode == LOGGING_MODE::FULL)
            std::cout << "run blocksum kernel --> " << dev->name << std::endl;
        this->kernel_runner(
            this->blocksum,
            blocksum_global_work_size,
            blocksum_local_work_size,
            0,
            this->blocksum_buffer,
            this->blocksum_buffer,
            cl::Local(sizeof(int) * N_GROUPS),
            null_buffer);
    }

    int run_coalesce()
    {
        //Scan fixed args
        //Coalesce fixed args
        static size_t coalesce_global_work_size = (BUCK * N_GROUPS * WG_SIZE) / 2;
        static size_t coalesce_local_work_size = coalesce_global_work_size / N_GROUPS;

        if (log_mode == LOGGING_MODE::FULL)
            std::cout << "run coalesce kernel --> " << dev->name << std::endl;
        this->kernel_runner(
            this->coalesce,
            coalesce_global_work_size,
            coalesce_local_work_size,
            0,
            this->scan_buffer,
            this->blocksum_buffer);
    }

    int run_reorder(int pass)
    {
        //Reorder fixed args
        static size_t reorder_global_work_size = N_GROUPS * WG_SIZE;
        static size_t reorder_local_work_size = WG_SIZE;
        if (log_mode == LOGGING_MODE::FULL)
            std::cout << "run reorder kernel --> " << dev->name << std::endl;
        this->kernel_runner(
            this->reorder,
            reorder_global_work_size,
            reorder_local_work_size,
            0,
            this->array_buffer,
            this->scan_buffer,
            this->output_buffer,
            pass,
            this->size,
            cl::Local(sizeof(int) * BUCK * WG_SIZE));
    }

    template <typename U, typename... Args>
    int kernel_runner(
        cl::Kernel& ker,
        unsigned int global_dim,
        unsigned int local_dim,
        unsigned int arg_pos,
        U& arg,
        Args... args)
    {
        ker.setArg(arg_pos, arg);
        return kernel_runner(ker, global_dim, local_dim, ++arg_pos, args...);
    }

    template <typename U>
    int kernel_runner(
        cl::Kernel& ker,
        unsigned int global_dim,
        unsigned int local_dim,
        unsigned int arg_pos,
        U& arg)
    {
        ker.setArg(arg_pos, arg);

        int err = queue.enqueueNDRangeKernel(
            ker, cl::NullRange, cl::NDRange(global_dim),
#if defined(__APPLE__)
            cl::NullRange, nullptr, nullptr);
#else
            cl::NDRange(local_dim), nullptr, nullptr);
#endif
#if QUEUE_EACH_KERNEL
        queue.finish();
#endif
        if (err != CL_SUCCESS) {
            throw "An ERROR is appearing during work of kernel";
        }
        return err;
    }
};
#endif // OCL_SORT_SOLVER_HPP