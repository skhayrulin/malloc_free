#ifndef DEVICE
#define DEVICE
#include <iostream>
#include <string>
#if defined(__APPLE__) || defined(__MACOSX)
#include "../inc/OpenCL/cl.hpp"
//	#include <OpenCL/cl_d3d10.h>
#else
#include <CL/cl.hpp>
#endif
enum type { CPU, GPU };
struct device {
  type t;
  int platform_id;
  int dev_id;
  int is_busy;
  std::string name;
  cl::Device device;
  cl::Context context;
  int device_coumpute_unit_num; // criteria to sort devices
  bool operator<(const device &d1) {
    return device_coumpute_unit_num < d1.device_coumpute_unit_num;
  }
  void show_info() {
    char c_buffer[100];
    cl_int result;
    int work_group_size, comp_unints_count, memory_info;
    // Print some information about chosen platform
    result = device.getInfo(CL_DEVICE_NAME,
                            &c_buffer); // CL_INVALID_VALUE = -30;
    if (result == CL_SUCCESS) {
      result = device.getInfo(CL_DEVICE_NAME,
                              &c_buffer); // CL_INVALID_VALUE = -30;
      if (result == CL_SUCCESS) {
        std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                  << "]: CL_DEVICE_NAME [" << dev_id << "]:\t" << c_buffer
                  << "\n"
                  << std::endl;
      }
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_NAME [" << dev_id << "]:\t" << c_buffer << "\n"
                << std::endl;
    }
    if (strlen(c_buffer) < 1000) {
    }
    result = device.getInfo(CL_DEVICE_TYPE, &c_buffer);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_TYPE [" << dev_id << "]:\t"
                << (((int)c_buffer[0] == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU")
                << std::endl;
    }
    result = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &work_group_size);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_MAX_WORK_GROUP_SIZE [" << dev_id << "]: \t"
                << work_group_size << std::endl;
    }
    result = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &comp_unints_count);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_MAX_COMPUTE_UNITS [" << dev_id << "]: \t"
                << comp_unints_count << std::endl;
    }
    result = device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &memory_info);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_GLOBAL_MEM_SIZE [" << dev_id << "]: \t"
                << memory_info << std::endl;
    }
    result = device.getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &memory_info);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM [" << platform_id
                << "]: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE [" << dev_id << "]:\t"
                << memory_info << std::endl;
    }
    result = device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &memory_info);
    if (result == CL_SUCCESS) {
      std::cout << "CL_CONTEXT_PLATFORM " << platform_id
                << ": CL_DEVICE_LOCAL_MEM_SIZE [" << dev_id << "]:\t"
                << memory_info << std::endl;
    }
  }
};

#endif