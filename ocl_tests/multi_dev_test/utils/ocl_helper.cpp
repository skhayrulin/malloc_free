#include "ocl_helper.h"
#include "error.h"
#include <iostream>
enum DEVICE { CPU = 0, GPU = 1, ALL = 2 };

std::priority_queue<std::shared_ptr<device>> get_dev_queue() {
  std::priority_queue<std::shared_ptr<device>> q;
  if (q.size() < 1) {
    throw ocl_error("No OpenCL devices were found");
  }
  return q;
}

size_t get_device_count(cl::Platform &p) {
  cl::Context contex;
  cl_context_properties cprops[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[pl_id])(), 0};
  context = cl::Context(device_type[ALL], cprops, NULL, NULL, &err);
  std::vector<cl::Device> devices;
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  return device.size();
}

void init_cl_devices() {
  cl_int err;
  cl::Context context;
  std::vector<cl::Platform> platform_list;
  err = cl::Platform::get(
      &platform_list); // TODO make check that returned value isn't error
  if (platform_list.size() < 1 || err != CL_SUCCESS) {
    throw ocl_error("No OpenCL platforms were found");
  }
  cl_platform_id cl_pl_id[10];
  cl_uint n_pl;
  clGetPlatformIDs(10, cl_pl_id, &n_pl);
  cl_int ciErrNum;
  int sz;
  char c_buffer[100];
  for (int i = 0; i < (int)n_pl; i++) {
    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo(cl_pl_id[i], CL_PLATFORM_VERSION,
                                 sz = sizeof(c_buffer), c_buffer, NULL);
    if (ciErrNum == CL_SUCCESS) {
      std::cout << "CL_PLATFORM_VERSION [" << i << "]: \t" << c_buffer << "\n";
    } else {
      std::cout << "No information available for platform "
                << "\n ";
    }
  }
  // 0-CPU, 1-GPU // depends on the time order of system OpenCL drivers
  // installation on your local machine
  // CL_DEVICE_TYPE
  // added autodetection of device number corresonding to preferrable device
  // type (CPU|GPU) | otherwise the choice will be made from list of existing
  // devices
  cl_uint ciDeviceCount = 0;
  cl_device_id *devices_t;
  cl_int result;
  unsigned int deviceNum = 0;
  // Selection of more appropriate device
  // Select platforms with bigest amount of devices
  auto it =
      std::max_element(platform_list.begin(), platform_list.end(),
                       [&](cl::Platform &p1, cl::Platform &p2) {
                         return get_device_count(p1) > get_device_count(p2);
                       });

  cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM,
                                     (cl_context_properties)(*it)(), 0};
  context = cl::Context(device_type[ALL], cprops, NULL, NULL, &err);
  std::vector<cl::Device> devices;
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  for (size_t i = 0; i < devices.size(); ++i) {
    std::shared_ptr<device> d(new device(1, i, devices[i], context));
  }