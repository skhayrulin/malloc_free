#include "ocl_helper.h"
#include <iostream>

enum DEVICE { CPU = 0, GPU = 1, ALL = 2 };

std::priority_queue<std::shared_ptr<device>> get_dev_queue() {
  std::priority_queue<std::shared_ptr<device>> q;
  if (q.size() < 1) {
    throw std::runtime_error("No OpenCL devices were found");
  }
  return q;
}

void init_cl_devices() {
  cl_int err;
  cl::Context context;
  std::vector<cl::Platform> platformList;
  err = cl::Platform::get(
      &platformList); // TODO make check that returned value isn't error
  if (platformList.size() < 1 || err != CL_SUCCESS) {
    throw std::runtime_error("No OpenCL platforms found");
  }
  char cBuffer[1024];
  cl_platform_id cl_pl_id[10];
  cl_uint n_pl;
  clGetPlatformIDs(10, cl_pl_id, &n_pl);
  cl_int ciErrNum;
  int sz;
  for (int i = 0; i < (int)n_pl; i++) {
    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo(cl_pl_id[i], CL_PLATFORM_VERSION,
                                 sz = sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS) {
      printf(" CL_PLATFORM_VERSION [%d]: \t%s\n", i, cBuffer);
    } else {
      printf(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
    }
  }
  // 0-CPU, 1-GPU // depends on the time order of system OpenCL drivers
  // installation on your local machine
  // CL_DEVICE_TYPE
  cl_device_type type;
  unsigned int device_type[] = {CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU,
                                CL_DEVICE_TYPE_ALL};

  int plList = -1; // selected platform index in platformList array [choose CPU
                   // by default]
  // added autodetection of device number corresonding to preferrable device
  // type (CPU|GPU) | otherwise the choice will be made from list of existing
  // devices
  cl_uint ciDeviceCount = 0;
  cl_device_id *devices_t;
  bool bPassed = true, findDevice = false;
  cl_int result;
  cl_uint device_coumpute_unit_num;
  cl_uint device_coumpute_unit_num_current = 0;
  unsigned int deviceNum = 0;
  // Selection of more appropriate device
  while (!findDevice) {
    for (int pl_id = 0; pl_id < (int)n_pl; pl_id++) {

      cl_context_properties cprops[3] = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[pl_id])(),
          0};
      context = cl::Context(device_type[ALL], cprops, NULL, NULL, &err);
      std::vector<cl::Device> devices;
      devices = context.getInfo<CL_CONTEXT_DEVICES>();
      for (size_t i = 0; i < devices.size(); ++i) {
        std::shared_ptr<device> d(new device());
        d->dev_id = i;
        d->platform_id = pl_id;
        d->device = devices[i];
        d->context = context;
        char name[100];
        result = devices[i].getInfo(CL_DEVICE_NAME,
                                    &name); // CL_INVALID_VALUE = -30;
        if (result == CL_SUCCESS) {
          d->name = name;
        } else {
          continue;
        }
      }
    }
  }
}