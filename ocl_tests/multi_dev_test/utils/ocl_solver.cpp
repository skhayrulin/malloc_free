#include "ocl_solver.h"

const int local_NDRange_size = 256;
enum DEVICE { CPU = 0, GPU = 1, ALL = 2 };
DEVICE get_device_type() { return CPU; }

ocl_solver::ocl_solver(cv::Mat img, const std::vector<int> &rec) {
  assert(rec.size() == 4);
  c.rows = rec[2];
  c.cols = rec[3];
  init_ocl();
  init_buffs(img);
  init_kernels();
}
void ocl_solver::init_buffs(cv::Mat img) {
  // buf_img = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  //                      cl::ImageFormat(CL_BGRA, CL_FLOAT), c.cols, c.rows, 0,
  //                      (void *)img.data);
  c.total = c.rows * c.cols;
  create_buffer("img", buf_img, CL_MEM_READ_WRITE, c.total * 4 * sizeof(float));
  create_buffer("ret_img", buf_res_img, CL_MEM_READ_WRITE,
                c.total * 4 * sizeof(float));
  create_buffer("mask", buf_mask, CL_MEM_READ_WRITE, 9 * sizeof(float));
  float m[] = {0.f,       1.f / 6.f, 0.f,       1.f / 6.f, 1.f / 3.f,
               1.f / 6.f, 0.f,       1.f / 6.f, 0.f};
  std::vector<std::array<float, 4>> data(c.total);
  for (int i = 0; i < c.rows; ++i) {
    for (int j = 0; j < c.cols; ++j) {
      cv::Vec3f v = img.at<cv::Vec3f>(i, j);
      data[i * c.cols + j][0] = v[0]; // v[0];
      data[i * c.cols + j][1] = v[1]; // v[1];
      data[i * c.cols + j][2] = v[2]; // v[2];
      data[i * c.cols + j][3] = v[3]; // v[3];
    }
  }
  copy_buffer_to_device((void *)(&data[0]), buf_img,
                        c.total * sizeof(float) * 4);
  copy_buffer_to_device((void *)m, buf_mask, sizeof(float) * 9);
}
void ocl_solver::init_ocl() {
  cl_int err;
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
    for (int clSelectedPlatformID = 0; clSelectedPlatformID < (int)n_pl;
         clSelectedPlatformID++) {
      // if(findDevice)
      //	break;
      clGetDeviceIDs(cl_pl_id[clSelectedPlatformID],
                     device_type[get_device_type()], 0, NULL, &ciDeviceCount);
      if ((devices_t = static_cast<cl_device_id *>(
               malloc(sizeof(cl_device_id) * ciDeviceCount))) == NULL)
        bPassed = false;
      if (bPassed) {
        result = clGetDeviceIDs(cl_pl_id[clSelectedPlatformID],
                                device_type[get_device_type()], ciDeviceCount,
                                devices_t, &ciDeviceCount);
        if (result == CL_SUCCESS) {
          for (cl_uint i = 0; i < ciDeviceCount; ++i) {
            clGetDeviceInfo(devices_t[i], CL_DEVICE_TYPE, sizeof(type), &type,
                            NULL);
            if (type & device_type[get_device_type()]) {
              clGetDeviceInfo(devices_t[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(device_coumpute_unit_num),
                              &device_coumpute_unit_num, NULL);
              if (device_coumpute_unit_num_current <=
                  device_coumpute_unit_num) {
                plList = clSelectedPlatformID;
                device_coumpute_unit_num_current = device_coumpute_unit_num;
                findDevice = true;
                deviceNum = i;
              }
              // break;
            }
          }
        }
        free(devices_t);
      }
    }
    if (!findDevice) {
      // plList = 0;
      deviceNum = 0;
      std::string deviceTypeName =
          (get_device_type() == ALL)
              ? "ALL"
              : (get_device_type() == CPU) ? "CPU" : "GPU";
      std::cout << "Unfortunately OpenCL couldn't find device "
                << deviceTypeName << std::endl;
      std::cout << "OpenCL try to init existing device " << std::endl;
      if (get_device_type() != ALL) {
      } else
        throw std::runtime_error("Sibernetic can't find any OpenCL devices. "
                                 "Please check you're environment "
                                 "configuration.");
    }
  }
  cl_context_properties cprops[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[plList])(), 0};
  context =
      cl::Context(device_type[get_device_type()], cprops, NULL, NULL, &err);
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  if (devices.size() < 1) {
    throw std::runtime_error("No OpenCL devices were found");
  }
  // Print some information about chosen platform
  size_t compUnintsCount, memoryInfo, workGroupSize;
  result = devices[deviceNum].getInfo(CL_DEVICE_NAME,
                                      &cBuffer); // CL_INVALID_VALUE = -30;
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList << "]: CL_DEVICE_NAME ["
              << deviceNum << "]:\t" << cBuffer << "\n"
              << std::endl;
  }
  if (strlen(cBuffer) < 1000) {
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_TYPE, &cBuffer);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList << "]: CL_DEVICE_TYPE ["
              << deviceNum << "]:\t"
              << (((int)cBuffer[0] == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU")
              << std::endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &workGroupSize);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList
              << "]: CL_DEVICE_MAX_WORK_GROUP_SIZE [" << deviceNum << "]: \t"
              << workGroupSize << std::endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compUnintsCount);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList
              << "]: CL_DEVICE_MAX_COMPUTE_UNITS [" << deviceNum << "]: \t"
              << compUnintsCount << std::endl;
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &memoryInfo);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList
              << "]: CL_DEVICE_GLOBAL_MEM_SIZE [" << deviceNum << "]: \t"
              << deviceNum << std::endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &memoryInfo);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM [" << plList
              << "]: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE [" << deviceNum << "]:\t"
              << memoryInfo << std::endl;
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &memoryInfo);
  if (result == CL_SUCCESS) {
    std::cout << "CL_CONTEXT_PLATFORM " << plList
              << ": CL_DEVICE_LOCAL_MEM_SIZE [" << deviceNum << "]:\t"
              << memoryInfo << std::endl;
  }
  queue = cl::CommandQueue(context, devices[deviceNum], 0, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create command queue");
  }
  std::ifstream file(c.cl_program_file);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file with OpenCL program check "
                             "input arguments oclsourcepath: ./test.cl");
  }
  std::string programSource(std::istreambuf_iterator<char>(file),
                            (std::istreambuf_iterator<char>()));
  if (0) {
    programSource = "#define _DOUBLE_PRECISSION\n" + programSource;
  }
  cl::Program::Sources source(
      1, std::make_pair(programSource.c_str(), programSource.length() + 1));
  program = cl::Program(context, source);
#if defined(__APPLE__)
  err = program.build(devices, "-g -cl-opt-disable -I .");
#else
#if INTEL_OPENCL_DEBUG
  err = program.build(devices,
                      OPENCL_DEBUG_PROGRAM_PATH + "-g -cl-opt-disable -I .");
#else
  err = program.build(devices, "-I .");
#endif
#endif
  if (err != CL_SUCCESS) {
    std::string compilationErrors;
    compilationErrors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
    std::cerr << "Compilation failed: " << std::endl
              << compilationErrors << std::endl;
    throw std::runtime_error("failed to build program");
  }
  std::cout
      << "OPENCL program was successfully build. Program file oclsourcepath: "
      << "./test.cl" << std::endl;
  return;
}
void ocl_solver::create_buffer(const char *name, cl::Buffer &b,
                               const cl_mem_flags flags, const int size) {
  int err;
  b = cl::Buffer(context, flags, size, NULL, &err);
  if (err != CL_SUCCESS) {
    std::string error_m = "Buffer creation failed: ";
    error_m.append(name);
    throw std::runtime_error(error_m);
  }
}
void ocl_solver::create_kernel(const char *name, cl::Kernel &k) {
  int err;
  k = cl::Kernel(program, name, &err);
  if (err != CL_SUCCESS) {
    std::string error_m = "Kernel creation failed: ";
    error_m.append(name);
    throw std::runtime_error(error_m);
  }
}

void ocl_solver::copy_buffer_to_device(const void *host_b, cl::Buffer &ocl_b,
                                       const size_t size) {
  // Actualy we should check  size and type
  int err = queue.enqueueWriteBuffer(ocl_b, CL_TRUE, 0, size, host_b);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "Could not enqueue read data from buffer  error code is");
  }
  queue.finish();
}

unsigned int ocl_solver::_run_kernel_blur() {
  // Stage HashParticles
  ker_blur.setArg(0, buf_img);
  ker_blur.setArg(1, buf_res_img);
  ker_blur.setArg(2, buf_mask);
  ker_blur.setArg(3, c.cols);
  ker_blur.setArg(4, c.rows);
  int err = queue.enqueueNDRangeKernel(
      ker_blur, cl::NullRange, cl::NDRange(c.rows, c.cols), cl::NullRange);
  queue.finish();
  if (err != CL_SUCCESS) {
    std::cout << err << std::endl;
    throw std::runtime_error(
        "An ERROR is appearing during work of kernel ker_blur");
  }
  return err;
}
cv::Mat ocl_solver::convertToMat(std::vector<std::array<float, 4>> &buffer) {
  cv::Mat tmp(c.rows, c.cols, CV_32FC4);
  for (int x = 0; x < c.rows; x++) {
    for (int y = 0; y < c.cols; y++) {
      cv::Vec4f &v = tmp.at<cv::Vec4f>(x, y);
      v[0] = buffer[x * c.cols + y][0];
      v[1] = buffer[x * c.cols + y][1];
      v[2] = buffer[x * c.cols + y][2];
      // v[3] = buffer[x * c.cols + y][3];
    }
  }
  return tmp;
}
void ocl_solver::init_kernels() { create_kernel("ker_blur", ker_blur); }
cv::Mat ocl_solver::run() {
  _run_kernel_blur();
  std::vector<std::array<float, 4>> data(c.total);
  copy_buffer_from_device((void *)(&data[0]), buf_res_img,
                          c.total * sizeof(float) * 4);
  return convertToMat(data);
}

void ocl_solver::copy_buffer_from_device(void *host_b, const cl::Buffer &ocl_b,
                                         const int size) {
  // Actualy we should check  size and type
  int err = queue.enqueueReadBuffer(ocl_b, CL_TRUE, 0, size, host_b);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "Could not enqueue read data from buffer  error code is");
  }
  queue.finish();
}