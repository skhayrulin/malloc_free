#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__MACOSX)
#include "../inc/OpenCL/cl.hpp"
//	#include <OpenCL/cl_d3d10.h>
#else
#include <CL/cl.hpp>
#endif

#include "cl_struct.h"
using std::cout;
using std::endl;
// TODO write the docs
template <class T, size_t dim = 4> struct alignas(16) particle {
  typedef std::array<T, dim> container;
  container pos;
  container vel;
  size_t type;
  size_t cell_id;
  size_t get_dim() const { return dim; }
  T density;
  T pressure;
  std::string pos_str() {
    std::stringstream s;
    std::for_each(pos.begin(), pos.end(), [&s](T c) { s << c << ' '; });
    s << '\n';
    return s.str();
  }
};

cl::Kernel work_with_struct;
cl::Kernel _init_ext_particles;
cl::Buffer cl_particles;
cl::Buffer ext_particles;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;
const size_t size = 10;
const int local_NDRange_size = 256;
std::vector<particle<float>> particles(size);
enum DEVICE { CPU = 0, GPU = 1, ALL = 2 };
DEVICE get_device_type() { return ALL; }
void initialize_ocl() {
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
  std::ifstream file("./test.cl");
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
void create_ocl_buffer(const char *name, cl::Buffer &b,
                       const cl_mem_flags flags, const int size) {
  int err;
  b = cl::Buffer(context, flags, size, NULL, &err);
  if (err != CL_SUCCESS) {
    std::string error_m = "Buffer creation failed: ";
    error_m.append(name);
    throw std::runtime_error(error_m);
  }
}
void create_ocl_kernel(const char *name, cl::Kernel &k) {
  int err;
  k = cl::Kernel(program, name, &err);
  if (err != CL_SUCCESS) {
    std::string error_m = "Kernel creation failed: ";
    error_m.append(name);
    throw std::runtime_error(error_m);
  }
}

void copy_buffer_to_device(const void *host_b, cl::Buffer &ocl_b,
                           const int size) {
  // Actualy we should check  size and type
  int err = queue.enqueueWriteBuffer(ocl_b, CL_TRUE, 0, size, host_b);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "Could not enqueue read data from buffer  error code is");
  }
  queue.finish();
}

void copy_buffer_from_device(void *host_b, const cl::Buffer &ocl_b,
                             const int size) {
  // Actualy we should check  size and type
  int err = queue.enqueueReadBuffer(ocl_b, CL_TRUE, 0, size, host_b);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "Could not enqueue read data from buffer  error code is");
  }
  queue.finish();
}

template <class T> void init_ocl_stuff() {
  create_ocl_buffer("particles", cl_particles, CL_MEM_READ_WRITE,
                    particles.size() * sizeof(particle<T>));
  create_ocl_buffer("ext_particles", ext_particles, CL_MEM_READ_WRITE,
                    particles.size() * sizeof(extendet_particle));
  create_ocl_kernel("work_with_struct", work_with_struct);
  create_ocl_kernel("_init_ext_particles", _init_ext_particles);
  copy_buffer_to_device(&particles[0], cl_particles,
                        particles.size() * sizeof(particle<T>));
}

unsigned int _run_work_with_struct() {
  // Stage HashParticles
  work_with_struct.setArg(0, ext_particles);
  work_with_struct.setArg(1, cl_particles);
  int err = queue.enqueueNDRangeKernel(work_with_struct, cl::NullRange,
                                       cl::NDRange(particles.size()),
#if defined(__APPLE__)
                                       cl::NullRange, NULL, NULL);
#else
                                       cl::NDRange((int)(local_NDRange_size)),
                                       NULL, NULL);
#endif
#if QUEUE_EACH_KERNEL
  queue.finish();
#endif
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "An ERROR is appearing during work of kernel _runHashParticles");
  }
  return err;
}

unsigned int _run_init_ext_particles() {
  // Stage HashParticles
  _init_ext_particles.setArg(0, ext_particles);
  int err = queue.enqueueNDRangeKernel(_init_ext_particles, cl::NullRange,
                                       cl::NDRange(particles.size()),
#if defined(__APPLE__)
                                       cl::NullRange, NULL, NULL);
#else
                                       cl::NDRange((int)(local_NDRange_size)),
                                       NULL, NULL);
#endif
#if QUEUE_EACH_KERNEL
  queue.finish();
#endif
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "An ERROR is appearing during work of kernel _init_ext_particles");
  }
  return err;
}

int main() {
  cout << "SIZEOF PARTICLE STRUCT FLOAT IS " << sizeof(particle<float>) << endl;
  cout << "SIZEOF PARTICLE STRUCT DOUBLE IS " << sizeof(particle<double>)
       << endl;

  initialize_ocl();
  init_ocl_stuff<float>();
  _run_init_ext_particles();
  _run_work_with_struct();
  copy_buffer_from_device(&particles[0], cl_particles,
                          particles.size() * sizeof(particle<float>));
  std::for_each(particles.begin(), particles.end(),
                [](particle<float> &p) { cout << p.pos_str(); });
  return 0;
}