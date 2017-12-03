#ifndef OCL_SOLVER
#define OCL_SOLVER
#include "device.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <string>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__MACOSX)
#include "../OpenCL/cl.hpp"
//	#include <OpenCL/cl_d3d10.h>
#else
#include <CL/cl.hpp>
#endif
struct config {
  const std::string cl_program_file = "cl_code/blur.cl";
  int rows;
  int cols;
  int start_x;
  int start_y;
  size_t total;
};
class ocl_solver {
public:
  ocl_solver(cv::Mat, const std::vector<int> &, std::shared_ptr<device>);
  cv::Mat run();

private:
  void init_ocl();
  void init_kernels();
  void init_buffs(cv::Mat img);
  void create_kernel(const char *name, cl::Kernel &k);
  void create_buffer(const char *name, cl::Buffer &b, const cl_mem_flags flags,
                     const int size);
  unsigned int _run_kernel_blur();
  void copy_buffer_to_device(const void *host_b, cl::Buffer &ocl_b,
                             const size_t size);
  void copy_buffer_from_device(void *host_b, const cl::Buffer &ocl_b,
                               const int size);
  cv::Mat convert_to_mat(std::vector<std::array<float, 4>> &buffer);
  cl::Kernel ker_blur;
  cl::Buffer buf_img;
  cl::Buffer buf_res_img;
  cl::Buffer buf_mask;
  cl::CommandQueue queue;
  cl::Program program;
  std::shared_ptr<device> dev;
  config c;
};

#endif // OCL_SOLVER
