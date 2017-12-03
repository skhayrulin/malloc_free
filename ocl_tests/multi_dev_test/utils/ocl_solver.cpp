#include "ocl_solver.h"

const int local_NDRange_size = 256;

ocl_solver::ocl_solver(cv::Mat img, const std::vector<int> &rec,
                       std::shared_ptr<device> dev)
    : dev(dev) {
  assert(rec.size() == 4);

  c.rows = rec[2];
  c.cols = rec[3];
  c.start_x = rec[0];
  c.start_y = rec[1];
  c.total = c.rows * c.cols;
  init_ocl();
  init_buffs(img);
  init_kernels();
}
void ocl_solver::init_buffs(cv::Mat img) {
  create_buffer("img", buf_img, CL_MEM_READ_WRITE, c.total * 4 * sizeof(float));
  create_buffer("ret_img", buf_res_img, CL_MEM_READ_WRITE,
                c.total * 4 * sizeof(float));
  create_buffer("mask", buf_mask, CL_MEM_READ_WRITE, 9 * sizeof(float));
  float m[] = {0.f,       1.f / 6.f, 0.f,       1.f / 6.f, 1.f / 3.f,
               1.f / 6.f, 0.f,       1.f / 6.f, 0.f};
  std::vector<std::array<float, 4>> data(c.total);
  for (int i = 0; i < c.rows; ++i) {
    for (int j = 0; j < c.cols; ++j) {
      cv::Vec3f v = img.at<cv::Vec3f>(c.start_x + i, c.start_y + j);
      data[i * c.cols + j][0] = v[0];
      data[i * c.cols + j][1] = v[1];
      data[i * c.cols + j][2] = v[2];
      // data[i * c.cols + j][3] = v[3];
    }
  }
  copy_buffer_to_device((void *)(&data[0]), buf_img,
                        c.total * sizeof(float) * 4);
  copy_buffer_to_device((void *)m, buf_mask, sizeof(float) * 9);
}
void ocl_solver::init_ocl() {
  int err;
  queue = cl::CommandQueue(dev->context, dev->dev, 0, &err);
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
  b = cl::Buffer(dev->context, flags, size, NULL, &err);
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
cv::Mat ocl_solver::convert_to_mat(std::vector<std::array<float, 4>> &buffer) {
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
  return convert_to_mat(data);
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