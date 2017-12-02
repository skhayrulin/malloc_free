#ifndef SOLVER_CONTAINER
#define SOLVER_CONTAINER
#include "ocl_solver.h"
#include <memory>
#include <vector>

class solver_container {
public:
  solver_container &operator=(const solver_container &) = delete;
  solver_container(const solver_container &) = delete;
  static solver_container &instance(size_t count, cv::Mat img) {
    static solver_container s_c(count, img);
    return s_c;
  }
  cv::Mat run();

private:
  void synk() {}
  solver_container(size_t count, cv::Mat img);
  std::vector<std::shared_ptr<ocl_solver>> container;
};
#endif // SOLVER_CONTAINER