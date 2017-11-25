#include "solver_container.h"

solver_container::solver_container(size_t count, cv::Mat img) {
  std::vector<int> lims = {0, 0, img.rows, img.cols >> 1};
  for (size_t i = 0; i < count; ++i) {
    std::shared_ptr<ocl_solver> solver(new ocl_solver(img, lims));
    container.push_back(solver);
  }
}

cv::Mat solver_container::run() {
  for (auto s : container) {
    cv::Mat result = s->run();
    return result;
  }
}