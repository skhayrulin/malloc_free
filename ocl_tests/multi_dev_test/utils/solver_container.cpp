#include "solver_container.h"
#include "error.h"
#include "ocl_helper.h"
solver_container::solver_container(size_t count, cv::Mat img) {
  std::priority_queue<std::shared_ptr<device>> dev_q = get_dev_queue();
  int i = 0;
  while (!dev_q.empty()) {
    int cols = (dev_q.size() > 1) ? img.cols >> 1 : img.cols;
    std::vector<int> lims = {0, i * cols, img.rows, cols};
    ++i;
    std::shared_ptr<ocl_solver> solver(new ocl_solver(img, lims, dev_q.top()));
    dev_q.pop();
    container.push_back(solver);
  }
}

cv::Mat solver_container::run() {
  cv::Mat result;
  for (auto s : container) {
    result = s->run();
  }
  return result;
}