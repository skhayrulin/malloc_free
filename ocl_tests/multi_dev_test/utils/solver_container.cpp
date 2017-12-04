#include "solver_container.h"
#include "error.h"
#include "ocl_helper.h"
#include <thread>
void run_thread(cv::Mat, std::shared_ptr<ocl_solver> &);
solver_container::solver_container(size_t count, cv::Mat img) {
  real_size = {0, 0, img.rows, img.cols};
  std::priority_queue<std::shared_ptr<device>> dev_q = get_dev_queue();
  int i = 0;
  int cols = (dev_q.size() > 1) ? img.cols >> 1 : img.cols;
  while (!dev_q.empty()) {
    std::vector<int> lims = {0, i * cols, img.rows, cols};
    ++i;
    std::shared_ptr<ocl_solver> solver(new ocl_solver(img, lims, dev_q.top()));
    dev_q.pop();
    container.push_back(solver);
  }
}

cv::Mat solver_container::run() {
  cv::Mat result(real_size[2], real_size[3], CV_32FC4);
  std::vector<std::thread> t_pool;
  std::for_each(
      container.begin(), container.end(), [&](std::shared_ptr<ocl_solver> &s) {
        t_pool.push_back(std::thread(run_thread, result, std::ref(s)));
      });
  std::for_each(t_pool.begin(), t_pool.end(), [](std::thread &t) { t.join(); });
  return result;
}

void run_thread(cv::Mat result, std::shared_ptr<ocl_solver> &s) {
  cv::Mat tmp = s->run();
  auto c = s->get_c();
  for (int i = 0; i < c.rows; ++i) {
    for (int j = 0; j < c.cols; ++j) {
      cv::Vec4f &v = result.at<cv::Vec4f>(c.start_x + i, c.start_y + j);
      cv::Vec4f &v1 = tmp.at<cv::Vec4f>(i, j);
      v = v1;
    }
  }
}