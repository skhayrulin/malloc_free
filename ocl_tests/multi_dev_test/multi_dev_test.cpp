#include "utils/ocl_solver.h"
#include "utils/solver_container.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file
  // cvtColor(image, image, CV_BGRA2BGR); // 1. change the number of channels
  image.convertTo(image, CV_32FC4,
                  1 / 255.0); // 2. change type to float and scale
  if (!image.data)            // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }
  solver_container &s_c = solver_container::instance(1, image);
  Mat ret_img = s_c.run();
  namedWindow("Display window",
              WINDOW_AUTOSIZE);      // Create a window for display.
  imshow("Display window", ret_img); // Show our image inside it.
  waitKey(0);
  return 0;
}