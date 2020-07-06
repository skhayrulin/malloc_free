#include "util/ocl_helper.h"
#include "util/ocl_radix_sort.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
int main(int argc, char** argv)
{
    std::vector<int> data(512, 0);
    for (int i = 0; i < data.size(); ++i) {
        int i_secret = rand() % 10;
        data[i] = i_secret;
    }
    auto dev_q = get_dev_queue();
    try {
        ocl_radix_sort_solver<int> solver(&data, dev_q.top(), 0);
        solver.sort();
        for (int i = 0; i < data.size(); ++i) {
            cout << data[i] << endl;
        }
    } catch (std::string& s) {
        cout << s << endl;
    }
    cout << "Num of dev is " << dev_q.size() << endl;
    return 0;
}