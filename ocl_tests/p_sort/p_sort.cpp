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
    for (size_t i = 0; i < data.size(); ++i) {
        int i_secret = rand() % 10;
        data[i] = i_secret;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        cout << data[i] << ",";
    }
    cout << endl;
    cout << "===============" << endl;
    auto dev_q = get_dev_queue();
    cout << "Num of dev is " << dev_q.size() << endl;
    try {
        ocl_radix_sort_solver<int> solver(data, dev_q.top());
        solver.sort();
        for (size_t i = 0; i < data.size(); ++i) {
            cout << data[i] << ",";
        }
        cout << endl;
    } catch (std::string& s) {
        cout << s << endl;
    }
    return 0;
}