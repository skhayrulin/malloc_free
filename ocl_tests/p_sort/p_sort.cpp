#include "util/ocl_helper.h"
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
int main(int argc, char** argv){
    auto dev_q = get_dev_queue();
    cout << "Num of dev is " << dev_q.size() << endl;
    return 0;
}