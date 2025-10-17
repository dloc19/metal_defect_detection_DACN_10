#include "mdproc.hpp"
#include <iostream>

int main() {
    auto results = analyze_folder("../data/images"); // thư mục ảnh test
    for (auto& r : results) {
        std::cout << r.filename << " - Edge density: " << r.edge_density
                  << " - Mean: " << r.mean_intensity << "\n";
    }
    return 0;
}
