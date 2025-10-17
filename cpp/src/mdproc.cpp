#include "mdproc.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<DefectInfo> analyze_folder(const std::string& folder) {
    std::vector<DefectInfo> result;

    for (auto& p : fs::directory_iterator(folder)) {
        if (p.is_regular_file()) {
            cv::Mat img = cv::imread(p.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            cv::Mat edges;
            cv::Canny(img, edges, 50, 150);
            double edge_density = (double)cv::countNonZero(edges) / (img.rows * img.cols);
            cv::Scalar mean_val = cv::mean(img);

            result.push_back({ p.path().filename().string(), edge_density, mean_val[0] });
        }
    }
    return result;
}
