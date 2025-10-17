#pragma once
#include <string>
#include <vector>

struct DefectInfo {
    std::string filename;
    double edge_density;
    double mean_intensity;
};

std::vector<DefectInfo> analyze_folder(const std::string& folder);
