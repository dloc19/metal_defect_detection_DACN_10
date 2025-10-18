#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

// Defect detection result structure
struct DefectInfo {
    std::string filename;
    double edge_density;
    double mean_intensity;
    double std_intensity;
    double contrast_ratio;
    int defect_count;
    std::vector<cv::Rect> defect_regions;
    double processing_time_ms;
};

// Image preprocessing parameters
struct PreprocessParams {
    int target_width = 224;
    int target_height = 224;
    bool apply_gaussian_blur = true;
    int blur_kernel_size = 5;
    bool apply_histogram_eq = false;
    bool apply_clahe = true;
    double clahe_clip_limit = 2.0;
    int clahe_tile_size = 8;
};

// Defect detection parameters
struct DetectionParams {
    double edge_threshold_low = 50.0;
    double edge_threshold_high = 150.0;
    int min_defect_area = 80;
    int max_defect_area = 10000;
    double contrast_threshold = 0.1;
    bool use_morphology = true;
    int morphology_kernel_size = 3;
};

// Main processing class
class MetalDefectProcessor {
public:
    MetalDefectProcessor();
    ~MetalDefectProcessor() = default;
    
    // Configuration
    void setPreprocessParams(const PreprocessParams& params);
    void setDetectionParams(const DetectionParams& params);
    
    // Single image processing
    DefectInfo processImage(const cv::Mat& image, const std::string& filename = "");
    DefectInfo processImageFile(const std::string& image_path);
    
    // Batch processing
    std::vector<DefectInfo> analyzeFolder(const std::string& folder_path, bool parallel = true);
    std::vector<DefectInfo> processImages(const std::vector<cv::Mat>& images, 
                                        const std::vector<std::string>& filenames = {});
    
    // Performance utilities
    void enableOptimizations(bool enable = true);
    void setNumThreads(int num_threads);
    double benchmarkProcessing(const cv::Mat& test_image, int num_runs = 100);
    
    // Utility functions
    static cv::Mat preprocessImage(const cv::Mat& input, const PreprocessParams& params);
    static std::vector<cv::Rect> detectDefects(const cv::Mat& image, const DetectionParams& params);
    static double calculateEdgeDensity(const cv::Mat& image, double low_thresh = 50, double high_thresh = 150);
    static double calculateContrastRatio(const cv::Mat& image);

private:
    PreprocessParams preprocess_params_;
    DetectionParams detection_params_;
    bool optimizations_enabled_;
    int num_threads_;
    
    // Internal processing methods
    DefectInfo analyzeImageInternal(const cv::Mat& image, const std::string& filename);
    cv::Mat applyPreprocessing(const cv::Mat& image);
    std::vector<cv::Rect> applyDefectDetection(const cv::Mat& image);
    void optimizeImage(cv::Mat& image);
};

// Standalone utility functions
std::vector<DefectInfo> analyze_folder(const std::string& folder);
cv::Mat enhanceImage(const cv::Mat& input);
std::vector<cv::Rect> findDefectRegions(const cv::Mat& image);
double measureImageQuality(const cv::Mat& image);
