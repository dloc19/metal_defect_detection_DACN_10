#include "mdproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// MetalDefectProcessor Implementation
MetalDefectProcessor::MetalDefectProcessor() 
    : optimizations_enabled_(true), num_threads_(std::thread::hardware_concurrency()) {
    // Set OpenCV to use optimized implementations
    cv::setUseOptimized(true);
    cv::setNumThreads(num_threads_);
}

void MetalDefectProcessor::setPreprocessParams(const PreprocessParams& params) {
    preprocess_params_ = params;
}

void MetalDefectProcessor::setDetectionParams(const DetectionParams& params) {
    detection_params_ = params;
}

void MetalDefectProcessor::enableOptimizations(bool enable) {
    optimizations_enabled_ = enable;
    cv::setUseOptimized(enable);
}

void MetalDefectProcessor::setNumThreads(int num_threads) {
    num_threads_ = std::max(1, num_threads);
    cv::setNumThreads(num_threads_);
}

DefectInfo MetalDefectProcessor::processImage(const cv::Mat& image, const std::string& filename) {
    return analyzeImageInternal(image, filename);
}

DefectInfo MetalDefectProcessor::processImageFile(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return DefectInfo{fs::path(image_path).filename().string(), 0, 0, 0, 0, 0, {}, 0};
    }
    return analyzeImageInternal(image, fs::path(image_path).filename().string());
}

std::vector<DefectInfo> MetalDefectProcessor::analyzeFolder(const std::string& folder_path, bool parallel) {
    std::vector<DefectInfo> results;
    std::vector<std::string> image_files;
    
    // Collect all image files
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    
    if (parallel && image_files.size() > 1) {
        // Parallel processing
        std::vector<std::future<DefectInfo>> futures;
        for (const auto& file : image_files) {
            futures.push_back(std::async(std::launch::async, [this, file]() {
                return processImageFile(file);
            }));
        }
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
    } else {
        // Sequential processing
        for (const auto& file : image_files) {
            results.push_back(processImageFile(file));
        }
    }
    
    return results;
}

std::vector<DefectInfo> MetalDefectProcessor::processImages(const std::vector<cv::Mat>& images, 
                                                           const std::vector<std::string>& filenames) {
    std::vector<DefectInfo> results;
    results.reserve(images.size());
    
    for (size_t i = 0; i < images.size(); ++i) {
        std::string filename = (i < filenames.size()) ? filenames[i] : "image_" + std::to_string(i);
        results.push_back(analyzeImageInternal(images[i], filename));
    }
    
    return results;
}

double MetalDefectProcessor::benchmarkProcessing(const cv::Mat& test_image, int num_runs) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        analyzeImageInternal(test_image, "benchmark");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / (1000.0 * num_runs); // Return average time in ms
}

DefectInfo MetalDefectProcessor::analyzeImageInternal(const cv::Mat& image, const std::string& filename) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DefectInfo result;
    result.filename = filename;
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply preprocessing
    cv::Mat processed = applyPreprocessing(gray);
    
    // Calculate basic statistics
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(processed, mean_val, std_val);
    result.mean_intensity = mean_val[0];
    result.std_intensity = std_val[0];
    
    // Calculate edge density
    result.edge_density = calculateEdgeDensity(processed, 
        detection_params_.edge_threshold_low, 
        detection_params_.edge_threshold_high);
    
    // Calculate contrast ratio
    result.contrast_ratio = calculateContrastRatio(processed);
    
    // Detect defects
    result.defect_regions = applyDefectDetection(processed);
    result.defect_count = static_cast<int>(result.defect_regions.size());
    
    // Calculate processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0;
    
    return result;
}

cv::Mat MetalDefectProcessor::applyPreprocessing(const cv::Mat& image) {
    cv::Mat result = image.clone();
    
    // Resize if needed
    if (result.rows != preprocess_params_.target_height || 
        result.cols != preprocess_params_.target_width) {
        cv::resize(result, result, 
            cv::Size(preprocess_params_.target_width, preprocess_params_.target_height),
            cv::INTER_LANCZOS4);
    }
    
    // Apply Gaussian blur
    if (preprocess_params_.apply_gaussian_blur) {
        cv::GaussianBlur(result, result, 
            cv::Size(preprocess_params_.blur_kernel_size, preprocess_params_.blur_kernel_size), 0);
    }
    
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if (preprocess_params_.apply_clahe) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            preprocess_params_.clahe_clip_limit,
            cv::Size(preprocess_params_.clahe_tile_size, preprocess_params_.clahe_tile_size));
        clahe->apply(result, result);
    }
    
    // Apply histogram equalization
    if (preprocess_params_.apply_histogram_eq) {
        cv::equalizeHist(result, result);
    }
    
    return result;
}

std::vector<cv::Rect> MetalDefectProcessor::applyDefectDetection(const cv::Mat& image) {
    return detectDefects(image, detection_params_);
}

// Static utility functions
cv::Mat MetalDefectProcessor::preprocessImage(const cv::Mat& input, const PreprocessParams& params) {
    MetalDefectProcessor processor;
    processor.setPreprocessParams(params);
    return processor.applyPreprocessing(input);
}

std::vector<cv::Rect> MetalDefectProcessor::detectDefects(const cv::Mat& image, const DetectionParams& params) {
    std::vector<cv::Rect> defects;
    
    // Edge detection
    cv::Mat edges;
    cv::Canny(image, edges, params.edge_threshold_low, params.edge_threshold_high);
    
    // Morphological operations
    if (params.use_morphology) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
            cv::Size(params.morphology_kernel_size, params.morphology_kernel_size));
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(edges, edges, cv::MORPH_OPEN, kernel);
    }
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter contours by area
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= params.min_defect_area && area <= params.max_defect_area) {
            cv::Rect bounding_rect = cv::boundingRect(contour);
            defects.push_back(bounding_rect);
        }
    }
    
    return defects;
}

double MetalDefectProcessor::calculateEdgeDensity(const cv::Mat& image, double low_thresh, double high_thresh) {
    cv::Mat edges;
    cv::Canny(image, edges, low_thresh, high_thresh);
    return static_cast<double>(cv::countNonZero(edges)) / (image.rows * image.cols);
}

double MetalDefectProcessor::calculateContrastRatio(const cv::Mat& image) {
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(image, mean_val, std_val);
    return std_val[0] / (mean_val[0] + 1e-6); // Avoid division by zero
}

// Standalone utility functions
std::vector<DefectInfo> analyze_folder(const std::string& folder) {
    MetalDefectProcessor processor;
    return processor.analyzeFolder(folder, true);
}

cv::Mat enhanceImage(const cv::Mat& input) {
    PreprocessParams params;
    params.apply_clahe = true;
    params.apply_gaussian_blur = true;
    return MetalDefectProcessor::preprocessImage(input, params);
}

std::vector<cv::Rect> findDefectRegions(const cv::Mat& image) {
    DetectionParams params;
    return MetalDefectProcessor::detectDefects(image, params);
}

double measureImageQuality(const cv::Mat& image) {
    MetalDefectProcessor processor;
    DefectInfo info = processor.processImage(image);
    return info.contrast_ratio * (1.0 - info.edge_density); // Higher is better
}
