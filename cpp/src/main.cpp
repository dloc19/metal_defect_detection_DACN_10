#include "mdproc.hpp"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

void printDefectInfo(const DefectInfo& info) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "File: " << info.filename << "\n";
    std::cout << "  Edge Density: " << info.edge_density << "\n";
    std::cout << "  Mean Intensity: " << info.mean_intensity << "\n";
    std::cout << "  Std Intensity: " << info.std_intensity << "\n";
    std::cout << "  Contrast Ratio: " << info.contrast_ratio << "\n";
    std::cout << "  Defect Count: " << info.defect_count << "\n";
    std::cout << "  Processing Time: " << info.processing_time_ms << " ms\n";
    
    if (!info.defect_regions.empty()) {
        std::cout << "  Defect Regions:\n";
        for (size_t i = 0; i < info.defect_regions.size(); ++i) {
            const auto& rect = info.defect_regions[i];
            std::cout << "    [" << i << "] (" << rect.x << ", " << rect.y 
                      << ", " << rect.width << ", " << rect.height << ")\n";
        }
    }
    std::cout << "\n";
}

void demonstrateAdvancedFeatures() {
    std::cout << "=== Advanced Metal Defect Processing Demo ===\n\n";
    
    // Create processor with custom parameters
    MetalDefectProcessor processor;
    
    // Configure preprocessing
    PreprocessParams preprocess_params;
    preprocess_params.target_width = 512;
    preprocess_params.target_height = 512;
    preprocess_params.apply_clahe = true;
    preprocess_params.clahe_clip_limit = 3.0;
    preprocess_params.apply_gaussian_blur = true;
    preprocess_params.blur_kernel_size = 3;
    processor.setPreprocessParams(preprocess_params);
    
    // Configure detection
    DetectionParams detection_params;
    detection_params.edge_threshold_low = 30.0;
    detection_params.edge_threshold_high = 100.0;
    detection_params.min_defect_area = 50;
    detection_params.max_defect_area = 5000;
    detection_params.use_morphology = true;
    processor.setDetectionParams(detection_params);
    
    // Enable optimizations
    processor.enableOptimizations(true);
    processor.setNumThreads(4);
    
    std::cout << "Processor configured with optimizations enabled.\n";
    std::cout << "Threads: " << std::thread::hardware_concurrency() << "\n\n";
    
    // Create a test image for benchmarking
    cv::Mat test_image = cv::Mat::zeros(512, 512, CV_8UC1);
    cv::rectangle(test_image, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(255), -1);
    cv::rectangle(test_image, cv::Point(300, 300), cv::Point(400, 400), cv::Scalar(128), -1);
    
    // Benchmark processing
    std::cout << "Benchmarking processing speed...\n";
    double avg_time = processor.benchmarkProcessing(test_image, 100);
    std::cout << "Average processing time: " << avg_time << " ms\n\n";
    
    // Process test image
    std::cout << "Processing test image...\n";
    DefectInfo test_result = processor.processImage(test_image, "test_image");
    printDefectInfo(test_result);
}

int main(int argc, char* argv[]) {
    std::cout << "Metal Defect Detection - C++ Optimized Version\n";
    std::cout << "==============================================\n\n";
    
    // Check command line arguments
    std::string image_folder = "../data/images";
    if (argc > 1) {
        image_folder = argv[1];
    }
    
    std::cout << "Processing images from: " << image_folder << "\n\n";
    
    try {
        // Demonstrate advanced features
        demonstrateAdvancedFeatures();
        
        // Process folder
        std::cout << "=== Processing Image Folder ===\n";
        auto results = analyze_folder(image_folder);
        
        if (results.empty()) {
            std::cout << "No images found in folder: " << image_folder << "\n";
            std::cout << "Supported formats: .jpg, .jpeg, .png, .bmp, .tiff\n";
            return 1;
        }
        
        std::cout << "Found " << results.size() << " images.\n\n";
        
        // Print results
        for (const auto& result : results) {
            printDefectInfo(result);
        }
        
        // Summary statistics
        double total_time = 0;
        int total_defects = 0;
        for (const auto& result : results) {
            total_time += result.processing_time_ms;
            total_defects += result.defect_count;
        }
        
        std::cout << "=== Summary ===\n";
        std::cout << "Total images processed: " << results.size() << "\n";
        std::cout << "Total processing time: " << total_time << " ms\n";
        std::cout << "Average time per image: " << total_time / results.size() << " ms\n";
        std::cout << "Total defects found: " << total_defects << "\n";
        std::cout << "Average defects per image: " << static_cast<double>(total_defects) / results.size() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
