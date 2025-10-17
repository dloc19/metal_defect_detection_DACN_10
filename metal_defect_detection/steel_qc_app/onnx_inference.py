"""
ONNX Runtime inference module for faster real-time processing.
Provides optimized inference with GPU acceleration when available.
"""
import numpy as np
import cv2
from PIL import Image
import json
import os
from typing import Tuple, Optional, Dict, Any

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not available. Install with: pip install onnxruntime onnxruntime-gpu")

class ONNXInference:
    """Optimized ONNX inference for MobileNetV2 defect detection."""
    
    def __init__(self, onnx_path: str, map_path: str, img_size: int = 224, 
                 use_gpu: bool = True, alpha_thr: float = 0.6):
        """
        Initialize ONNX inference.
        
        Args:
            onnx_path: Path to ONNX model file
            map_path: Path to idx_to_class.json mapping
            img_size: Input image size
            use_gpu: Whether to use GPU acceleration
            alpha_thr: Confidence threshold for defect detection
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not available")
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Mapping file not found: {map_path}")
        
        self.img_size = img_size
        self.alpha_thr = alpha_thr
        
        # Load class mapping
        with open(map_path, "r", encoding="utf-8") as f:
            self.idx_to_class = json.load(f)
        
        # Setup ONNX Runtime providers
        providers = []
        if use_gpu:
            # Try GPU providers in order of preference
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                print("✅ Using CUDA GPU acceleration")
            elif 'DmlExecutionProvider' in ort.get_available_providers():
                providers.append('DmlExecutionProvider')
                print("✅ Using DirectML GPU acceleration")
        
        # Fallback to CPU
        providers.append('CPUExecutionProvider')
        
        # Create inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ ONNX model loaded: {onnx_path}")
        print(f"Input: {self.input_name}, Output: {self.output_name}")
        print(f"Providers: {self.session.get_providers()}")
        
        # Pre-compute normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized preprocessing for ONNX inference.
        
        Args:
            image: BGR image array
            
        Returns:
            Preprocessed image tensor ready for ONNX inference
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        
        # Transpose to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on image.
        
        Args:
            image: BGR image array
            
        Returns:
            Tuple of (class_name, confidence, probabilities)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Get prediction
        cls_idx = int(np.argmax(probs))
        cls_name = self.get_class_name(cls_idx)
        confidence = float(probs[cls_idx])
        
        return cls_name, confidence, probs
    
    def get_class_name(self, cls_idx: int) -> str:
        """Get class name from index."""
        try:
            return self.idx_to_class[cls_idx]
        except KeyError:
            return self.idx_to_class[str(cls_idx)]
    
    def is_defect(self, confidence: float) -> bool:
        """Check if confidence exceeds defect threshold."""
        return confidence >= self.alpha_thr
    
    def batch_predict(self, images: list) -> list:
        """
        Batch inference for multiple images.
        
        Args:
            images: List of BGR image arrays
            
        Returns:
            List of (class_name, confidence, probabilities) tuples
        """
        if not images:
            return []
        
        # Preprocess all images
        input_tensors = [self.preprocess(img) for img in images]
        batch_tensor = np.concatenate(input_tensors, axis=0)
        
        # Run batch inference
        outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})
        logits = outputs[0]
        
        results = []
        for i in range(len(images)):
            # Apply softmax
            exp_logits = np.exp(logits[i] - np.max(logits[i]))
            probs = exp_logits / np.sum(exp_logits)
            
            # Get prediction
            cls_idx = int(np.argmax(probs))
            cls_name = self.get_class_name(cls_idx)
            confidence = float(probs[cls_idx])
            
            results.append((cls_name, confidence, probs))
        
        return results

def create_onnx_inference(onnx_path: str, map_path: str, **kwargs) -> Optional[ONNXInference]:
    """
    Factory function to create ONNX inference instance.
    
    Args:
        onnx_path: Path to ONNX model
        map_path: Path to class mapping
        **kwargs: Additional arguments for ONNXInference
        
    Returns:
        ONNXInference instance or None if not available
    """
    if not ONNX_AVAILABLE:
        print("❌ ONNX Runtime not available")
        return None
    
    try:
        return ONNXInference(onnx_path, map_path, **kwargs)
    except Exception as e:
        print(f"❌ Failed to create ONNX inference: {e}")
        return None

# Performance comparison utilities
def benchmark_inference(onnx_inference: ONNXInference, test_image: np.ndarray, 
                       num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark ONNX inference performance.
    
    Args:
        onnx_inference: ONNX inference instance
        test_image: Test image for benchmarking
        num_runs: Number of runs for timing
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Warmup
    for _ in range(10):
        onnx_inference.predict(test_image)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        onnx_inference.predict(test_image)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'fps': 1.0 / np.mean(times)
    }
