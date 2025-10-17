#!/usr/bin/env python3
"""
Export PyTorch MobileNetV2 model to ONNX format for faster inference.
Usage: python export_onnx.py --ckpt mobilenetv2_best.pth --output mobilenetv2_best.onnx
"""
import argparse
import torch
import torch.onnx
from torchvision import transforms
import json
import os
from model import build_model

def export_to_onnx(ckpt_path, map_path, output_path, img_size=224, batch_size=1):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        ckpt_path: Path to PyTorch checkpoint
        map_path: Path to idx_to_class.json mapping
        output_path: Output ONNX file path
        img_size: Input image size
        batch_size: Batch size for export
    """
    print(f"Loading model from {ckpt_path}...")
    
    # Load class mapping
    with open(map_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    
    # Build and load model
    model = build_model(num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    
    # Clean state dict (remove module. prefix if present)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    
    model.load_state_dict(new_state, strict=False)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Exporting to ONNX format: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model exported successfully to {output_path}")
    print(f"Input shape: {batch_size}x3x{img_size}x{img_size}")
    print(f"Output shape: {batch_size}x{num_classes}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
    except ImportError:
        print("⚠️  onnx package not available for verification")
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--ckpt", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--map", required=True, help="Path to idx_to_class.json")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint file not found: {args.ckpt}")
        return
    
    if not os.path.exists(args.map):
        print(f"❌ Mapping file not found: {args.map}")
        return
    
    export_to_onnx(args.ckpt, args.map, args.output, args.img_size, args.batch_size)

if __name__ == "__main__":
    main()
