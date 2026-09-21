"""
Export YOLOv8 PyTorch model (.pt) to ONNX format.

Usage:
    python scripts/export_onnx.py [--model path/to/best.pt] [--output path/to/best.onnx]
"""

import argparse
import sys
from pathlib import Path


def export_to_onnx(model_path: str, output_path: str, img_size: int = 640):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed.")
        print("Install it: pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading PyTorch model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Exporting to ONNX (imgsz={img_size})...")
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        simplify=True,
        dynamic=False,
        opset=12,
    )

    # Move to desired output path
    if output_path and export_path != output_path:
        import shutil
        shutil.move(export_path, output_path)
        print(f"ONNX model saved to: {output_path}")
    else:
        print(f"ONNX model saved to: {export_path}")

    # Validate
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path or export_path)
        print(f"Validation: ONNX model loads successfully")
        print(f"  Input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
        print(f"  Outputs: {len(session.get_outputs())}")
        for o in session.get_outputs():
            print(f"    {o.name}: {o.shape}")
    except ImportError:
        print("Warning: onnxruntime not installed, skipping validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--model", default="backend/models/best.pt", help="Path to .pt model")
    parser.add_argument("--output", default="backend/models/best.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.imgsz)
