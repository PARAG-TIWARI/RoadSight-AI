"""
Benchmark inference performance.

Measures latency, FPS, CPU, and memory for available backends.
Outputs results as JSON to stdout.

Usage:
    python scripts/benchmark.py [--iterations 20] [--output results.json]
"""

import argparse
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark RoadSight AI inference")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--format", type=str, default="all", choices=["all", "onnx", "pytorch"], help="Backend format to benchmark (all, onnx, pytorch)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    print(f"Running benchmark ({args.iterations} iterations, {args.warmup} warmup, format: {args.format})...")
    result = run_benchmark(iterations=args.iterations, warmup=args.warmup)

    if args.format != "all":
        target = "ONNX Runtime" if args.format == "onnx" else "PyTorch"
        result.results = [r for r in result.results if target.lower() in r.backend.lower()]

    output = result.model_dump()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to: {args.output}")
    else:
        print(json.dumps(output, indent=2))

    # Summary
    print("\n--- Summary ---")
    for r in result.results:
        print(f"  {r.backend}: {r.avg_latency_ms:.1f}ms avg, {r.fps:.1f} FPS, {r.memory_mb:.0f}MB RAM")


if __name__ == "__main__":
    main()
