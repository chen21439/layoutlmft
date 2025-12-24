#!/usr/bin/env python
"""Parallel test runner for Detect-Order-Construct modules

Usage:
    python run_parallel.py                    # Run all tests
    python run_parallel.py --group models     # Run only model tests
    python run_parallel.py --group metrics    # Run only metrics tests
    python run_parallel.py --env test         # Run on server (GPU1)
"""
import subprocess
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Tests grouped by DOC modules
TEST_GROUPS = {
    # Models (4.2, 4.3)
    "models": [
        "test_432_transformer.py",      # 4.3.2 Transformer
        "test_433_order_head.py",       # 4.3.3 Order Head
        "test_434_relation_head.py",    # 4.3.4 Relation Head
        "test_43_order_module.py",      # 4.3 Complete Order
        "test_doc_pipeline.py",         # 4.2 + 4.3 Pipeline
    ],
    # Metrics (Detect, Order, Construct)
    "metrics": [
        "test_metrics_detect.py",       # Detect: Classification, Intra-order
        "test_metrics_order.py",        # Order: Reading order TEDS
        "test_metrics_construct.py",    # Construct: Tree TEDS
    ],
}

ALL_TESTS = TEST_GROUPS["models"] + TEST_GROUPS["metrics"]


def run_test(test_file, env=None):
    """Run a single test file"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, test_file)

    cmd = [sys.executable, test_path]
    if env:
        cmd.extend(["--env", env])

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=test_dir,
        env={**os.environ, "COMP_HRDOC_ENV": env or "dev"},
    )
    elapsed = time.time() - start

    return {
        "file": test_file,
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel test runner")
    parser.add_argument("--group", choices=["all", "models", "metrics"], default="all")
    parser.add_argument("--env", type=str, default=None, help="Environment (dev/test)")
    parser.add_argument("--workers", type=int, default=6, help="Max parallel workers")
    args = parser.parse_args()

    # Select tests
    if args.group == "all":
        tests = ALL_TESTS
    else:
        tests = TEST_GROUPS[args.group]

    print("=" * 60)
    print(f"Detect-Order-Construct Tests ({args.group})")
    print(f"Environment: {args.env or 'dev (default)'}")
    print(f"Tests: {len(tests)}")
    print("=" * 60)

    start_all = time.time()
    results = []

    # Run tests in parallel
    max_workers = min(args.workers, len(tests))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_test, t, args.env): t for t in tests}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "[PASS]" if result["success"] else "[FAIL]"
            print(f"{status} {result['file']} ({result['time']:.2f}s)")
            if not result["success"]:
                stderr = result["stderr"][:300] if result["stderr"] else ""
                if stderr:
                    print(f"  Error: {stderr}")

    total_time = time.time() - start_all
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed ({total_time:.2f}s)")

    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
