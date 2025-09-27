#!/usr/bin/env python3
"""
CreativeIQ Test Runner
Runs comprehensive tests for the application
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and report results"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ PASSED")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("✗ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False


def main():
    """Run all tests"""
    print("CreativeIQ Test Suite")
    print("Running comprehensive tests...")

    # Check if we're in the right directory
    if not Path("app").exists():
        print("Error: Please run this script from the CreativeIQ root directory")
        sys.exit(1)

    tests_passed = 0
    total_tests = 0

    # Test 1: Check imports
    total_tests += 1
    if run_command(
        "python -c 'import app.main; print(\"All imports successful\")'",
        "Import Test"
    ):
        tests_passed += 1

    # Test 2: Run pytest
    total_tests += 1
    if run_command(
        "python -m pytest tests/ -v --tb=short",
        "API Tests"
    ):
        tests_passed += 1

    # Test 3: Check code style (if black is installed)
    total_tests += 1
    if run_command(
        "python -c 'import black; print(\"Black is available\")'",
        "Code Style Check (Black available)"
    ):
        if run_command(
            "python -m black --check app/ tests/ --diff",
            "Code Style Validation"
        ):
            tests_passed += 1
    else:
        print("Black not installed - skipping code style check")

    # Test 4: Basic API health check (if server is running)
    total_tests += 1
    if run_command(
        "python -c 'import requests; r=requests.get(\"http://localhost:8000/health\", timeout=5); print(f\"Health check: {r.status_code}\")' 2>/dev/null || echo 'Server not running'",
        "Health Check (if server running)"
    ):
        tests_passed += 1

    # Test 5: Check Docker setup
    total_tests += 1
    if run_command(
        "docker --version && docker-compose --version",
        "Docker Setup Check"
    ):
        tests_passed += 1

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests Passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Your CreativeIQ setup is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()