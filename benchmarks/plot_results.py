import argparse
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt


Json = Dict[str, object]


def extract_results_in_file(results: List[Json], file_name: str) -> List[Json]:
    """Extract records of the benchmark result in a specific file.

    Args:
        results: List of benchmark records whose key is "benchmarks" in the exported JSON file.
        file_name: A test file name to extract its results.
    """
    return list(filter(lambda x: file_name in x["fullname"], results))


def extract_result_for_one_test(results: List[Json], test_name: str) -> Json:
    """Extract a record of the benchmark result corresponding to 1 test function.

    Args:
        results: List of benchmark records.
        test_name: The name of the test function to extract the result.
    """
    return list(filter(lambda x: test_name in x["name"], results))[0]


def plot_binary_classification(all_results: List[Json], output_dir: str) -> None:
    results = extract_results_in_file(all_results, "test_binary_classification.py")
    skqulacs_result = extract_result_for_one_test(results, "test_skqulacs")
    pennylane_result = extract_result_for_one_test(results, "pennylane")
    means = [skqulacs_result["stats"]["mean"], pennylane_result["stats"]["mean"]]
    plt.bar(["skqulacs", "pennylane"], means)
    plt.savefig(f"{output_dir}/binary_classification.png")


def plot_results(output_dir: str, file_path: str) -> None:
    with open(file_path, 'r') as f:
        benchmark_result = json.load(f)["benchmarks"]

    plot_binary_classification(benchmark_result, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results exported by pytest-benchmark")
    parser.add_argument("file_path", type=str, help="Path to JSON file containing benchmark results")
    parser.add_argument("-o", "--output_dir", type=str, default="./.benchmarks/outputs", help="Path to directory to output plot images")
    args = parser.parse_args()

    file_path = args.file_path
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    plot_results(output_dir, file_path)


if __name__ == "__main__":
    main()
