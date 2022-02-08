# Benchmarks

## Running benchmarks
To run benchmarks, run following command at the project root.
```bash
make benchmarks
```
Then benchmark statistics are displayed and you will find a JSON file at `.benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json`.
This is the summary of the benchmark runs.

You can plot the result by `benchmarks/plot_result.py` with the generated JSON file.
```bash
python benchmarks/plot_result.py .benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json
```
The rendered image is exported at `.benchmarks/outputs`.
You can specify the directory wherever you like by `-o`(`--output_dir`) option.
```bash
python benchmarks/plot_result.py .benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json -o doc/source/figures
```

## Add benchmarks
In this section, we introduce how to add new benchmarks taking `test_binary_classification.py` as an example.

### Write a function to measure
This benchmarks uses [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/stable/) for benchmark suite.
This library 