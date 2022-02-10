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
In this section, we introduce how to add new benchmarks taking `benchmark/test_binary_classification.py` as an example.

### Benchmark unit
This benchmarks uses [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/stable/) for benchmark suite.
This library treats one function as a unit of one benchmark run, so you have to wrap a process to measure performance with a function.

### Write benchmark
All you have to do is
1. Create a function doing stuff to measure performance
2. Call the function through `benchmark` in a `test_*` function in `test_*` file

#### Create a function
Create a file like `benchmark/binary_classification_skqulacs.py`.
And define a function `binary_classification_skqulacs` and write a process to measure performance.

#### Call the function from test file
First, create a test file whose name starts with `test_`.
__NOTICE__: Your benchmark will be never executed if you don't follow this rule.
Here we created `benchmark/test_binary_classification.py`.

In this file, there are two functions `test_skqulacs` and `test_pennylane`.
__NOTICE__: You have to name test functions with `test_` to executed by test suite.
Take a look at `test_skqulacs`, it takes `benchmark` as an argument.
This is supplied by test suite.
You can configure the benchmark through this variable.
For farther information, follow the documentation.

this function calls `benchmark.pedantic(binary_classification_skqulacs, args=[n_qubit], rounds=10)`.
`binary_classification_skqulacs` is a function in `benchmark/binary_classification_skqulacs.py`.
Test suite runs this function 10 times, which is specified with `rounds`, and pass `n_qubit` to the function as an argument.

## Run benchmarks
To run benchmarks, run following command at the project root.
```bash
make benchmarks
# Run specific file(s):
python -m pytest benchmark/test_binary_classification.py --benchmark-autosave
```
Then benchmark statistics are displayed and you will find a JSON file at `.benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json`.
This is the summary of the benchmark runs.

## Plot
### Generate plotted images
You can plot the result by `benchmarks/plot_result.py` with the generated JSON file.
```bash
python benchmarks/plot_result.py .benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json
```
The rendered image is exported at `.benchmarks/outputs`.
You can specify the directory wherever you like by `-o`(`--output_dir`) option.
```bash
python benchmarks/plot_result.py .benchmarks/<SYSTEM_SPECIFIC_DIR>/*.json -o doc/source/figures
```

### Write a plot
In this section, we introduce how to add a new plot.

This is a structure of the generated JSON file:
```json
{
    ...
    "benchmarks": [
        {
            "group": null,
            "name": "test_skqulacs",
            "fullname": "benchmarks/test_binary_classification.py::test_skqulacs",
            ...
        },
        {
            "group": null,
            "name": "test_pennylane",
            "fullname": "benchmarks/test_binary_classification.py::test_pennylane",
            ...
        },
        ...
    ]
}
```

To create a new plot, you have to do following things:
1. Create a function to plot the data
2. Extract data from the JSON file(support functions available)
3. Plot with `matplotlib`

#### Create function
Create a function like `plot_binary_classification`, which extracts data and plot them.
And the function should have `all_results` and `output_dir` as auguments.
`all_results` is a list of results; this is a value for `"benchmarks"` key in the JSON file above.
`output_dir` is a directory name to export plotted images.

#### Extract data
To extract data, there are two useful functions: `extract_results_in_file` and `extract_result_for_one_test`.

`extract_results_in_file` picks up benchmark results in one test file.
For example, `extract_results_in_file(all_results, "test_binary_classification.py")` extracts benchmark results which is defined at `benchmarks/test_binary_classification.py`.

`extract_result_for_one_test` picks up benchmark results for a function.
For example, `extract_result_for_one_test(results, "test_skqulacs")` extracts results of `test_skqulacs` function.

#### Plot
Then you get all the data to plot, just call `matplotlib` and save the plot as images with `plt.savefig`.
