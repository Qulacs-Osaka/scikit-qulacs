# Contributing scikit-qulacs

## Requirements
- poetry([installation](https://python-poetry.org/docs/#installation))

## Start coding
Set up the project.
1. First, clone this repository.
```bash
git clone git@github.com:Qulacs-Osaka/scikit-qulacs.git
cd scikit-qulacs
```

2. Install dependencies and development tools.
```bash
poetry install
```

Next, workflow through modification to merge.
3. Synchronize with `main`(not necessary for the first time).
```bash
git switch main
git pull # Shorthand for `git pull origin main`
```

4. Create branch with name describing what you are going to develop.
```bash
git switch -c 99-wonderful-model
```

5. Format, lint and test code before commit.
```bash
make check
make test
```

Code format and some lint errors can be fixed by `make fix`.
Rest of lint errors should be fixed by hand along error messages.

6. Commit and push modified files.
```bash
git add MODIFIED_FILE
git commit
# For the first push in the branch
git push -u origin 99-wonderful-model
# After first push
git push
```

7. Create a pull request(PR) after you finish the development at the branch. Basically you need someone to review your code. If modification is subtle, you might not need a review.

## Testing
Write tests when you develop a new feature. Tests are executed automatically.

1. Create `test_*.py` in `tests` directory. Describe what to test in the file name.
2. Create a function whose name starts with `test_`. Write assertion to check if a function you developed is compliant with a specification. For example, a test for a function calculating sum of two integers is like following.
```python
from skqulacs import add # This function does not exist in the module.

def test_add():
    assert 3 == add(1, 2)
```

3. Then run tests.
```bash
make test
```
If assertion fail, error contents are displayed with red. If you do not see that, all test are successful.

You might want to run tests only in specific files.
In that case, run `make` with file(s) you want to test.
```
make tests/test_circuit.py tests/test_qnn_regressor.py
```

We use `pytest` for testing. Detailed instructions are available in the [document](https://docs.pytest.org/en/6.2.x/).

## CI
Run CI at GitHub Actions. You cannot merge a branch unless CI passes.
In CI, we run tests and check code format and linter error.
The purpose of CI is
* Share our code works properly in the team.
* Find error you cannot notice at your local machine.
* Avoid unnecessary diff by forcing code format and linter error.

## Documentation
This repository's documentation includes API document and Jupyter Notebook style tutorials.

The documentation is available here: https://qulacs-osaka.github.io/scikit-qulacs/index.html
It is built and deployed on pushing(merged from PR) to `main` branch.

### Build document
Just run following command.
```bash
make html
```

In `doc/build/html`, you can find build artifacts including HTML files.

Or you can check artifacts via browser easily.
```bash
make serve
```
By running this command, the documentation is built and you can access it from [`localhost:8080`](http://localhost:8000/).


### Create Page from jupyter notebook
You can create a documentation page from jupyter notebook.
It is useful to show an example usage of APIs in this library.
1. Create `.ipynb` file in the `doc/source/notebooks`(suppose `0_tutorial.ipynb` here).
2. Edit the contents and **be sure to execute all cells without error**. Code and results in the notebook is embedded to documentation without modification.
3. Add the file name without its extension to `doc/source/notebooks/index.rst`.
4. Execute `make html` or `make serve` to generate HTMLs.

Example of `doc/source/notebook/index.rst`:
```
Notebooks
---------

.. toctree::

   0_tutorial
```

You can display LaTeX and images.
