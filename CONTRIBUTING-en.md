# Contributing scikit-qulacs

## Start coding
Set up the project.
1. First, clone this repository.
```bash
git clone git@github.com:Qulacs-Osaka/scikit-qulacs.git
cd scikit-qulacs
```

2. Install dependencies and development tools.
```bash
pip install -r requirements-dev.txt
# This installs dependencies and creates a symbolic link to this directory in 
# the site-packages directory.
make install
```

Next, workflow through modification to merge.
3. Synchronize with `main`(not necessary for the first time).
```bash
git switch main
git pull main
```

4. Create branch with name describing what you are going to develop.
```bash
git switch -c 99-wonderful-model
```

5. Run test and format code before commit.
```bash
make format
make test
```

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

We use `pytest` for testing. Detailed instructions are available in the [document](https://docs.pytest.org/en/6.2.x/).

## CI
Run CI at GitHub Actions. You cannot merge a branch unless CI passes.
In CI, we run tests and check code format.
The purpose of CI is
* Share our code works properly in the team.
* Find error you cannot notice at your local machine.
* Avoid unnecessary diff by forcing code format.

## Installation
You can install skqulacs to your python's site-packages by `setup-tools`.
Although `make install` just creates a symlink to this directory, this method builds a complete package.

First, install `build`.
```bash
pip install build
```
Then, build and install this package.
```bash
python -m build
# This file name might be different among environments.
pip install dist/scikit_qulacs-0.0.1-py3-none-any.whl
```