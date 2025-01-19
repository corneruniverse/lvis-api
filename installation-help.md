# machine-learning
Custom machine learning work

# Setup

[Install the LVIS API](https://github.com/lvis-dataset/lvis-api)


## Help installing LVIS API
If numpy isn’t installed in your environment before building the cocoapi wheel. Because the cocoapi’s setup tries to import numpy, the build process fails if numpy doesn’t exist first.

Here’s the quickest fix:

1. Make sure you have the newest pip and wheel:
```bash
pip install --upgrade pip setuptools wheel
```

2. Install Cython (sometimes also needed for cocoapi):
```bash
pip install cython
```

3. Install numpy:
```bash
pip install numpy
```

4. Finally, install cocoapi from GitHub:
```bash
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Following that sequence should resolve the `ModuleNotFoundError: No module named 'numpy'` and allow the wheel to build successfully. If you still run into issues after installing numpy, double-check that your virtual environment is active (if applicable) and that you’re installing everything into the same environment.