# Simple Geographic Recommendation System Library

*A simple library for building geographic recommendation systems.*

> [!WARNING]  
> **Project Archived**
> 
> This project is no longer actively maintained. 
> 
> **Context:**  
> Originally, this repository began as a modular component of a large-scale university assignment. Rather than simply completing the task and discarding the code, I chose to expand it into a structured mini-project to explore geographic data processing and machine learning concepts. That said, it was purely a personal experiment and was never intended to be a production-ready or educational tool.
> 
> **Reason for Archiving:**  
> Consequently, I no longer have an interest in further maintaining this codebase. This repository is now archived purely as a historical artifact, serving as a snapshot of my early learning journey and a reminder of where I started in software development.

## Brief Overview

This project is developed for educational purposes and demonstrates how to combine geographic data processing with machine learning. 

Key features include:

- **Haversine distance calculation** for accurate geographic distances.
- **Radius-based search** for finding places within a specified distance.
- **Single best result** returned based on proximity and relevance.
- **Callable interface** for intuitive and straightforward usage.

## Project Structure

```
simple-lib/
├── src/                # Library source code
├── tests/              # Tests and usage examples
├── pyproject.toml      # Python package build configuration
├── pixi.toml           # Pixi project configuration, environments, and dependencies
└── pixi.lock           # Locked dependency versions for reproducible environments
```

## Dependencies Overview

| Dependency | Repository | What it is |
| :--- | :--- | :--- |
| **CPython** | [python/cpython](https://github.com/python/cpython) | The de facto standard implementation of the Python programming language. |
| **scikit-learn** | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | Machine learning algorithms and nearest neighbor search. |
| **pandas** | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | Data manipulation and analysis library. |
| **numpy** | [numpy/numpy](https://github.com/numpy/numpy) | Fundamental package for scientific computing and numerical operations. |
| **Pydantic** | [pydantic/pydantic](https://github.com/pydantic/pydantic) | Data validation and settings management using Python type annotations. |
| **joblib** | [joblib/joblib](https://github.com/joblib/joblib) | Efficient serialization (pickling) of Python objects, especially large numpy arrays. |
| **huggingface_hub** | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Client library for interacting with the Hugging Face Hub for model storage and sharing. |

## Pre-built Installation

If you only need to use the library without developing it, you can install the pre-built wheel directly.

**Via pip:**
```bash
pip install "simple-lib @ https://github.com/snegoviol/simple-lib/releases/download/v0.1.0/simple_lib-0.1.0-py3-none-any.whl"
```

**Via Pixi:**
```bash
pixi add --pypi "simple-lib @ https://github.com/snegoviol/simple-lib/releases/download/v0.1.0/simple_lib-0.1.0-py3-none-any.whl"
```

> **Note:**  
> You can find an example of how to use the library in the [interactive notebook](tests/test-lib.ipynb).

## Local Development

### I. Prerequisites

| Component            | Requirement                               |
| :------------------- | :---------------------------------------- |
| **Package Manager**  | [`Pixi`](https://pixi.prefix.dev/latest/) |
| **Operating System** | GNU/Linux-based system                    |
| **Architecture**     | `x86_64`                                  |

> **Note:**  
> These are not strict requirements, but rather the baseline environment where the code is 
> guaranteed to run out of the box. Alternative setups are possible but may require manual 
> configuration.

### II. Environment Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/snegoviol/simple-lib.git
    cd simple-lib
    ```

2. **Install dependencies**

    ```bash
    pixi install --all
    ```

### III. Build and Install

1. **Build the package:**

    ```bash
    pixi run -e build-env build-lib
    ```

2. **Install the package in the test environment:**

    ```bash
    pixi run -e test-env install-lib
    ```

### IV. Testing

Once the environment is set up, you can test the package functionality using either automated tests or an interactive notebook.

1. **Run automated tests with a code coverage report:**

    ```bash
    pixi run -e test-env test-lib
    ```

2. **Explore the interactive notebook:**

    For a more hands-on approach, you can run the [interactive Jupyter notebook](tests/test-lib.ipynb) to test code functionality and explore usage examples step-by-step.

## License

Each file in this repository is licensed under the [MIT No Attribution License](LICENSE.txt).
