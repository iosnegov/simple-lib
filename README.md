# **Simple Geographic Recommendation System Library**

*A simple library for building geographic recommendation systems.*

---

## **Overview**

This project is developed for educational purposes and demonstrates
how to combine geographic data processing with machine learning.

The implementation uses:

- [scikit-learn](https://github.com/scikit-learn/scikit-learn) —
  for machine learning algorithms and nearest neighbor search.
- [pandas](https://github.com/pandas-dev/pandas) and
  [numpy](https://github.com/numpy/numpy) —
  for data handling and numerical computations.
- [Pydantic](https://github.com/pydantic/pydantic) —
  for strict data validation and type safety.
- [joblib](https://github.com/joblib/joblib) —
  for efficient model serialization.
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) —
  for cloud model storage and sharing.

## **Key Features**

- Haversine distance calculation for accurate geographic distances.
- Radius-based search for finding places within specified distance.
- Returns single best result.
- Callable interface for intuitive usage.

## **Library Installation**

```bash
pip install https://github.com/Sierra-Arn/simple-lib/releases/download/v0.1.0/simple_lib-0.1.0-py3-none-any.whl
```

> **Note:** You can find an example of how to use the library [here](tests/test-lib.ipynb).

## **Development and Testing**

### **I. Prerequisites**

- Package manager [Pixi](https://pixi.sh/latest/).

### **II. Project Structure**

```bash
simple-lib/
├── src/                # Library source code
├── pyproject.toml      # Python package build configuration
├── dist/               # Package builds and distributions
├── tests/              # Tests and usage examples
├── pixi.toml           # Pixi project configuration and dependencies
└── pixi.lock           # Locked dependency versions
```

### **III. Environment Setup**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sierra-arn/simple-lib.git
   cd simple-lib
   ```

2. **Install dependencies:**

   ```bash
   pixi install --all
   ```

### **IV. Main Pixi Commands**

1. Build the package:

   ```bash
   pixi run -e build-env build-lib
   ```

2. Install the package in the test environment:

   ```bash
   pixi run -e test-env install-lib
   ```

3. Test the package with code coverage report:

   ```bash
   pixi run -e test-env test-lib
   ```

   > **Note:** an [interactive version](tests/test-lib.ipynb) is also available for testing code functionality.

## **License**

The project is distributed under the [BSD-3-Clause License](LICENSE).