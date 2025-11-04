"""
A simple library for building geographic recommendation systems.

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

Example
-------
>>> from simple_lib.model import GeoNearestNeighbor
>>> from simple_lib.types import Coordinates, PredictRequest
>>> model = GeoNearestNeighbor()
>>> model.fit(dataframe)
>>> coordinates = Coordinates(latitude = 55.7, longitude = 37.56)
>>> predict_request = PredictRequest(coordinates = coordinates, radius_meters = 5000)
>>> response = model(predict_request)
>>> print(response.model_dump_json(indent=2))
"""

__version__ = "0.1.0"
__author__ = "Sierra Arn"