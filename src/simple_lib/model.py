# src/simple_lib/model.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib
from huggingface_hub import hf_hub_download
from .types import Coordinates, PredictRequest, SearchResult, PredictResponse


class GeoNearestNeighbor(NearestNeighbors):
    """
    Extension of scikit-learn's NearestNeighbors for geographic (latitude/longitude) data
    using the Haversine metric.

    Parameters
    ----------
    algorithm : str, optional
        Algorithm used by the underlying NearestNeighbors estimator (default 'ball_tree').
    leaf_size : int, optional
        Leaf size passed to tree-based algorithms (default 30).
    n_jobs : int or None, optional
        Number of parallel jobs to run for neighbors search. -1 uses all processors.

    Attributes
    ----------
    METRIC : str
        Distance metric name used ('haversine').
    EARTH_RADIUS_METERS : float
        Earth's radius in meters for converting between radians and meters.
    DEFAULT_FILEPATH : str
        Default path used for model serialization.
    DESCRIPTION_COL, LATITUDE_COL, LONGITUDE_COL : str
        Standardized column names expected in input DataFrame.
    data_ : pandas.DataFrame
        Stored validated data used for formatting search results after fitting.

    Notes
    -----
    The column names are fixed to enforce a consistent and standardized
    DataFrame schema. This ensures compatibility across different datasets
    and prevents errors caused by mismatched or missing column names.
    """

    METRIC: str = 'haversine'
    EARTH_RADIUS_METERS: float = 6371000.0
    DEFAULT_FILEPATH: str = "models/geo_nearest_neighbor.joblib"
    DESCRIPTION_COL: str = 'description'
    LATITUDE_COL: str = 'latitude'
    LONGITUDE_COL: str = 'longitude'

    def __init__(
        self,
        algorithm: str = 'ball_tree',
        leaf_size: int = 30,
        n_jobs: int = -1
    ) -> None:
        """
        Initialize GeoNearestNeighbor.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm to use for the underlying nearest neighbors implementation.
        leaf_size : int, optional
            Leaf size for tree-based algorithms.
        n_jobs : int, optional
            Number of parallel jobs to run.

        Returns
        -------
        None

        Notes
        -----
        Passes provided configuration to the parent NearestNeighbors constructor
        and fixes the metric to 'haversine'.
        """
        
        super().__init__(
            metric=self.METRIC,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )

    def fit(self, X: pd.DataFrame) -> "GeoNearestNeighbor":
        """
        Fit the model with place data.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing required columns:
            - description (str)
            - latitude (float)
            - longitude (float)

        Returns
        -------
        GeoNearestNeighbor
            The fitted estimator (self).

        Raises
        ------
        ValueError
            If required columns are missing or any coordinate values fail validation.

        Notes
        -----
        The method validates column presence and coordinate ranges using the
        :class:`Coordinates` pydantic model, stores a reduced copy of the data
        in ``self.data_``, converts coordinates to radians and fits the underlying
        scikit-learn NearestNeighbors estimator.
        """
        
        required_cols = [self.DESCRIPTION_COL, self.LATITUDE_COL, self.LONGITUDE_COL]
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        for idx, row in X.iterrows():
            try:
                Coordinates(latitude=row[self.LATITUDE_COL], longitude=row[self.LONGITUDE_COL])
            except Exception as e:
                raise ValueError(f"Invalid coordinates at row {idx}: {e}")
        
        self.data_ = X[required_cols].copy()
        
        coords = np.column_stack((
            X[self.LATITUDE_COL].astype(np.float64),
            X[self.LONGITUDE_COL].astype(np.float64)
        ))
        coords_rad = self.transform(coords)
        super().fit(coords_rad)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from degrees to radians.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2) or (2,)
            Coordinate array where columns are [latitude, longitude] in degrees.
            A single coordinate pair (1-D array of length 2) is accepted.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Coordinates converted to radians.

        Notes
        -----
        The haversine metric in scikit-learn requires input coordinates
        expressed in radians instead of degrees.  
        This helper method performs two preprocessing steps before conversion:
        
        1. **Dimensional normalization:**  
           If a single coordinate pair is provided as a 1-D array, it is reshaped
           into a 2-D array of shape (1, 2) to ensure consistent handling.
        
        2. **Degree-to-radian conversion:**  
           The input array is converted to a NumPy array (if not already one),
           and `numpy.radians` is applied to convert latitude and longitude
           values from degrees to radians.  
           
        The resulting array preserves the input shape (n_samples, 2).
        """

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.radians(X)

    def predict(self, X: PredictRequest) -> PredictResponse:
        """
        Search for the closest place within a geographic radius.

        Parameters
        ----------
        X : PredictRequest
            Request object containing:
            - coordinates : Coordinates (latitude, longitude)
            - radius_meters : float (search radius in meters)

        Returns
        -------
        PredictResponse
            Response object with:
            - found : bool indicating if any neighbor was found
            - result : SearchResult or None for the best match

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted prior to calling predict.

        Notes
        -----
        Performs:
        1. Conversion of the query coordinates to radians.
        2. Radius neighbors search in radians (radius converted from meters).
        3. Selection of the closest neighbor (if any) and construction of a
           validated :class:`SearchResult` with distance converted back to meters.
        """

        # Ensure the model has been fitted before attempting predictions
        check_is_fitted(self, ['data_'])

        def _prepare_coordinates() -> np.ndarray:
            """
            Prepare query coordinates for haversine calculation.

            Returns
            -------
            np.ndarray, shape (1, 2)
                Query coordinates in radians.
            """
            coords_array = X.coordinates.to_array()
            return self.transform(coords_array)

        def _find_neighbors(
            coords_rad: np.ndarray, 
            radius_rad: float
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Find all neighbors within the specified search radius.

            Parameters
            ----------
            coords_rad : np.ndarray
                Query coordinates in radians.
            radius_rad : float
                Search radius in radians.

            Returns
            -------
            tuple of (np.ndarray, np.ndarray)
                Distances (in radians) and corresponding indices of neighbors,
                sorted by increasing distance.
            """
            return self.radius_neighbors(
                coords_rad,
                radius=radius_rad,
                return_distance=True,
                sort_results=True
            )

        def _process_results(
            distances_rad: np.ndarray, 
            indices: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Convert radial distances to meters and extract the closest neighbor.

            Parameters
            ----------
            distances_rad : np.ndarray
                Distances to neighbors in radians.
            indices : np.ndarray
                Indices of neighbors corresponding to distances.

            Returns
            -------
            tuple
                (closest_idx, closest_dist), where:
                - closest_idx : int — index of the nearest place
                - closest_dist : float — distance to it in meters
            """
            distances_m = distances_rad[0] * self.EARTH_RADIUS_METERS
            closest_idx, closest_dist = indices[0][0], distances_m[0]
            return closest_idx, closest_dist

        def _format_result(idx: int, distance: float) -> PredictResponse:
            """
            Format a single search result into a PredictResponse object.

            Parameters
            ----------
            idx : int
                Index of the matched record in the training data.
            distance : float
                Distance to the found place in meters.

            Returns
            -------
            PredictResponse
                Structured response containing the found location and distance.
            """
            row = self.data_.iloc[idx]

            result = SearchResult(
                description=str(row[self.DESCRIPTION_COL]),
                coordinates=Coordinates(
                    latitude=float(row[self.LATITUDE_COL]),
                    longitude=float(row[self.LONGITUDE_COL])
                ),
                geo_distance=float(round(distance, 2))
            )

            return PredictResponse(result=result, found=True)

        # Main logic
        coords_rad = _prepare_coordinates()
        radius_rad = X.radius_meters / self.EARTH_RADIUS_METERS

        distances_rad, indices = _find_neighbors(coords_rad, radius_rad)

        if len(indices[0]) == 0:
            return PredictResponse(found=False, result=None)

        closest_idx, closest_dist = _process_results(distances_rad, indices)

        return _format_result(closest_idx, closest_dist)

    def __call__(self, X: PredictRequest) -> PredictResponse:
        """
        Alias to allow using the estimator as a callable.

        Parameters
        ----------
        X : PredictRequest
            Same as for :meth:`predict`.

        Returns
        -------
        PredictResponse
            Result of calling :meth:`predict` with the provided request.

        Notes
        -----
        Convenience wrapper so that ``model(request)`` is equivalent to
        ``model.predict(request)``.
        """
        return self.predict(X)

    def save_model(self, filepath: str | None = None) -> None:
        """
        Save the complete model state to disk.

        Parameters
        ----------
        filepath : str or None, optional
            Filesystem path where the model will be saved. If None, uses
            the class-level DEFAULT_FILEPATH.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the model has not been fitted before saving.

        Notes
        -----
        The method ensures the parent directory exists and serializes the full
        estimator (including ``self.data_``) using joblib with compression.
        """
        if filepath is None:
            filepath = self.DEFAULT_FILEPATH
        
        try:
            check_is_fitted(self, ['data_'])
        except NotFittedError:
            raise ValueError("Model must be fitted before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath, compress=3)

    def load_model(self, filepath: str | None = None) -> "GeoNearestNeighbor":
        """
        Load model state into the current instance.

        Parameters
        ----------
        filepath : str or None, optional
            Path to a saved model file. If None, uses DEFAULT_FILEPATH.

        Returns
        -------
        GeoNearestNeighbor
            The current instance updated with the loaded model state.

        Raises
        ------
        Exception
            Propagates exceptions raised by joblib.load when loading fails.

        Notes
        -----
        This method deserializes a joblib file and updates ``self.__dict__`` with
        the loaded object's attributes, effectively restoring fitted parameters
        and stored metadata.
        """
        if filepath is None:
            filepath = self.DEFAULT_FILEPATH
        
        loaded_model = joblib.load(filepath)
        self.__dict__.update(loaded_model.__dict__)
        return self

    @classmethod
    def from_file(cls, filepath: str | None = None) -> "GeoNearestNeighbor":
        """
        Create a new instance from a saved file.

        Parameters
        ----------
        filepath : str or None, optional
            Path to the saved model file. If None, uses DEFAULT_FILEPATH.

        Returns
        -------
        GeoNearestNeighbor
            A new estimator instance loaded from the file.

        Raises
        ------
        Exception
            Propagates exceptions raised by joblib.load when loading fails.
        """
        if filepath is None:
            filepath = cls.DEFAULT_FILEPATH
        
        return joblib.load(filepath)

    @classmethod
    def from_huggingface_hub(
        cls, 
        repo_id: str,
        filename: str | None = None,
        cache_dir: str = "models/",
        force_download: bool = False,
        token: str | None = None
    ) -> "GeoNearestNeighbor":
        """
        Download and load model from the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Repository identifier on Hugging Face (e.g. "username/model-name").
        filename : str or None, optional
            Filename in the repository to download; if None, defaults to DEFAULT_FILEPATH.
        cache_dir : str, optional
            Local directory to cache downloaded files (default "models/").
        force_download : bool, optional
            If True, re-download even when a cached file exists.
        token : str or None, optional
            Authentication token for private repositories.

        Returns
        -------
        GeoNearestNeighbor
            The estimator instance loaded from the downloaded file.

        Raises
        ------
        RuntimeError
            If the download or loading fails; the original exception message is wrapped.
        """
        if filename is None:
            filename = cls.DEFAULT_FILEPATH

        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token
            )
            
            return joblib.load(model_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model from Hugging Face Hub: {e}")