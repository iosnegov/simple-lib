# tests/test_lib.py
import os
import tempfile
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError
from simple_lib.model import GeoNearestNeighbor
from simple_lib.types import Coordinates, PredictRequest, PredictResponse

class TestGeoNearestNeighbor:
    """Test suite for GeoNearestNeighbor class."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'description': ['Moscow Kremlin', 'Red Square', 'Gorky Park', 'Arbat Street', 'Bolshoi Theatre'],
            'latitude': [55.752023, 55.753930, 55.729600, 55.749000, 55.760101],
            'longitude': [37.617499, 37.620795, 37.603900, 37.594600, 37.618778]
        })
    
    @pytest.fixture
    def invalid_data(self):
        return pd.DataFrame({
            'description': ['Invalid Place'],
            'latitude': [92.0], # out of valid range
            'longitude': [37.617499]
        })
    
    @pytest.fixture
    def incomplete_data(self):
        return pd.DataFrame({
            'description': ['Place A'],
            'latitude': [55.752023]
            # Missing longitude column
        })

    @pytest.fixture
    def data_with_extra(self):
        return pd.DataFrame({
            'description': ['Moscow Kremlin', 'Red Square'],
            'latitude': [55.752023, 55.753930],
            'longitude': [37.617499, 37.620795],
            'extra_column': ['extra1', 'extra2']
        })
    
    @pytest.fixture
    def geo_model(self):
        return GeoNearestNeighbor()
    
    @pytest.fixture
    def fitted_model(self, geo_model: GeoNearestNeighbor, sample_data: pd.DataFrame):
        return geo_model.fit(sample_data)
    
    def test_init_default_parameters(self):
        model = GeoNearestNeighbor()
        params = model.get_params()
        
        assert params['algorithm'] == 'ball_tree'
        assert params['leaf_size'] == 30
        assert params['n_jobs'] == -1
    
    def test_init_custom_parameters(self):
        model = GeoNearestNeighbor(
            algorithm='kd_tree',
            leaf_size=50,
            n_jobs=1
        )
        params = model.get_params()
        
        assert params['algorithm'] == 'kd_tree'
        assert params['leaf_size'] == 50
        assert params['n_jobs'] == 1
    
    def test_fit_accepts_valid_data_and_sets_data_attribute(
        self, 
        geo_model: GeoNearestNeighbor, 
        sample_data: pd.DataFrame
    ):
        result = geo_model.fit(sample_data)
        
        assert result is geo_model
        assert hasattr(geo_model, 'data_')
        assert len(geo_model.data_) == len(sample_data)

        expected_cols = ['description', 'latitude', 'longitude']
        assert list(geo_model.data_.columns) == expected_cols
    
    def test_fit_raises_when_required_columns_missing(
        self, 
        geo_model: GeoNearestNeighbor, 
        incomplete_data: pd.DataFrame
    ):
        with pytest.raises(ValueError, match="Missing required columns"):
            geo_model.fit(incomplete_data)
    
    def test_fit_raises_on_invalid_coordinates(
        self, 
        geo_model: GeoNearestNeighbor, 
        invalid_data: pd.DataFrame
    ):
        with pytest.raises(ValueError, match="Invalid coordinates"):
            geo_model.fit(invalid_data)
    
    def test_fit_ignores_extra_columns(
        self, 
        geo_model: GeoNearestNeighbor, 
        data_with_extra: pd.DataFrame
    ):
        result = geo_model.fit(data_with_extra)
        
        expected_cols = ['description', 'latitude', 'longitude']
        assert list(result.data_.columns) == expected_cols
        assert 'extra_column' not in result.data_.columns
    
    def test_predict_raises_if_not_fitted(
        self, 
        geo_model: GeoNearestNeighbor
    ):
        query = PredictRequest(
            coordinates=Coordinates(latitude=55.752023, longitude=37.617499),
            radius_meters=1000
        )
        
        with pytest.raises(NotFittedError):
            geo_model.predict(query)
    
    def test_predict_returns_result_for_query_within_radius(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        query = PredictRequest(
            coordinates=Coordinates(latitude=55.752023, longitude=37.617499),
            radius_meters=1000
        )
        response = fitted_model.predict(query)
        
        assert isinstance(response, PredictResponse)
        assert response.found is True
        assert response.result is not None
    
    def test_predict_returns_not_found_when_no_neighbors_in_radius(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        query = PredictRequest(
            coordinates=Coordinates(latitude=78, longitude=78),
            radius_meters=1
        )
        response = fitted_model.predict(query)
        
        assert isinstance(response, PredictResponse)
        assert response.found is False
        assert response.result is None
    
    def test_call_invokes_predict(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        query = PredictRequest(
            coordinates=Coordinates(latitude=55.752023, longitude=37.617499),
            radius_meters=1000
        )
        call_response = fitted_model(query)
        predict_response = fitted_model.predict(query)
        
        assert isinstance(call_response, PredictResponse)
        assert isinstance(predict_response, PredictResponse)
        assert call_response.found == predict_response.found
    
    def test_save_model_raises_if_not_fitted(
        self, 
        geo_model: GeoNearestNeighbor
    ):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Model must be fitted before saving"):
                geo_model.save_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                # Remove the temporary file from the filesystem to clean up after the test
                os.unlink(tmp_path)
    
    def test_save_and_load_preserves_data_and_behavior(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
            
            try:
                fitted_model.save_model(tmp_path)
                assert os.path.exists(tmp_path)
                
                loaded_model = GeoNearestNeighbor()
                loaded_model.load_model(tmp_path)
                pd.testing.assert_frame_equal(loaded_model.data_, fitted_model.data_)
                
                query = PredictRequest(
                    coordinates=Coordinates(latitude=55.752023, longitude=37.617499),
                    radius_meters=1000
                )
                response = loaded_model.predict(query)
                assert isinstance(response, PredictResponse)
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_save_uses_default_filepath_when_none_provided(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        try:
            fitted_model.save_model()
            assert os.path.exists(GeoNearestNeighbor.DEFAULT_FILEPATH)
            
        finally:
            if os.path.exists(GeoNearestNeighbor.DEFAULT_FILEPATH):
                os.unlink(GeoNearestNeighbor.DEFAULT_FILEPATH)
                # Clean up directory if empty
                dir_path = os.path.dirname(GeoNearestNeighbor.DEFAULT_FILEPATH)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
    
    def test_from_file_loads_instance_from_path(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fitted_model.save_model(tmp_path)
            loaded_model = GeoNearestNeighbor.from_file(tmp_path)
            assert isinstance(loaded_model, GeoNearestNeighbor)
            pd.testing.assert_frame_equal(loaded_model.data_, fitted_model.data_)
            
            query = PredictRequest(
                coordinates=Coordinates(latitude=55.752023, longitude=37.617499),
                radius_meters=1000
            )
            response = loaded_model.predict(query)
            assert isinstance(response, PredictResponse)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_from_file_uses_default_filepath_when_none_provided(
        self, 
        fitted_model: GeoNearestNeighbor
    ):
        try:
            fitted_model.save_model()
            loaded_model = GeoNearestNeighbor.from_file()
            assert isinstance(loaded_model, GeoNearestNeighbor)
            pd.testing.assert_frame_equal(loaded_model.data_, fitted_model.data_)
            
        finally:
            if os.path.exists(GeoNearestNeighbor.DEFAULT_FILEPATH):
                os.unlink(GeoNearestNeighbor.DEFAULT_FILEPATH)
                dir_path = os.path.dirname(GeoNearestNeighbor.DEFAULT_FILEPATH)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)