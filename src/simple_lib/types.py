# src/simple_lib/types.py
import numpy as np
from pydantic import BaseModel, Field

class Coordinates(BaseModel):
    """
    Geographic coordinates.

    Attributes
    ----------
    latitude : float
        Latitude in degrees. Must be between -90 and 90.
    longitude : float
        Longitude in degrees. Must be between -180 and 180.
    """

    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")

    def to_array(self) -> np.ndarray:
        """
        Convert coordinates to a NumPy array.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (2,) containing [latitude, longitude].
        """
        return np.array([self.latitude, self.longitude])


class PredictRequest(BaseModel):
    """
    Request object for a geographic nearest neighbor search.

    Attributes
    ----------
    coordinates : Coordinates
        Geographic coordinates to use as the search center.
    radius_meters : float
        Search radius in meters. Must be greater than 0.
    """

    coordinates: Coordinates
    radius_meters: float = Field(..., gt=0, description="Search radius in meters")


class SearchResult(BaseModel):
    """
    Result of a geographic nearest neighbor search.

    Attributes
    ----------
    description : str
        Description of the found location.
    coordinates : Coordinates
        Geographic coordinates of the found location.
    geo_distance : float
        Distance between the query and result in meters. Must be non-negative.
    """

    description: str = Field(..., min_length=1, description="Place description")
    coordinates: Coordinates = Field(..., description="Geographic coordinates")
    geo_distance: float = Field(..., ge=0, description="Distance in meters")


class PredictResponse(BaseModel):
    """
    Response wrapper for the geographic search operation.

    Attributes
    ----------
    found : bool
        Indicates whether a matching result was found.
    result : SearchResult or None
        Search result if found, otherwise None.
    """

    found: bool = Field(default=True, description="Whether a result was found")
    result: SearchResult | None = Field(..., description="Search result")