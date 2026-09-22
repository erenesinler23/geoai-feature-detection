from abc import ABC, abstractmethod
import geopandas as gpd
import numpy as np


class Detector(ABC):
    """Shared interface for detectors used by the pipeline."""

    @abstractmethod
    def detect(self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
               transform, raster_crs: str) -> gpd.GeoDataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
