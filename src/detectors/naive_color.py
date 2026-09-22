import numpy as np
from scipy import ndimage
from pyproj import Transformer
from shapely.geometry import Polygon
import geopandas as gpd
from .base import Detector


class NaiveColorDetector(Detector):
    """Group pixels that pass the red colour thresholds into candidate regions."""

    def __init__(self, red_min: float, red_minus_green_min: float, red_minus_blue_min: float):
        self.red_min = red_min
        self.red_minus_green_min = red_minus_green_min
        self.red_minus_blue_min = red_minus_blue_min

    @property
    def name(self) -> str:
        return "naive"

    def detect(self, r, g, b, transform, raster_crs) -> gpd.GeoDataFrame:
        r, g, b = (np.asarray(band, dtype=float) for band in (r, g, b))
        red_mask = (
            (r > self.red_min)
            & (r - g > self.red_minus_green_min)
            & (r - b > self.red_minus_blue_min)
        )
        labeled, n_blobs = ndimage.label(red_mask)
        to_lonlat = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

        boxes = []
        for i in range(1, n_blobs + 1):
            ys, xs = np.where(labeled == i)
            row_min, row_max, col_min, col_max = ys.min(), ys.max(), xs.min(), xs.max()
            corners = [transform * (x, y) for x, y in
                       [(col_min, row_min), (col_max + 1, row_min),
                        (col_max + 1, row_max + 1), (col_min, row_max + 1)]]
            boxes.append(Polygon([to_lonlat.transform(x, y) for x, y in corners]))

        return gpd.GeoDataFrame(
            {"detection_id": range(1, len(boxes) + 1), "geometry": boxes},
            crs="EPSG:4326",
        )
