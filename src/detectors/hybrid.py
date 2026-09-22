import geopandas as gpd
from .base import Detector
from src.validation.validator import metric_frame, plausible_mask


def fuse_detections(naive_detections, owlvit_detections, max_size_m=500.0):
    if naive_detections.empty or owlvit_detections.empty:
        return naive_detections.iloc[:0].copy()
    crs = naive_detections.estimate_utm_crs()
    naive = metric_frame(naive_detections, crs)
    owlvit = metric_frame(owlvit_detections, crs)
    owlvit = owlvit.loc[plausible_mask(owlvit, max_size_m)]
    centres = naive.copy()
    centres.geometry = naive.geometry.centroid
    confirmed = gpd.sjoin(centres, owlvit[["geometry"]], how="inner", predicate="within")
    result = naive_detections[naive_detections.detection_id.isin(confirmed.detection_id)].copy()
    return result.reset_index(drop=True)


class HybridDetector(Detector):
    """Keep colour candidates supported by a plausible model box."""

    def __init__(self, primary, confirming, max_size_m=500.0):
        self.primary = primary
        self.confirming = confirming
        self.max_size_m = max_size_m

    @property
    def name(self):
        return "hybrid"

    def detect(self, r, g, b, transform, raster_crs):
        return fuse_detections(self.primary.detect(r, g, b, transform, raster_crs),
                               self.confirming.detect(r, g, b, transform, raster_crs), self.max_size_m)
