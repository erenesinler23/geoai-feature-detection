import geopandas as gpd
from shapely.geometry import box
from src.detectors.hybrid import HybridDetector


class FakePrimary:
    name = "naive"
    def detect(self, r, g, b, transform, crs):
        return gpd.GeoDataFrame({
            "detection_id": [1, 2, 3],
            "geometry": [
                box(0.0, 0.0, 0.001, 0.001),
                box(1.0, 1.0, 1.001, 1.001),
                box(0.0002, 0.0002, 0.0007, 0.0007),
            ],
        }, crs="EPSG:4326")


class FakeConfirmingPlausible:
    name = "owlvit"
    def detect(self, r, g, b, transform, crs):
        return gpd.GeoDataFrame({
            "detection_id": [1],
            "score": [0.05],
            "geometry": [box(-0.0005, -0.0005, 0.0015, 0.0015)],
        }, crs="EPSG:4326")


class FakeConfirmingOversized:
    name = "owlvit"
    def detect(self, r, g, b, transform, crs):
        return gpd.GeoDataFrame({
            "detection_id": [1],
            "score": [0.02],
            "geometry": [box(-1.0, -1.0, 5.0, 5.0)],
        }, crs="EPSG:4326")


def test_confirms_only_plausible_overlaps():
    hybrid = HybridDetector(FakePrimary(), FakeConfirmingPlausible(), max_size_m=500)
    result = hybrid.detect(None, None, None, None, None)
    assert len(result) == 2


def test_rejects_when_confirming_detector_is_oversized():
    hybrid = HybridDetector(FakePrimary(), FakeConfirmingOversized(), max_size_m=500)
    result = hybrid.detect(None, None, None, None, None)
    assert len(result) == 0


if __name__ == "__main__":
    test_confirms_only_plausible_overlaps()
    test_rejects_when_confirming_detector_is_oversized()
    print("Both tests passed.")
