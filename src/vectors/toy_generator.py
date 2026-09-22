import geopandas as gpd
from shapely.geometry import Polygon


class ToyBuildingGenerator:
    """Create four sample building polygons relative to the given origin."""

    def __init__(self, origin_lon: float, origin_lat: float):
        self.lon0 = origin_lon
        self.lat0 = origin_lat

    def create(self) -> gpd.GeoDataFrame:
        lon0, lat0 = self.lon0, self.lat0
        return gpd.GeoDataFrame({
            "building_id": [1, 2, 3, 4],
            "roof_color": ["red", "grey", "grey", "brown"],
            "floors": [2, 1, 3, 2],
            "geometry": [
                Polygon([(lon0+0.004, lat0-0.004), (lon0+0.005, lat0-0.004),
                          (lon0+0.005, lat0-0.003), (lon0+0.004, lat0-0.003)]),
                Polygon([(lon0-0.001, lat0-0.006), (lon0+0.000, lat0-0.006),
                          (lon0+0.000, lat0-0.005), (lon0-0.001, lat0-0.005)]),
                Polygon([(lon0+0.010, lat0-0.005), (lon0+0.011, lat0-0.005),
                          (lon0+0.011, lat0-0.004), (lon0+0.010, lat0-0.004)]),
                Polygon([(lon0+0.007, lat0-0.001), (lon0+0.008, lat0-0.001),
                          (lon0+0.008, lat0-0.000), (lon0+0.007, lat0-0.000)]),
            ]
        }, crs="EPSG:4326")
