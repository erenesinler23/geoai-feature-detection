from dataclasses import dataclass, field
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union


@dataclass
class FetchedScene:
    scene_id: str
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    transform: object
    raster_crs: str
    buildings: gpd.GeoDataFrame
    valid_area: object = None
    metadata: dict = field(default_factory=dict)
    valid_mask: object = None


@dataclass
class PipelineRun:
    scene_id: str
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    transform: object
    raster_crs: str
    detections: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    result: object


def clean_polygons(frame):
    frame = frame.copy()
    frame.geometry = frame.geometry.make_valid()
    def polygons(geometry):
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            return geometry
        return unary_union([part for part in getattr(geometry, "geoms", [])
                            if part.geom_type in {"Polygon", "MultiPolygon"}])
    frame.geometry = frame.geometry.apply(polygons)
    return frame.loc[~frame.geometry.is_empty].reset_index(drop=True)


def restrict_detections(detections, scene):
    projected = detections.to_crs(scene.raster_crs)
    if scene.valid_area is not None:
        projected.geometry = projected.geometry.intersection(scene.valid_area)
    projected = projected.loc[~projected.geometry.is_empty & (projected.geometry.area > 0)].copy()
    return clean_polygons(projected.to_crs("EPSG:4326"))


class Pipeline:
    def __init__(self, sentinel_fetcher, osm_fetcher, validator):
        self.sentinel_fetcher = sentinel_fetcher
        self.osm_fetcher = osm_fetcher
        self.validator = validator

    def fetch(self, target_lon, target_lat, osm_bbox=None):
        r, g, b, transform, crs, scene_id = self.sentinel_fetcher.fetch(target_lon, target_lat)
        valid = getattr(self.sentinel_fetcher, "valid_mask", np.ones(r.shape, dtype=bool))
        area = unary_union([shape(geom) for geom, value in shapes(valid.astype("uint8"), mask=valid, transform=transform) if value == 1])
        geo_area = gpd.GeoSeries([area], crs=crs).to_crs("EPSG:4326")
        west, south, east, north = geo_area.total_bounds
        buildings = clean_polygons(self.osm_fetcher.fetch((south, west, north, east)).to_crs(crs))
        # Score only complete buildings visible in the usable part of the crop.
        complete = buildings.geometry.covered_by(area)
        excluded = int((~complete).sum())
        buildings = clean_polygons(buildings.loc[complete].copy().to_crs("EPSG:4326"))
        metadata = {"imagery": self.sentinel_fetcher.metadata, "osm": self.osm_fetcher.metadata,
                    "boundary_or_mask_buildings_excluded": excluded,
                    "target": [target_lon, target_lat], "reference_scope": "complete OSM buildings of any roof colour"}
        return FetchedScene(scene_id, r, g, b, transform, crs, buildings, area, metadata, valid)

    def run(self, detector, scene):
        rgb = [np.where(scene.valid_mask, band, 0) if scene.valid_mask is not None else band for band in (scene.r, scene.g, scene.b)]
        detections = restrict_detections(detector.detect(*rgb, scene.transform, scene.raster_crs), scene)
        result = self.validator.validate(detections, scene.buildings, detector.name, scene.raster_crs)
        return PipelineRun(scene.scene_id, scene.r, scene.g, scene.b, scene.transform, scene.raster_crs,
                           detections, scene.buildings, result)
