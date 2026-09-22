from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import geopandas as gpd
from scipy.optimize import linear_sum_assignment
from pyproj import CRS


def metric_frame(frame, crs=None):
    if frame.crs is None:
        raise ValueError("Geometry needs a coordinate reference system.")
    if crs is None:
        crs = frame.estimate_utm_crs() if len(frame) else "EPSG:3857"
    projected = CRS(crs)
    if not projected.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in projected.axis_info):
        raise ValueError("Evaluation needs a projected CRS measured in metres.")
    return frame.to_crs(crs)


def plausible_mask(frame, max_size_m):
    bounds = frame.geometry.bounds
    return ((bounds.maxx - bounds.minx) <= max_size_m) & ((bounds.maxy - bounds.miny) <= max_size_m)


@dataclass
class ValidationResult:
    detector_name: str
    n_total: int
    n_oversized_excluded: int
    n_matched: int
    precision: float | None
    false_discovery_rate: float | None
    n_buildings_found: int
    n_buildings_total: int
    recall: float | None
    f1: float | None
    iou_threshold: float
    matched_ids: set = field(default_factory=set)
    filtered_detections: gpd.GeoDataFrame = None
    matches: list = field(default_factory=list)
    n_raw: int = 0


class DetectionValidator:
    """Each prediction can match one reference building."""

    def __init__(self, max_size_m=500.0, iou_threshold=0.25):
        if max_size_m <= 0 or not 0 < iou_threshold <= 1:
            raise ValueError("Size must be positive and IoU must be in (0, 1].")
        self.max_size_m = max_size_m
        self.iou_threshold = iou_threshold

    def validate(self, detections, buildings, detector_name, metric_crs=None):
        if "detection_id" not in detections or not detections.detection_id.is_unique:
            raise ValueError("Each detection needs a unique detection_id.")
        for frame in (detections, buildings):
            if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
                raise ValueError("Evaluation received empty or invalid geometry.")
        source = buildings if len(buildings) else detections
        crs = metric_crs or (source.estimate_utm_crs() if len(source) else "EPSG:3857")
        d = metric_frame(detections, crs).reset_index(drop=True)
        b = metric_frame(buildings, crs).reset_index(drop=True)
        mask = plausible_mask(d, self.max_size_m)
        filtered = detections.loc[mask.to_numpy()].copy().reset_index(drop=True)
        d = d.loc[mask].reset_index(drop=True)
        matches = []
        if len(d) and len(b):
            # Reward valid matches first, then prefer the larger overlap.
            scores = np.zeros((len(d), len(b)))
            for i, geom in enumerate(d.geometry):
                for j in b.sindex.query(geom, predicate="intersects"):
                    other = b.geometry.iloc[j]
                    intersection = geom.intersection(other).area
                    union = geom.area + other.area - intersection
                    scores[i, j] = intersection / union if union else 0
            valid = scores >= self.iou_threshold
            reward = valid * (min(len(d), len(b)) + 1 + scores)
            rows, cols = linear_sum_assignment(reward, maximize=True)
            for i, j in zip(rows, cols):
                if valid[i, j]:
                    matches.append({"detection_id": int(d.detection_id.iloc[i]),
                                    "reference_index": int(j),
                                    "reference_id": str(b.iloc[j].get("osm_id", j)),
                                    "iou": float(scores[i, j])})
        tp, count, refs = len(matches), len(d), len(b)
        precision = 100 * tp / count if count else None
        recall = 100 * tp / refs if refs else None
        f1 = 200 * tp / (count + refs) if count + refs else None
        return ValidationResult(detector_name, count, int((~mask).sum()), tp,
                                precision, 100 - precision if precision is not None else None,
                                tp, refs, recall, f1, self.iou_threshold,
                                {m["detection_id"] for m in matches}, filtered, matches, len(detections))
