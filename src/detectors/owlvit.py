import os
import torch
from PIL import Image
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon
import geopandas as gpd
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from .base import Detector


class OwlViTDetector(Detector):
    """Load the model once and use an available GPU."""

    _processor = None
    _model = None
    _device = None
    _revision = None

    def __init__(self, text_queries: list[str], confidence_threshold: float):
        self.text_queries = text_queries
        self.confidence_threshold = confidence_threshold
        if OwlViTDetector._model is None:
            if torch.backends.mps.is_available():
                OwlViTDetector._device = torch.device("mps")
            elif torch.cuda.is_available():
                OwlViTDetector._device = torch.device("cuda")
            else:
                OwlViTDetector._device = torch.device("cpu")

            revision = os.environ.get("OWL_MODEL_REVISION", "cbc355fb364588351c5d51c7f74465e8e7ec6f72")
            OwlViTDetector._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", revision=revision)
            OwlViTDetector._model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32", revision=revision
            ).to(OwlViTDetector._device)

            OwlViTDetector._model.eval()
            OwlViTDetector._revision = OwlViTDetector._model.config._commit_hash
            device = next(OwlViTDetector._model.parameters()).device
            label = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}[device.type]
            print(f"OWL ViT loaded on {label} ({device}).", flush=True)
        self.model_revision = OwlViTDetector._revision
        self.last_raw = None

    @property
    def name(self) -> str:
        return "owlvit"

    def detect_pixels(self, image: Image.Image):
        """Return pixel boxes and scores before geographic conversion."""
        processor, model = OwlViTDetector._processor, OwlViTDetector._model
        inputs = processor(text=self.text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(OwlViTDetector._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(OwlViTDetector._device)

        if hasattr(processor, "post_process_object_detection"):
            results = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target_sizes
            )[0]
        else:
            results = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0, text_labels=[self.text_queries],
            )[0]
            if "text_labels" in results and "labels" not in results:
                results["labels"] = [self.text_queries.index(t) for t in results["text_labels"]]

        # Move the output back so the rest of the code can use it.
        results["boxes"] = results["boxes"].cpu()
        results["scores"] = results["scores"].cpu()
        if hasattr(results["labels"], "cpu"):
            results["labels"] = results["labels"].cpu()
        self.last_raw = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in results.items()}
        return results

    def detect(self, r, g, b, transform, raster_crs) -> gpd.GeoDataFrame:
        rgb = np.dstack([r, g, b]).astype(np.uint8)
        image = Image.fromarray(rgb)
        raw = self.detect_pixels(image)

        keep = raw["scores"] >= self.confidence_threshold
        pt_boxes = raw["boxes"][keep]
        scores = raw["scores"][keep]

        to_lonlat = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        boxes, score_values = [], []
        for pt_box, score in zip(pt_boxes, scores):
            x_min, y_min, x_max, y_max = pt_box.tolist()
            x_min, x_max = np.clip([x_min, x_max], 0, image.width)
            y_min, y_max = np.clip([y_min, y_max], 0, image.height)
            if x_max <= x_min or y_max <= y_min:
                continue
            corners = [transform * (x, y) for x, y in
                       [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]]
            boxes.append(Polygon([to_lonlat.transform(x, y) for x, y in corners]))
            score_values.append(float(score))

        return gpd.GeoDataFrame(
            {"detection_id": range(1, len(boxes) + 1), "score": score_values, "geometry": boxes},
            crs="EPSG:4326",
        )
