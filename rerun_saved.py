import argparse
import json
from pathlib import Path
import geopandas as gpd
import rasterio
from src.pipeline import FetchedScene, Pipeline
from src.validation.validator import DetectionValidator
from src.detectors.naive_color import NaiveColorDetector
from src.detectors.owlvit import OwlViTDetector
from src.detectors.hybrid import fuse_detections
from src.archive import save_run, write_json
from src.plotting import save_scene_plot, save_validation_plot


def rerun(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    settings = manifest["settings"]
    with rasterio.open(source / "image.tif") as image:
        rgb, transform, crs, mask = image.read().astype(float), image.transform, image.crs, image.dataset_mask() > 0
    buildings = gpd.read_file(source / "buildings.geojson")
    area = gpd.read_file(source / "valid_area.geojson").to_crs(crs).geometry.iloc[0]
    scene = FetchedScene(manifest["scene_id"], *rgb, transform, str(crs), buildings, area, manifest["metadata"], mask)
    pipeline = Pipeline(None, None, DetectionValidator(settings["max_size_m"], settings["iou_threshold"]))
    naive = NaiveColorDetector(settings["red_min"], settings["red_minus_green_min"], settings["red_minus_blue_min"])
    owlvit = OwlViTDetector(settings["text_queries"], settings["confidence_threshold"])
    runs = []
    for detector in [naive, owlvit]:
        run = pipeline.run(detector, scene)
        runs.append((detector.name, run.detections, run.result))
    hybrid = fuse_detections(runs[0][1], runs[1][1], settings["max_size_m"])
    runs.append(("hybrid", hybrid, pipeline.validator.validate(hybrid, buildings, "hybrid", crs)))
    settings["model_revision"] = owlvit.model_revision
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "owlvit_raw.json", owlvit.last_raw)
    save_scene_plot(*rgb, f"Satellite crop {scene.scene_id}", destination / "scene.png")
    for name, detections, result in runs:
        save_validation_plot(*rgb, result.filtered_detections, result.matched_ids, transform, crs,
                             name, destination / f"{name}.png")
    save_run(destination, scene, runs, settings)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(rerun(args.source, args.destination))
