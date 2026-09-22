import argparse
import json
from pathlib import Path
import geopandas as gpd
import rasterio
from src.validation.validator import DetectionValidator
from src.archive import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--iou", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    args = parser.parse_args()
    manifest = json.loads((args.folder / "manifest.json").read_text())
    buildings = gpd.read_file(args.folder / "buildings.geojson")
    with rasterio.open(args.folder / "image.tif") as image:
        crs = image.crs
    results = []
    for name in ["naive", "owlvit", "hybrid"]:
        detections = gpd.read_file(args.folder / f"{name}_detections.geojson")
        if "detection_id" not in detections and detections.empty:
            detections["detection_id"] = []
        for threshold in args.iou:
            result = DetectionValidator(manifest["settings"]["max_size_m"], threshold).validate(detections, buildings, name, crs)
            results.append({"detector": name, "iou": threshold, "precision": result.precision,
                            "recall": result.recall, "f1": result.f1, "matches": result.n_matched})
    write_json(args.folder / "sensitivity.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
