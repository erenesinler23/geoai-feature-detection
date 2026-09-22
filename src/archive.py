from datetime import datetime, timezone
from pathlib import Path
from dataclasses import fields
import hashlib
import importlib.metadata
import json
import platform
import zipfile
from urllib.parse import urlsplit, urlunsplit
import rasterio
import geopandas as gpd
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str))


def source_files(root):
    allowed = {".py", ".html", ".txt", ".json"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix in allowed]
    # Only include project code, never installed libraries or saved runs.
    for name in ("src", "templates", "tests"):
        folder = root / name
        if folder.is_dir():
            files.extend(p for p in folder.rglob("*")
                         if p.is_file() and p.suffix in allowed
                         and "__pycache__" not in p.parts and not p.is_symlink())
    return sorted(files)


def save_run(folder, scene, runs, settings):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with rasterio.open(folder / "image.tif", "w", driver="GTiff", height=scene.r.shape[0],
                       width=scene.r.shape[1], count=3, dtype="uint8", crs=scene.raster_crs,
                       transform=scene.transform) as dst:
        dst.write(np.stack([scene.r, scene.g, scene.b]).astype("uint8"))
        dst.write_mask(scene.valid_mask.astype("uint8") * 255)
    (folder / "buildings.geojson").write_text(scene.buildings.to_json())
    area = gpd.GeoDataFrame(geometry=[scene.valid_area], crs=scene.raster_crs).to_crs("EPSG:4326")
    (folder / "valid_area.geojson").write_text(area.to_json())
    results = []
    for name, detections, result in runs:
        (folder / f"{name}_detections.geojson").write_text(detections.to_json())
        results.append({f.name: getattr(result, f.name) for f in fields(result)
                        if f.name not in {"filtered_detections", "matched_ids"}})
    write_json(folder / "results.json", results)
    packages = ["numpy", "scipy", "geopandas", "shapely", "rasterio", "torch", "transformers", "pystac-client", "planetary-computer", "flask", "pyproj"]
    versions = {p: importlib.metadata.version(p) for p in packages}
    root = Path(__file__).resolve().parents[1]
    files = source_files(root)
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, source.relative_to(root))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in files if p.suffix == ".py"}
    imagery = scene.metadata.get("imagery", {})
    for asset in imagery.get("stac_item", {}).get("assets", {}).values():
        parts = urlsplit(asset["href"])
        asset["href"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    write_json(folder / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "scene_id": scene.scene_id, "settings": settings, "metadata": scene.metadata,
               "python": platform.python_version(), "dependencies": versions, "source_sha256": sources})
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name != "checksums.json"}
    write_json(folder / "checksums.json", hashes)
