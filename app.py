import os
import re
import time
from threading import Lock
from datetime import datetime, timezone
from uuid import uuid4
from src.archive import save_run, write_json
from flask import Flask, render_template, request, send_from_directory
import config
from src.geocoding import geocode
from src.imagery.sentinel_fetcher import SentinelFetcher
from src.vectors.osm_fetcher import OSMBuildingFetcher
from src.validation.validator import DetectionValidator
from src.detectors.naive_color import NaiveColorDetector
from src.detectors.owlvit import OwlViTDetector
from src.detectors.hybrid import fuse_detections
from src.pipeline import Pipeline
from src.plotting import save_scene_plot, save_validation_plot

app = Flask(__name__)
STARTUP_VERSION = int(time.time())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR_ABS = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR_ABS, exist_ok=True)

PRESETS = {
    "ankara": {
        "target_lon": 32.862617, "target_lat": 39.931789,
        "search_bbox": [32.842617, 39.911789, 32.882617, 39.951789],
        "red_min": 120, "red_minus_green_min": 25, "red_minus_blue_min": 25,
        "confidence_threshold": 0.01,
    },
    "marrakech": {
        "target_lon": -7.9811, "target_lat": 31.6295,
        "search_bbox": [-8.00, 31.61, -7.96, 31.65],
        "red_min": 120, "red_minus_green_min": 25, "red_minus_blue_min": 25,
        "confidence_threshold": 0.01,
    },
    "nottingham": {
        "target_lon": -1.1949, "target_lat": 52.9408,
        "search_bbox": [-1.22, 52.92, -1.17, 52.96],
        "red_min": 140, "red_minus_green_min": 40, "red_minus_blue_min": 40,
        "confidence_threshold": 0.02,
    },
}

RESULTS_CACHE = {}
RECENT_SEARCHES = []
MAX_RECENT = 8
RUN_LOCK = Lock()




def get_owlvit_detector(text_queries, confidence_threshold):
    return OwlViTDetector(text_queries, confidence_threshold)


def make_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "location"


def run_full_pipeline(target_lon, target_lat, search_bbox, red_min,
                       red_minus_green_min, red_minus_blue_min, confidence_threshold, slug):
    started = time.perf_counter()
    osm_bbox = (search_bbox[1], search_bbox[0], search_bbox[3], search_bbox[2])

    sentinel_fetcher = SentinelFetcher(
        search_bbox=search_bbox, date_range=config.DATE_RANGE,
        cloud_cover_max=config.CLOUD_COVER_MAX, window_half_pixels=config.WINDOW_HALF_PIXELS,
    )
    osm_fetcher = OSMBuildingFetcher()
    validator = DetectionValidator(config.MAX_SIZE_M, config.IOU_THRESHOLD)
    pipeline = Pipeline(sentinel_fetcher, osm_fetcher, validator)

    scene = pipeline.fetch(target_lon, target_lat, osm_bbox)
    fetched = time.perf_counter()

    naive = NaiveColorDetector(red_min, red_minus_green_min, red_minus_blue_min)
    owlvit = get_owlvit_detector(config.TEXT_QUERIES, confidence_threshold)

    loaded = time.perf_counter()
    naive_run = pipeline.run(naive, scene)
    owlvit_run = pipeline.run(owlvit, scene)

    hybrid_detections = fuse_detections(naive_run.detections, owlvit_run.detections, config.MAX_SIZE_M)
    hybrid_result = validator.validate(hybrid_detections, scene.buildings, "hybrid", scene.raster_crs)

    evaluated = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    slug = f"{slug}/{run_id}"
    location_dir_abs = os.path.join(OUTPUTS_DIR_ABS, slug)
    os.makedirs(location_dir_abs, exist_ok=True)
    url_prefix = f"outputs/{slug}"

    scene_path = os.path.join(location_dir_abs, "scene.png")
    naive_path = os.path.join(location_dir_abs, "naive.png")
    owlvit_path = os.path.join(location_dir_abs, "owlvit.png")
    hybrid_path = os.path.join(location_dir_abs, "hybrid.png")

    save_scene_plot(scene.r, scene.g, scene.b, f"Real satellite scene, {scene.scene_id}", scene_path)
    save_validation_plot(scene.r, scene.g, scene.b, naive_run.result.filtered_detections,
                          naive_run.result.matched_ids, scene.transform, scene.raster_crs,
                          "Naive detector", naive_path)
    save_validation_plot(scene.r, scene.g, scene.b, owlvit_run.result.filtered_detections,
                          owlvit_run.result.matched_ids, scene.transform, scene.raster_crs,
                          "OWL-ViT", owlvit_path)
    save_validation_plot(scene.r, scene.g, scene.b, hybrid_result.filtered_detections,
                          hybrid_result.matched_ids, scene.transform, scene.raster_crs,
                          "Hybrid", hybrid_path)

    plotted = time.perf_counter()
    write_json(os.path.join(location_dir_abs, "owlvit_raw.json"), owlvit.last_raw)
    save_run(location_dir_abs, scene, [
        ("naive", naive_run.detections, naive_run.result),
        ("owlvit", owlvit_run.detections, owlvit_run.result),
        ("hybrid", hybrid_detections, hybrid_result),
    ], {"red_min": red_min, "red_minus_green_min": red_minus_green_min,
        "red_minus_blue_min": red_minus_blue_min, "confidence_threshold": confidence_threshold,
        "text_queries": config.TEXT_QUERIES, "max_size_m": config.MAX_SIZE_M,
        "iou_threshold": config.IOU_THRESHOLD, "model_revision": owlvit.model_revision,
        "date_range": config.DATE_RANGE, "cloud_cover_max": config.CLOUD_COVER_MAX,
        "max_local_bad_fraction": sentinel_fetcher.max_local_bad_fraction})

    finished = time.perf_counter()
    print(f"Run completed in {finished - started:.1f}s: "
          f"data {fetched - started:.1f}s, model setup {loaded - fetched:.1f}s, "
          f"detection and matching {evaluated - loaded:.1f}s, "
          f"plots {plotted - evaluated:.1f}s, archive {finished - plotted:.1f}s.", flush=True)
    return {
        "scene_id": scene.scene_id,
        "scene_image": f"{url_prefix}/scene.png",
        "results": [
            {"name": "Naive detector", "image": f"{url_prefix}/naive.png", "result": naive_run.result},
            {"name": "OWL-ViT", "image": f"{url_prefix}/owlvit.png", "result": owlvit_run.result},
            {"name": "Hybrid", "image": f"{url_prefix}/hybrid.png", "result": hybrid_result},
        ],
    }


def update_recent(cache_key, display_name):
    if cache_key in PRESETS:
        return
    RECENT_SEARCHES[:] = [r for r in RECENT_SEARCHES if r["key"] != cache_key]
    RECENT_SEARCHES.insert(0, {"key": cache_key, "display_name": display_name})
    del RECENT_SEARCHES[MAX_RECENT:]


@app.context_processor
def inject_version():
    return {"version": STARTUP_VERSION}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES)


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Show saved plots in the browser."""
    return send_from_directory(OUTPUTS_DIR_ABS, filename)


@app.route("/run", methods=["POST"])
def run():
    preset_name = request.form.get("preset", "").strip()
    place_name = request.form.get("place_name", "").strip()
    force = request.form.get("force", "") == "true"

    cache_key = preset_name.lower() if preset_name else place_name.lower()
    if not cache_key:
        return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES,
                                error="I need either a place name or a preset.")

    if not force and cache_key in RESULTS_CACHE:
        cached = RESULTS_CACHE[cache_key]
        return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES,
                                display_name=cached["display_name"], output=cached["output"],
                                cache_key=cache_key, from_cache=True)

    with RUN_LOCK:
        if not force and cache_key in RESULTS_CACHE:
            cached = RESULTS_CACHE[cache_key]
            return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES,
                                   display_name=cached["display_name"], output=cached["output"],
                                   cache_key=cache_key, from_cache=True)
        try:
            if cache_key in PRESETS:
                preset = PRESETS[cache_key]
                display_name = cache_key.title()
                slug = cache_key
                output = run_full_pipeline(
                    preset["target_lon"], preset["target_lat"], preset["search_bbox"],
                    preset["red_min"], preset["red_minus_green_min"], preset["red_minus_blue_min"],
                    preset["confidence_threshold"], slug,
                )
            else:
                location = geocode(place_name or preset_name)
                display_name = location["display_name"]
                slug = make_slug(cache_key)
                output = run_full_pipeline(
                    location["lon"], location["lat"], location["search_bbox"],
                    config.RED_MIN, config.RED_MINUS_GREEN_MIN, config.RED_MINUS_BLUE_MIN,
                    config.CONFIDENCE_THRESHOLD, slug,
                )

            RESULTS_CACHE[cache_key] = {"display_name": display_name, "output": output}
            update_recent(cache_key, display_name)

            return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES,
                                    display_name=display_name, output=output, cache_key=cache_key,
                                    from_cache=False)

        except Exception as exc:
            return render_template("index.html", presets=list(PRESETS.keys()), recent=RECENT_SEARCHES,
                                    error=str(exc))

if __name__ == "__main__":
    port = int(os.environ.get("GEOAI_PORT", "5055"))
    print(f"Open http://127.0.0.1:{port} in your browser.", flush=True)
    print("The actual OWL ViT device will be reported when the first model loads.", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
