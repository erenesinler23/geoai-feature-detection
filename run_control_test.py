import os
from pathlib import Path
import config
from src.detectors.owlvit import OwlViTDetector
from src.control_test import run_control_test

FIXTURES_DIR = Path("tests/fixtures")
OUTPUT_DIR = Path("outputs/control_tests")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

detector = OwlViTDetector(config.TEXT_QUERIES, config.CONFIDENCE_THRESHOLD)

photos = sorted(p for p in FIXTURES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
if not photos:
    raise SystemExit(f"I found no photos in {FIXTURES_DIR}. Add one and try again.")

for photo in photos:
    print(f"Running control test on {photo.name}")
    out_path = OUTPUT_DIR / f"{photo.stem}_result.png"

    result = run_control_test(
        image_path=str(photo),
        detector=detector,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        max_plausible_fraction=0.25,
        out_path=str(out_path),
    )

    print(f"  total detections: {result['total_detections']}")
    print(f"  plausible: {result['plausible_count']}")
    print(f"  saved to {out_path}")