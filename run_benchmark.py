import argparse
import json
from pathlib import Path
from app import run_full_pipeline
from src.archive import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmark.json")
    parser.add_argument("--locations", nargs="*")
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text())
    report = []
    for loc in spec["locations"]:
        if args.locations and loc["name"] not in args.locations:
            continue
        lon, lat = loc["lon"], loc["lat"]
        print("Running", loc["name"], flush=True)
        try:
            result = run_full_pipeline(lon, lat, [lon-.02, lat-.02, lon+.02, lat+.02],
                                       **spec["settings"], slug=loc["name"])
            folder = str(Path(result["scene_image"]).parent)
            report.append({"location": loc["name"], "status": "complete", "folder": folder})
            print(loc["name"], folder, flush=True)
        except Exception as exc:
            report.append({"location": loc["name"], "status": "failed", "error": str(exc)})
            print(loc["name"], str(exc), flush=True)
        write_json("outputs/benchmark_status.json", report)


if __name__ == "__main__":
    main()
