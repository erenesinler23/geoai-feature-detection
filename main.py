import config
from app import run_full_pipeline


if __name__ == "__main__":
    output = run_full_pipeline(config.TARGET_LON, config.TARGET_LAT, config.SEARCH_BBOX,
                               config.RED_MIN, config.RED_MINUS_GREEN_MIN,
                               config.RED_MINUS_BLUE_MIN, config.CONFIDENCE_THRESHOLD,
                               config.LOCATION_NAME)
    for item in output["results"]:
        result = item["result"]
        print(item["name"], "precision", result.precision, "recall", result.recall, "F1", result.f1)
