import os
from src.imagery.toy_generator import ToySceneGenerator
from src.vectors.toy_generator import ToyBuildingGenerator
from src.detectors.naive_color import NaiveColorDetector
from src.validation.validator import DetectionValidator
from src.plotting import save_validation_plot

os.makedirs("outputs", exist_ok=True)

scene_gen = ToySceneGenerator(origin_lon=-1.20, origin_lat=52.945,
                               pixel_size=0.0001, image_size=100)
scene_gen.create("outputs/demo_toy_scene.tif")
scene_gen.plot("outputs/demo_toy_scene.tif", "outputs/demo_toy_plot.png")
print("I created and plotted the toy scene.")

building_gen = ToyBuildingGenerator(origin_lon=-1.20, origin_lat=52.945)
buildings = building_gen.create()
print(f"I created {len(buildings)} toy buildings.")

r, g, b, transform, crs = scene_gen.read_bands("outputs/demo_toy_scene.tif")

detector = NaiveColorDetector(red_min=160, red_minus_green_min=60, red_minus_blue_min=60)
detections = detector.detect(r, g, b, transform, crs)
print(f"The detector found {len(detections)} region(s).")

validator = DetectionValidator(max_size_m=500)
result = validator.validate(detections, buildings, detector.name)
print(f"False discovery rate: {result.false_discovery_rate:.1f}%")
print(f"Recall: {result.n_buildings_found}/{result.n_buildings_total} ({result.recall:.1f}%)")

save_validation_plot(r, g, b, result.filtered_detections, result.matched_ids,
                      transform, crs, "Toy detection overlay", "outputs/demo_toy_validated.png")
print("Saved outputs/demo_toy_validated.png")
