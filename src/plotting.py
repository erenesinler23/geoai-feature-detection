import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer


def geometry_to_pixel_bounds(geometry, transform, raster_crs, to_map=None):
    from shapely.ops import transform as map_geometry
    if to_map is None:
        to_map = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    projected = map_geometry(to_map.transform, geometry)
    inverse = ~transform
    pixels = [inverse * point for point in projected.envelope.exterior.coords]
    xs, ys = zip(*pixels)
    return min(xs), min(ys), max(xs), max(ys)


def save_scene_plot(r, g, b, title, out_path):
    rgb = np.dstack([r, g, b]).astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_validation_plot(r, g, b, detections, matched_ids, transform, raster_crs, title, out_path):
    rgb = np.dstack([r, g, b]).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)

    to_map = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    for _, row in detections.iterrows():
        col_min, row_min, col_max, row_max = geometry_to_pixel_bounds(
            row.geometry, transform, raster_crs, to_map
        )
        color = "lime" if row["detection_id"] in matched_ids else "red"
        ax.add_patch(plt.Rectangle((col_min, row_min), col_max - col_min, row_max - row_min,
                                    fill=False, edgecolor=color, linewidth=1.5))

    ax.set_title(title)
    ax.axis("off")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
