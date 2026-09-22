import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt


class ToySceneGenerator:
    """Create a synthetic RGB scene for local pipeline checks."""

    def __init__(self, origin_lon: float, origin_lat: float,
                 pixel_size: float, image_size: int):
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat
        self.pixel_size = pixel_size
        self.image_size = image_size

    def create(self, path: str) -> None:
        w = h = self.image_size
        red = np.random.randint(50, 150, (h, w), dtype=np.uint8)
        green = np.random.randint(80, 180, (h, w), dtype=np.uint8)
        blue = np.random.randint(60, 120, (h, w), dtype=np.uint8)
        red[20:35, 40:55], green[20:35, 40:55], blue[20:35, 40:55] = 200, 40, 30

        transform = from_origin(self.origin_lon, self.origin_lat,
                                 self.pixel_size, self.pixel_size)
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                            count=3, dtype=red.dtype, crs="EPSG:4326",
                            transform=transform) as dst:
            dst.write(red, 1)
            dst.write(green, 2)
            dst.write(blue, 3)

    def plot(self, path: str, out_path: str, title: str = "Toy satellite scene") -> None:
        with rasterio.open(path) as src:
            r, g, b = src.read(1), src.read(2), src.read(3)
        plt.figure(figsize=(5, 5))
        plt.imshow(np.dstack([r, g, b]))
        plt.title(title)
        plt.axis("off")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()

    def read_bands(self, path: str):
        with rasterio.open(path) as src:
            r = src.read(1).astype(float)
            g = src.read(2).astype(float)
            b = src.read(3).astype(float)
            return r, g, b, src.transform, str(src.crs)
