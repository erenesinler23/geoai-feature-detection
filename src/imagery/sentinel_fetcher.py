import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import Window
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from pystac_client.stac_api_io import StacApiIO
from pyproj import Transformer


class SentinelFetcher:
    """Read a complete crop and check the local scene classification."""

    def __init__(self, search_bbox, date_range, cloud_cover_max, window_half_pixels,
                 scene_id=None, max_local_bad_fraction=0.1):
        self.search_bbox = search_bbox
        self.date_range = date_range
        self.cloud_cover_max = cloud_cover_max
        self.window_half_pixels = window_half_pixels
        self.scene_id = scene_id
        self.max_local_bad_fraction = max_local_bad_fraction
        self.metadata = {}

    def fetch(self, target_lon, target_lat):
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                            modifier=planetary_computer.sign_inplace, stac_io=StacApiIO(timeout=30, max_retries=2))
        args = {"collections": ["sentinel-2-l2a"]}
        if self.scene_id:
            args["ids"] = [self.scene_id]
        else:
            args.update(bbox=self.search_bbox, datetime=self.date_range,
                        query={"eo:cloud_cover": {"lt": self.cloud_cover_max}})
        items = sorted(catalog.search(**args).items(),
                       key=lambda i: (i.properties.get("eo:cloud_cover", 100), i.id))
        failures = []
        for item in items:
            try:
                with rasterio.open(item.assets["visual"].href) as src:
                    x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(target_lon, target_lat)
                    row, col = src.index(x, y)
                    half = self.window_half_pixels
                    window = Window(col - half, row - half, 2 * half, 2 * half)
                    if window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > src.width or window.row_off + window.height > src.height:
                        raise ValueError("The complete crop is outside this tile.")
                    rgb = src.read([1, 2, 3], window=window, masked=True)
                    if np.ma.getmaskarray(rgb).any():
                        raise ValueError("The crop contains missing pixels.")
                    transform, crs = src.window_transform(window), src.crs
                scl = np.zeros(rgb.shape[1:], dtype=np.uint8)
                with rasterio.open(item.assets["SCL"].href) as src:
                    with WarpedVRT(src, crs=crs, transform=transform, width=rgb.shape[2],
                                   height=rgb.shape[1], resampling=Resampling.nearest) as crop:
                        scl = crop.read(1)
                clear = np.isin(scl, [4, 5, 6, 7])
                bad_fraction = float(1 - clear.mean())
                if bad_fraction > self.max_local_bad_fraction:
                    raise ValueError(f"Local cloud or invalid fraction is {bad_fraction:.3f}.")
                self.valid_mask = clear
                self.scl = scl
                self.metadata = {"scene_id": item.id, "datetime": item.properties.get("datetime"),
                                 "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
                                 "local_bad_fraction": bad_fraction, "window": list(window.flatten()),
                                 "stac_item": item.to_dict()}
                return (*rgb.filled(0).astype(float), transform, str(crs), item.id)
            except (ValueError, rasterio.errors.RasterioError, KeyError) as exc:
                failures.append(f"{item.id}: {exc}")
        raise RuntimeError("No complete usable crop was found. " + "; ".join(failures[:3]))
