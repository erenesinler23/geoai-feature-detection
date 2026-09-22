import requests
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

DEFAULT_MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]


class OSMBuildingFetcher:
    def __init__(self, mirrors=None):
        self.mirrors = mirrors or DEFAULT_MIRRORS
        self.metadata = {}

    def _request(self, osm_bbox):
        south, west, north, east = osm_bbox
        query = f'[out:json][timeout:40];(way["building"]({south},{west},{north},{east});relation["building"]["type"="multipolygon"]({south},{west},{north},{east}););out body geom;'
        errors = []
        for mirror in self.mirrors:
            try:
                response = requests.post(mirror, data={"data": query}, timeout=55,
                                         headers={"User-Agent": "GeoAIResearch/2.0"})
                response.raise_for_status()
                data = response.json()
                if data.get("remark"):
                    raise ValueError(data["remark"])
                metadata = {"endpoint": mirror, "query": query,
                                 "osm_base": data.get("osm3s", {}).get("timestamp_osm_base")}
                return data, metadata
            except (requests.RequestException, ValueError) as exc:
                errors.append(str(exc))
        raise RuntimeError("Could not retrieve complete OSM data. " + "; ".join(errors))

    def fetch(self, osm_bbox):
        try:
            data, metadata = self._request(osm_bbox)
            self.metadata = metadata
        except RuntimeError as first_error:
            south, west, north, east = osm_bbox
            mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
            tiles = [(south, west, mid_lat, mid_lon), (south, mid_lon, mid_lat, east),
                     (mid_lat, west, north, mid_lon), (mid_lat, mid_lon, north, east)]
            elements, parts = {}, []
            for tile in tiles:
                data, metadata = self._request(tile)
                parts.append(metadata)
                for element in data["elements"]:
                    elements[(element["type"], element["id"])] = element
            data = {"elements": list(elements.values())}
            self.metadata = {"subqueries": parts, "first_attempt_error": str(first_error)}
        self.raw_data = data
        return self._parse(data)

    def _parse(self, data):
        rows, used_members = [], set()
        relations = [e for e in data["elements"] if e["type"] == "relation"]
        for element in relations:
            outer, inner = [], []
            for member in element.get("members", []):
                coords = [(p["lon"], p["lat"]) for p in member.get("geometry", [])]
                if len(coords) >= 2:
                    (inner if member.get("role") == "inner" else outer).append(LineString(coords))
            geom = unary_union(list(polygonize(unary_union(outer))))
            holes = unary_union(list(polygonize(unary_union(inner))))
            if not geom.is_empty:
                geom = geom.difference(holes)
                rows.append(self._row(element, geom))
                used_members.update(m["ref"] for m in element.get("members", []) if m.get("role") != "inner")
        for element in data["elements"]:
            if element["type"] != "way" or element["id"] in used_members:
                continue
            coords = [(p["lon"], p["lat"]) for p in element.get("geometry", [])]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                rows.append(self._row(element, Polygon(coords)))
        frame = gpd.GeoDataFrame(rows, columns=["osm_id", "roof_colour", "building", "geometry"], crs="EPSG:4326")
        frame.geometry = frame.geometry.make_valid()
        return frame.loc[~frame.geometry.is_empty & frame.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)

    @staticmethod
    def _row(element, geom):
        tags = element.get("tags", {})
        return {"osm_id": f'{element["type"]}/{element["id"]}', "roof_colour": tags.get("roof:colour"),
                "building": tags.get("building"), "geometry": geom}
