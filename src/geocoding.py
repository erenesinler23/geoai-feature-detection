import requests


def geocode(place_name: str, bbox_half_deg: float = 0.02) -> dict:
    """Look up a place with Nominatim and build a search box."""
    headers = {"User-Agent": "geoai-feature-detection/1.0"}
    params = {"q": place_name, "format": "json", "limit": 1}
    response = requests.get("https://nominatim.openstreetmap.org/search",
                             params=params, headers=headers, timeout=10)
    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"I couldn't find a location for '{place_name}'. Try a more specific name.")

    result = results[0]
    lon, lat = float(result["lon"]), float(result["lat"])

    return {
        "lon": lon,
        "lat": lat,
        "search_bbox": [lon - bbox_half_deg, lat - bbox_half_deg,
                         lon + bbox_half_deg, lat + bbox_half_deg],
        "display_name": result["display_name"],
    }
