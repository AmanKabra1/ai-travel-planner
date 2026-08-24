"""
Nearby attractions discovery — all free, no API key required.

Sources:
  - Nominatim (OSM): city name → lat/lon
  - Overpass API (OSM): POIs within radius
  - Wikivoyage MediaWiki API: local food, culture, shopping tips
"""

import logging
import re
from math import atan2, cos, radians, sin, sqrt

import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "wandr-travel-planner/1.0 (portfolio-demo)"}
_TIMEOUT = 20


# ── Geocoding ─────────────────────────────────────────────────────────────────

def geocode_city(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a city name using Nominatim, or None on failure."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as exc:
        logger.warning("Nominatim geocode failed for '%s': %s", city, exc)
    return None


# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── Overpass API ──────────────────────────────────────────────────────────────

_CATEGORY_MAP = {
    "attraction":      "Tourist Attraction",
    "viewpoint":       "Viewpoint",
    "museum":          "Museum",
    "place_of_worship":"Temple / Shrine",
    "waterfall":       "Waterfall",
    "peak":            "Mountain / Peak",
    "marketplace":     "Market",
    "park":            "Park / Garden",
    "castle":          "Castle / Fort",
    "monument":        "Monument",
    "ruins":           "Historic Ruins",
    "beach":           "Beach",
    "cave":            "Cave",
    "hot_spring":      "Hot Spring",
    "theme_park":      "Theme Park",
    "zoo":             "Zoo / Wildlife",
    "art_gallery":     "Art Gallery",
}


def overpass_nearby(lat: float, lon: float, radius_m: int = 100_000) -> list[dict]:
    """Return POIs within radius_m metres of (lat, lon), sorted by distance."""
    r = radius_m
    query = f"""
[out:json][timeout:40];
(
  node(around:{r},{lat},{lon})["tourism"="attraction"]["name"];
  node(around:{r},{lat},{lon})["tourism"="viewpoint"]["name"];
  node(around:{r},{lat},{lon})["tourism"="museum"]["name"];
  node(around:{r},{lat},{lon})["tourism"="theme_park"]["name"];
  node(around:{r},{lat},{lon})["tourism"="zoo"]["name"];
  node(around:{r},{lat},{lon})["tourism"="art_gallery"]["name"];
  node(around:{r},{lat},{lon})["amenity"="place_of_worship"]["name"];
  node(around:{r},{lat},{lon})["natural"="waterfall"]["name"];
  node(around:{r},{lat},{lon})["natural"="peak"]["name"];
  node(around:{r},{lat},{lon})["natural"="beach"]["name"];
  node(around:{r},{lat},{lon})["natural"="cave_entrance"]["name"];
  node(around:{r},{lat},{lon})["natural"="hot_spring"]["name"];
  node(around:{r},{lat},{lon})["amenity"="marketplace"]["name"];
  node(around:{r},{lat},{lon})["leisure"="park"]["name"];
  node(around:{r},{lat},{lon})["historic"~"castle|monument|ruins"]["name"];
);
out body;
"""
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers=_HEADERS,
            timeout=40,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception as exc:
        logger.warning("Overpass query failed: %s", exc)
        return []

    pois: list[dict] = []
    seen_names: set[str] = set()

    for e in elements:
        tags = e.get("tags", {})
        name = tags.get("name", "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        raw_cat = (
            tags.get("tourism") or tags.get("natural") or
            tags.get("historic") or tags.get("amenity") or
            tags.get("leisure") or "attraction"
        )
        category = _CATEGORY_MAP.get(raw_cat, raw_cat.replace("_", " ").title())

        e_lat = float(e.get("lat", lat))
        e_lon = float(e.get("lon", lon))
        dist  = round(_haversine_km(lat, lon, e_lat, e_lon), 1)

        pois.append({
            "name":        name,
            "category":    category,
            "distance_km": dist,
            "lat":         e_lat,
            "lon":         e_lon,
            "website":     tags.get("website", ""),
            "description": tags.get("description", tags.get("note", "")),
            "opening":     tags.get("opening_hours", ""),
            "wikidata":    tags.get("wikidata", ""),
        })

    pois.sort(key=lambda p: p["distance_km"])
    return pois


def bucket_pois(pois: list[dict]) -> dict[str, list[dict]]:
    """Group POIs into distance buckets."""
    return {
        "within_10km":  [p for p in pois if p["distance_km"] <= 10],
        "10_to_30km":   [p for p in pois if 10 < p["distance_km"] <= 30],
        "30_to_50km":   [p for p in pois if 30 < p["distance_km"] <= 50],
        "50_to_100km":  [p for p in pois if 50 < p["distance_km"] <= 100],
    }


# ── Wikivoyage ────────────────────────────────────────────────────────────────

_WIKIVOYAGE_SECTIONS = ["eat", "buy", "do", "see", "understand", "drink"]


def _clean_wikitext(text: str) -> str:
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"==+[^=]*==+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def wikivoyage_local_tips(city: str) -> dict[str, str]:
    """Fetch Eat / Buy / Culture sections from Wikivoyage. Returns {section: text}."""
    base = "https://en.wikivoyage.org/w/api.php"
    try:
        r = requests.get(
            base,
            params={"action": "parse", "page": city,
                    "prop": "sections", "format": "json"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        sections = r.json().get("parse", {}).get("sections", [])
    except Exception as exc:
        logger.warning("Wikivoyage section list failed for '%s': %s", city, exc)
        return {}

    matched: dict[str, str] = {}
    for s in sections:
        line = s.get("line", "").lower()
        idx  = s.get("index", "")
        key  = next((k for k in _WIKIVOYAGE_SECTIONS if k in line), None)
        if key and key not in matched:
            try:
                r2 = requests.get(
                    base,
                    params={"action": "parse", "page": city,
                            "prop": "wikitext", "section": idx, "format": "json"},
                    headers=_HEADERS, timeout=_TIMEOUT,
                )
                r2.raise_for_status()
                raw = r2.json().get("parse", {}).get("wikitext", {}).get("*", "")
                matched[key] = _clean_wikitext(raw)[:1800]
            except Exception:
                pass
        if len(matched) >= 4:
            break

    return matched


def format_pois_text(pois: list[dict], max_n: int = 15) -> str:
    """Format a POI list as plain text for LLM input."""
    if not pois:
        return "None found in this radius."
    lines = []
    for p in pois[:max_n]:
        line = f"• {p['name']} [{p['category']}] — {p['distance_km']} km"
        if p.get("description"):
            line += f" | {p['description'][:120]}"
        if p.get("opening"):
            line += f" | Hours: {p['opening']}"
        lines.append(line)
    return "\n".join(lines)
