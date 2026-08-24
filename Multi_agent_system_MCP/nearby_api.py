"""
Nearby attractions discovery — all free, no API key required.

Sources:
  - Nominatim (OSM): city/village/hamlet → lat/lon, full address context
  - Overpass API (OSM): POIs within radius, auto-expands if sparse
  - Wikivoyage MediaWiki API: local food, culture, shopping tips
    (falls back to parent region if small place has no page)
"""

import logging
import re
from math import atan2, cos, radians, sin, sqrt

import requests

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "wandr-travel-planner/1.0 (portfolio-demo)"}
_TIMEOUT = 20


# ── Geocoding ─────────────────────────────────────────────────────────────────

def geocode_city(city: str) -> tuple[float, float, dict] | None:
    """Return (lat, lon, meta) for any settlement — city, town, village, hamlet.

    meta dict includes: display_name, region, country, place_type
    Returns None on failure.
    """
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q":              city,
                "format":         "jsonv2",
                "limit":          1,
                "addressdetails": 1,
            },
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            hit     = data[0]
            addr    = hit.get("address", {})
            region  = (
                addr.get("state") or addr.get("county") or
                addr.get("region") or addr.get("province") or ""
            )
            country = addr.get("country", "")
            meta = {
                "display_name": hit.get("display_name", city),
                "region":       region,
                "country":      country,
                "place_type":   hit.get("type", ""),
            }
            return float(hit["lat"]), float(hit["lon"]), meta
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
    "attraction":       "Tourist Attraction",
    "viewpoint":        "Viewpoint",
    "museum":           "Museum",
    "place_of_worship": "Temple / Shrine",
    "waterfall":        "Waterfall",
    "peak":             "Mountain / Peak",
    "marketplace":      "Market",
    "park":             "Park / Garden",
    "castle":           "Castle / Fort",
    "monument":         "Monument",
    "ruins":            "Historic Ruins",
    "beach":            "Beach",
    "cave_entrance":    "Cave",
    "hot_spring":       "Hot Spring",
    "theme_park":       "Theme Park",
    "zoo":              "Zoo / Wildlife",
    "art_gallery":      "Art Gallery",
    "archaeological_site": "Archaeological Site",
    "historic":         "Historic Site",
    "nature_reserve":   "Nature Reserve",
    "dam":              "Dam / Reservoir",
    "river":            "River / Lake",
    "forest":           "Forest",
    "national_park":    "National Park",
}


def _run_overpass(lat: float, lon: float, r: int) -> list:
    query = f"""
[out:json][timeout:45];
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
  node(around:{r},{lat},{lon})["natural"="wood"]["name"];
  node(around:{r},{lat},{lon})["amenity"="marketplace"]["name"];
  node(around:{r},{lat},{lon})["leisure"="park"]["name"];
  node(around:{r},{lat},{lon})["leisure"="nature_reserve"]["name"];
  node(around:{r},{lat},{lon})["historic"~"castle|monument|ruins|archaeological_site"]["name"];
  node(around:{r},{lat},{lon})["waterway"="dam"]["name"];
  way(around:{r},{lat},{lon})["tourism"="attraction"]["name"];
  way(around:{r},{lat},{lon})["historic"~"castle|monument|ruins|archaeological_site"]["name"];
  way(around:{r},{lat},{lon})["leisure"="nature_reserve"]["name"];
  way(around:{r},{lat},{lon})["boundary"="national_park"]["name"];
);
out center;
"""
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=_HEADERS,
        timeout=50,
    )
    resp.raise_for_status()
    return resp.json().get("elements", [])


def overpass_nearby(lat: float, lon: float, radius_m: int = 100_000) -> list[dict]:
    """Return POIs within radius_m metres, auto-expanding if results are sparse.

    For small/remote places with few POIs, expands the search to 150 km so the
    output is always meaningful.
    """
    pois = _parse_overpass(_run_overpass(lat, lon, radius_m), lat, lon)

    # If very sparse (small/remote place), expand radius once
    if len(pois) < 8 and radius_m < 150_000:
        logger.info("Sparse POIs (%d) — expanding to 150 km", len(pois))
        try:
            more = _parse_overpass(_run_overpass(lat, lon, 150_000), lat, lon)
            if len(more) > len(pois):
                pois = more
        except Exception:
            pass

    pois.sort(key=lambda p: p["distance_km"])
    return pois


def _parse_overpass(elements: list, center_lat: float, center_lon: float) -> list[dict]:
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
            tags.get("leisure") or tags.get("waterway") or "attraction"
        )
        category = _CATEGORY_MAP.get(raw_cat, raw_cat.replace("_", " ").title())

        # Ways have a "center" key; nodes have lat/lon directly
        if e.get("type") == "way":
            c = e.get("center", {})
            e_lat = float(c.get("lat", center_lat))
            e_lon = float(c.get("lon", center_lon))
        else:
            e_lat = float(e.get("lat", center_lat))
            e_lon = float(e.get("lon", center_lon))

        dist = round(_haversine_km(center_lat, center_lon, e_lat, e_lon), 1)

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

    return pois


def bucket_pois(pois: list[dict]) -> dict[str, list[dict]]:
    """Group POIs into distance buckets."""
    return {
        "within_10km": [p for p in pois if p["distance_km"] <= 10],
        "10_to_30km":  [p for p in pois if 10 < p["distance_km"] <= 30],
        "30_to_50km":  [p for p in pois if 30 < p["distance_km"] <= 50],
        "50_to_100km": [p for p in pois if 50 < p["distance_km"] <= 100],
    }


# ── Wikivoyage ────────────────────────────────────────────────────────────────

_WIKIVOYAGE_SECTIONS = ["eat", "buy", "do", "see", "understand", "drink"]
_WV_BASE = "https://en.wikivoyage.org/w/api.php"


def _clean_wikitext(text: str) -> str:
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"==+[^=]*==+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_wikivoyage(page_title: str) -> dict[str, str]:
    """Fetch Eat / Buy / Culture sections for a Wikivoyage page title."""
    try:
        r = requests.get(
            _WV_BASE,
            params={"action": "parse", "page": page_title,
                    "prop": "sections", "format": "json"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        sections = r.json().get("parse", {}).get("sections", [])
    except Exception as exc:
        logger.debug("Wikivoyage sections failed for '%s': %s", page_title, exc)
        return {}

    matched: dict[str, str] = {}
    for s in sections:
        line = s.get("line", "").lower()
        idx  = s.get("index", "")
        key  = next((k for k in _WIKIVOYAGE_SECTIONS if k in line), None)
        if key and key not in matched:
            try:
                r2 = requests.get(
                    _WV_BASE,
                    params={"action": "parse", "page": page_title,
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


def wikivoyage_local_tips(city: str, region: str = "", country: str = "") -> dict[str, str]:
    """Fetch local tips from Wikivoyage, falling back to region/country if city page missing.

    For small places that don't have their own Wikivoyage article, tries the parent
    region or country instead so the output is always useful.
    """
    # Try exact city name first
    tips = _fetch_wikivoyage(city)
    if tips:
        return tips

    # Fallback 1 — region (state / county / province)
    if region:
        logger.info("Wikivoyage: '%s' not found, trying region '%s'", city, region)
        tips = _fetch_wikivoyage(region)
        if tips:
            return tips

    # Fallback 2 — country
    if country:
        logger.info("Wikivoyage: region not found, trying country '%s'", country)
        tips = _fetch_wikivoyage(country)
        if tips:
            return tips

    return {}


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
