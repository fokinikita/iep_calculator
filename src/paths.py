from pathlib import Path

from settings.base_settings import base_settings

BASE_PATH = Path(base_settings.PROJECT_PATH)

MOSCOW_POLYGONS_RELATIVE_PATH = "iep_calculator/data/geo/moscow.geojson"
MOSCOW_AREAS_RELATIVE_PATH = "iep_calculator/data/geo/moscow_areas.geojson"
OBLAST_POLYGONS_RELATIVE_PATH = "iep_calculator/data/geo/moscow_oblast.geojson"
OBLAST_CITIES_RELATIVE_PATH = "iep_calculator/data/geo/moscow_oblast_cities.json"
METRO_RELATIVE_PATH = "iep_calculator/data/geo/metro.geojson"

MOSCOW_POLYGONS_ABSOLUTE_PATH = BASE_PATH / MOSCOW_POLYGONS_RELATIVE_PATH
MOSCOW_AREAS_ABSOLUTE_PATH = BASE_PATH / MOSCOW_AREAS_RELATIVE_PATH
OBLAST_POLYGONS_ABSOLUTE_PATH = BASE_PATH / OBLAST_POLYGONS_RELATIVE_PATH
OBLAST_CITIES_ABSOLUTE_PATH = BASE_PATH / OBLAST_CITIES_RELATIVE_PATH
METRO_ABSOLUTE_PATH = BASE_PATH / METRO_RELATIVE_PATH
