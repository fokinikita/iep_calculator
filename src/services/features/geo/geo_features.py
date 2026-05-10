import json
import math
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, TypedDict

import attr
import numpy as np
import polars as pl
from shapely.geometry import Point, shape
from sklearn.neighbors import BallTree

import paths


class Geometry(TypedDict):
    type: str
    coordinates: Any


class FeatureProperties(TypedDict, total=False):
    name: str
    name_ru: str
    name_en: str


class Feature(TypedDict):
    type: str
    geometry: Geometry
    properties: Dict[str, Any]


class FeatureCollection(TypedDict):
    type: str
    features: List[Feature]


CENTER_LAT: Final[float] = 55.751028
CENTER_LON: Final[float] = 37.617611
EARTH_RADIUS_KM: Final[float] = 6371.0


@attr.attrs(slots=True, auto_attribs=True)
class GeoFeaturesService:
    features: pl.DataFrame
    reference_df: Optional[pl.DataFrame] = None

    moscow_polygons: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    moscow_areas: Optional[List[Dict[str, Any]]] = attr.attrib(init=False, default=None)
    oblast_polygons: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    oblast_cities: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )
    metro_stations: Optional[List[Dict[str, Any]]] = attr.attrib(
        init=False, default=None
    )

    geo_features_names: List[str] = attr.attrib(init=False, factory=list)

    def _get_cols_features(self) -> List[str]:
        return self.features.columns

    @staticmethod
    def load_polygons(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data: FeatureCollection = json.load(f)

        polygons: List[Dict[str, Any]] = []

        for feat in data["features"]:
            props = feat["properties"]

            name: str = (
                props.get("name")
                or props.get("name:ru")
                or props.get("name:en")
                or "unknown"
            )

            polygons.append(
                {
                    "name": name,
                    "geometry": shape(feat["geometry"]),
                }
            )

        return polygons

    @staticmethod
    def load_points(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points: List[Dict[str, Any]] = []
        features = data["features"] if isinstance(data, dict) else data

        for feat in features:
            props = feat.get("properties", {})

            name: str = (
                props.get("name") or feat.get("name") or feat.get("city") or "unknown"
            )

            if "geometry" in feat:
                geom = shape(feat["geometry"])
                if isinstance(geom, Point):
                    points.append(
                        {
                            "name": name,
                            "geometry": geom,
                        }
                    )
            else:
                lat = feat.get("lat") or feat.get("latitude")
                lon = feat.get("lon") or feat.get("lng") or feat.get("longitude")

                if lat is not None and lon is not None:
                    points.append(
                        {
                            "name": name,
                            "geometry": Point(lon, lat),
                        }
                    )

        return points

    @staticmethod
    def load_metro(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        res: List[Dict[str, Any]] = []

        for feat in data["features"]:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]

            name = props.get("name")
            if not name:
                continue

            res.append(
                {
                    "name": name,
                    "lat": coords[1],
                    "lon": coords[0],
                }
            )

        return res

    def _ensure_data_loaded(self) -> None:
        if self.moscow_polygons is None:
            self.moscow_polygons = self.load_polygons(
                paths.MOSCOW_POLYGONS_ABSOLUTE_PATH
            )

        if self.moscow_areas is None:
            self.moscow_areas = self.load_polygons(paths.MOSCOW_AREAS_ABSOLUTE_PATH)

        if self.oblast_polygons is None:
            self.oblast_polygons = self.load_polygons(
                paths.OBLAST_POLYGONS_ABSOLUTE_PATH
            )

        if self.oblast_cities is None:
            self.oblast_cities = self.load_points(paths.OBLAST_CITIES_ABSOLUTE_PATH)

        if self.metro_stations is None:
            self.metro_stations = self.load_metro(paths.METRO_ABSOLUTE_PATH)

    @staticmethod
    def classify(
        row: Dict[str, float],
        moscow_polygons: List[Dict[str, Any]],
        moscow_areas: List[Dict[str, Any]],
        oblast_polygons: List[Dict[str, Any]],
        oblast_cities: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        lat = row.get("location.lat")
        lon = row.get("location.lon")

        if lat is None or lon is None:
            return ("unknown", None, None)

        point = Point(lon, lat)

        for poly in moscow_polygons:
            if poly["geometry"].contains(point):
                for area in moscow_areas:
                    if area["geometry"].contains(point):
                        return ("moscow", area["name"], "Москва")
                return ("moscow", None, "Москва")

        for poly in oblast_polygons:
            if poly["geometry"].contains(point):
                closest_city = None
                min_distance = float("inf")

                for city in oblast_cities:
                    dist = point.distance(city["geometry"])
                    if dist < min_distance:
                        min_distance = dist
                        closest_city = city["name"]

                return ("oblast", poly["name"], closest_city)

        return ("unknown", None, None)

    def add_region_feature(self) -> None:
        self._ensure_data_loaded()

        self.features = (
            self.features.with_columns(
                pl.struct(["location.lat", "location.lon"])
                .map_elements(
                    lambda row: self.classify(
                        row,
                        self.moscow_polygons,
                        self.moscow_areas,
                        self.oblast_polygons,
                        self.oblast_cities,
                    ),
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("tmp")
            )
            .with_columns(
                [
                    pl.col("tmp").list.get(0).alias("region_type"),
                    pl.col("tmp").list.get(1).alias("district_name"),
                    pl.col("tmp").list.get(2).alias("city_name"),
                ]
            )
            .with_columns(
                pl.when(pl.col("region_type") == "moscow")
                .then(pl.col("district_name"))
                .when(pl.col("region_type") == "oblast")
                .then(pl.col("city_name"))
                .otherwise(None)
                .alias("region_name")
            )
            .drop("tmp")
        )

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )

        return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def add_center_distance_feature(self) -> None:
        self.features = self.features.with_columns(
            pl.struct(["location.lat", "location.lon"])
            .map_elements(
                lambda row: self.haversine(
                    row["location.lat"],
                    row["location.lon"],
                    CENTER_LAT,
                    CENTER_LON,
                ),
                return_dtype=pl.Float64,
            )
            .alias("distance_m")
        ).with_columns(
            [
                pl.col("distance_m").sqrt().alias("distance_sqrt"),
                (pl.col("distance_m") ** 2).alias("distance_sq"),
            ]
        )

    @staticmethod
    def nearest_metro(
        row: Dict[str, float],
        metro: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lat = row.get("location.lat")
        lon = row.get("location.lon")

        if lat is None or lon is None:
            return {
                "metro_name": None,
                "metro_distance_m": None,
            }

        min_dist = float("inf")
        closest = None

        for st in metro:
            d = GeoFeaturesService.haversine(lat, lon, st["lat"], st["lon"])
            if d < min_dist:
                min_dist = d
                closest = st["name"]

        return {
            "metro_name": closest,
            "metro_distance_m": min_dist,
        }

    def add_metro_feature(self) -> None:
        self._ensure_data_loaded()

        self.features = (
            self.features.with_columns(
                pl.struct(["location.lat", "location.lon"])
                .map_elements(
                    lambda row: self.nearest_metro(row, self.metro_stations),
                    return_dtype=pl.Struct(
                        [
                            pl.Field("metro_name", pl.Utf8),
                            pl.Field("metro_distance_m", pl.Float64),
                        ]
                    ),
                )
                .alias("metro_tmp")
            )
            .unnest("metro_tmp")
            .with_columns(
                [
                    pl.col("metro_distance_m").sqrt().alias("metro_distance_sqrt"),
                    (pl.col("metro_distance_m") ** 2).alias("metro_distance_sq"),
                ]
            )
        )

    def _add_empty_neighbors(self) -> None:
        rows = self.features.height
        self.features = self.features.with_columns(
            [
                pl.Series(
                    "neighbors_guid", [[] for _ in range(rows)], dtype=pl.List(pl.Utf8)
                ),
                pl.Series(
                    "neighbors_distance_m",
                    [[] for _ in range(rows)],
                    dtype=pl.List(pl.Float64),
                ),
            ]
        )

    def add_nearest_neighbors(self, k: int = 5) -> None:
        if self.features.height == 0:
            self._add_empty_neighbors()
            return

        # Если reference_df не передан — не падаем
        if self.reference_df is None or self.reference_df.height == 0:
            self._add_empty_neighbors()
            return

        required_cols = {"guid", "location.lat", "location.lon"}
        if not required_cols.issubset(set(self.reference_df.columns)):
            self._add_empty_neighbors()
            return

        query_coords = self.features.select(["location.lat", "location.lon"]).to_numpy()
        ref_coords = self.reference_df.select(
            ["location.lat", "location.lon"]
        ).to_numpy()
        ref_guids = self.reference_df["guid"].to_list()

        if len(ref_coords) == 0:
            self._add_empty_neighbors()
            return

        query_radians = np.radians(query_coords)
        ref_radians = np.radians(ref_coords)

        k_eff = min(k, len(ref_coords))
        if k_eff <= 0:
            self._add_empty_neighbors()
            return

        tree = BallTree(ref_radians, metric="haversine")
        distances, indices = tree.query(query_radians, k=k_eff)

        distances_km = distances * EARTH_RADIUS_KM
        neighbors_guid = [[ref_guids[i] for i in row] for row in indices]

        self.features = self.features.with_columns(
            [
                pl.Series("neighbors_guid", neighbors_guid),
                pl.Series("neighbors_distance_m", distances_km.tolist()),
            ]
        )

    def calculate_features(self) -> pl.DataFrame:
        self._ensure_data_loaded()

        feature_pipeline: List[Callable[[], None]] = [
            self.add_region_feature,
            self.add_center_distance_feature,
            self.add_metro_feature,
            self.add_nearest_neighbors,
        ]

        features_before = self._get_cols_features()

        for feature in feature_pipeline:
            feature()

        features_after = self._get_cols_features()
        self.geo_features_names = list(set(features_after) - set(features_before))

        return self.features
