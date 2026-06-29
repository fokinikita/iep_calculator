import logging
from typing import List

import attr
import polars as pl

import config_features
from services.features.additional_base.additional_base_features import (
    AdditionalBaseFeaturesService,
)
from services.features.geo.geo_features import GeoFeaturesService
from services.features.nlp.nlp_features import NLPFeaturesService

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@attr.define(slots=True)
class FeaturesService:
    features: pl.DataFrame
    features_names: list[str] = attr.attrib(init=False, default=None)
    nlp_features_names: list[str] = attr.attrib(init=False, default=None)

    def calculate_features(self) -> pl.DataFrame:
        logger.info("Start feature generation pipeline")

        additional_base_features_names = self._get_additional_base_features()
        self.nlp_features_names = self._get_nlp_features()
        geo_features_names = self._get_geo_features()

        self.features_names = (
            additional_base_features_names
            + geo_features_names
            + self.nlp_features_names
        )

        self.features = self._add_target(self.features)

        logger.info("Feature generation completed")
        return self.features

    def _get_additional_base_features(self) -> List[str]:
        logger.info("Adding additional base features")

        service = AdditionalBaseFeaturesService(self.features)
        self.features = service.calculate_features()

        logger.info(
            f"Added {len(service.additional_base_features_names)} base features"
        )

        return service.additional_base_features_names

    def _get_geo_features(self) -> List[str]:
        logger.info("Adding geo features")

        service = GeoFeaturesService(self.features)
        self.features = service.calculate_features()

        self.features = self._add_neighbors_stats(self.features)

        self.features = self.features.drop(
            [
                col
                for col in ["neighbors_distance_m", "neighbors_guid"]
                if col in self.features.columns
            ]
        )

        geo_features_names = [
            feature
            for feature in service.geo_features_names
            if feature not in ["neighbors_distance_m", "neighbors_guid"]
        ]

        logger.info(f"Added {len(geo_features_names)} geo features")

        return geo_features_names

    def _get_nlp_features(self) -> List[str]:
        logger.info("Adding NLP features")

        service = NLPFeaturesService(self.features)
        self.features = service.calculate_features()

        logger.info(f"Added {len(service.nlp_features_names)} NLP features")

        return service.nlp_features_names

    @staticmethod
    def _add_neighbors_stats(df: pl.DataFrame) -> pl.DataFrame:

        lookup = df.select(
            [
                "guid",
                "price",
                "total_square",
                "total_room_count",
            ]
        )

        df_exploded = (
            df.with_row_index("row_id")
            .explode("neighbors_guid")
            .rename({"neighbors_guid": "neighbor_guid"})
        )

        df_joined = df_exploded.join(
            lookup, left_on="neighbor_guid", right_on="guid", how="left"
        ).rename(
            {
                "price": "neighbor_price",
                "total_square": "neighbor_total_square",
                "total_room_count": "neighbor_total_rooms",
            }
        )

        df_agg = df_joined.group_by("row_id").agg(
            [
                pl.col("neighbor_price").mean().alias("neighbors_mean_price"),
                pl.col("neighbor_total_square")
                .mean()
                .alias("neighbors_mean_total_square"),
                pl.col("neighbor_total_rooms")
                .median()
                .cast(pl.Int64)
                .alias("neighbors_median_total_rooms"),
            ]
        )

        df = (
            df.with_row_index("row_id")
            .join(df_agg, on="row_id", how="left")
            .drop("row_id")
        )

        return df

    @staticmethod
    def _add_target(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col("price") / pl.col("total_square")).log().alias("log_price_sq")
        )

    def get_feature_types(self) -> tuple[list[str], list[str]]:
        """
        Returns:
            continuous_features, categorical_features
        """

        # --- categorical
        categorical_features = [
            *config_features.BASE_CATEGORICAL_FEATURES,
            *config_features.GEO_CATEGORICAL_FEATURES,
        ] + [
            feature for feature in self.nlp_features_names if feature.startswith("has_")
        ]

        text_features = [
            *config_features.BASE_TEXT_FEATURES,
        ]

        continuous_features = [
            feature
            for feature in self.features_names
            if feature not in categorical_features and feature not in text_features
        ]

        return continuous_features, categorical_features
