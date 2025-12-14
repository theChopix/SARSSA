import polars as pl
import numpy as np
import scipy.sparse as sp

from .dataset_loader import DatasetLoader


class MovieLensLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 50
    MIN_ITEM_INTERACTIONS: int = 20
    def __init__(self, ratings_file_path: str = '../data/movieLens/MovieLensRatings.csv', tags_file_path: str = '../data/movieLens/MovieLensTags.csv'):
        super().__init__('MovieLens', ratings_file_path, tags_file_path)
        
    def _load_ratings(self, ratings_file_path: str) -> None:
        self.df_interactions = (
            pl.scan_csv(ratings_file_path, has_header=True)
            .select(['userId', 'movieId', 'rating'])
            .rename({'movieId': 'itemId'})
            .cast({'userId': pl.String, 'itemId': pl.String, 'rating': pl.Float64})
            .filter(pl.col('rating') >= 4.0)
            .select(['userId', 'itemId'])
            .unique()
            .sort(['userId', 'itemId'])
            .collect()
        )

    def _load_tags(self, tags_file_path: str, items: np.ndarray) -> None:
        self.df_tags = (
            pl.scan_csv(tags_file_path, has_header=True)
            .select(['movieId', 'tag'])
            .rename({'movieId': 'itemId'})
            .cast({'itemId': pl.String, 'tag': pl.String})
            .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars().alias("tag"))
            .filter(pl.col("itemId").is_in(items))
            .unique()
            .collect()
        )

    def has_tags(self) -> bool:
        return True

    def tag_ids(self):
        return self.df_tags["tag"].unique().sort().to_list()

    def tag_item_matrix(self):
        tag_ids = self.tag_ids()
        tag_to_idx = {t: i for i, t in enumerate(tag_ids)}
        item_to_idx = {i: idx for idx, i in enumerate(self.items)}

        rows, cols = [], []

        for row in self.df_tags.iter_rows(named=True):
            rows.append(tag_to_idx[row["tag"]])
            cols.append(item_to_idx[row["itemId"]])

        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(tag_ids), len(self.items))
        )