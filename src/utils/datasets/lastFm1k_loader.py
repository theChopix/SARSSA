import polars as pl
import numpy as np

from .dataset_loader import DatasetLoader


class LastFm1kLoader(DatasetLoader):
    MIN_USER_INTERACTIONS: int = 5
    MIN_ITEM_INTERACTIONS: int = 10
    
    def __init__(self, ratings_file_path: str = '../data/LastFm1k.tsv'):
        super().__init__('LastFM1k', ratings_file_path)
        
    def _load_ratings(self, ratings_file_path: str) -> None:
        skiprows = [
        2120260-1, 2446318-1, 11141081-1,
        11152099-1, 11152402-1, 11882087-1,
        12902539-1, 12935044-1, 17589539-1
        ]
        
        self.df_interactions = (
            pl.scan_csv(ratings_file_path, separator='\t', has_header=False, quote_char=None)
            .rename({'column_1': 'userId', 'column_3': 'itemId'})
            .select(['userId', 'itemId'])
            .with_row_index().filter(~pl.col("index").is_in(skiprows)).drop("index") # skip damaged rows
            .cast({'userId': pl.String, 'itemId': pl.String})
            .unique()
             .sort(['userId', 'itemId'])
            .collect()
        )

    def _load_tags(self, tags_file_path: str, items: np.ndarray) -> None:
        raise NotImplementedError('This dataset does not provide tag metadata.')