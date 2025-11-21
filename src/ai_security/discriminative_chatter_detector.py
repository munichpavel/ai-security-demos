"""
Chatter detection using discriminative ML
"""
from enum import Enum
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class DatasetName(Enum):
    baby_blackbriar = 'baby-blackbriar'
    toddler_blackbriar = 'toddler-blackbriar'
    blackbriar = 'blackbriar'
    extended = 'extended'


DATASET_NAME_FILENAME_MAPPING = {
    DatasetName.baby_blackbriar.value : 'chatter-detection-dataset - baby-chatter-detection-dataset.csv',
    DatasetName.toddler_blackbriar.value: 'chatter-detection-dataset - toddler-chatter-detection-dataset.csv',
    DatasetName.blackbriar.value: 'chatter-detection-dataset - blackbriar-chatter-detection-dataset.csv',
    DatasetName.extended.value: 'chatter-detection-dataset - extended-chatter-detection-dataset.csv',
}


def load_training_data(dataset_name: str) -> pd.DataFrame:
    use_cols = ['transcript', 'category', 'embedding_comment', 'category_comment']
    filename = DATASET_NAME_FILENAME_MAPPING[dataset_name]
    data_root = _get_data_root()
    data_path = data_root / filename
    df = pd.read_csv(data_path)
    res = df.loc[:, use_cols]
    return res

def _get_data_root() -> Path:
    return Path(__file__).parent.parent.parent / 'data'

class DiscriminativeChatterDetector:
    def __init__(self, dataset_name: str = 'blackbriar'):
        self.dataset_name = dataset_name
        self.model = None
        self.categories = None
        self._load_and_train()

    def _load_and_train(self):
        df = load_training_data(dataset_name=self.dataset_name)

        if len(df) == 0:
            raise ValueError(f"No training data found for filter: {self.filter_category}")

        # Store unique categories
        self.categories = df["category"].unique().tolist()

        # Create and train pipeline
        self.model = Pipeline([
            ("vectorizer", CountVectorizer(max_features=1000, stop_words="english")),
            ("classifier", LogisticRegression())
        ])

        self.model.fit(df["transcript"], df["category"])

    def predict(self, text: str) -> dict:
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]

        prob_dict = {cat: float(prob) for cat, prob in zip(self.categories, probabilities)}

        return {
            "label": prediction,
            "probabilities": prob_dict
        }

    def get_categories(self) -> list:
        """Return list of trained categories."""
        return self.categories
