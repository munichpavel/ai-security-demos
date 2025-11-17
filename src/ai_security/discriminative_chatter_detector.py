"""
Chatter detection using bag-of-words and multinomial bayes
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class DiscriminativeChatterDetector:
    def __init__(self, filter_category: str):
        self.filter_category = filter_category
        self.model = None
        self.categories = None
        self._load_and_train()

    def _load_data(self) -> pd.DataFrame:
        csv_path = Path(__file__).parent.parent.parent / "data" / "chatter-detection-dataset.csv"
        df = pd.read_csv(csv_path)

        # Filter by categories-predicted if specified
        if self.filter_category:
            df = df[df["categories-predicted"] == self.filter_category].copy()

        # Remove rows with missing transcript or category
        df = df.dropna(subset=["transcript", "category"])

        return df

    def _load_and_train(self):
        df = self._load_data()

        if len(df) == 0:
            raise ValueError(f"No training data found for filter: {self.filter_category}")

        # Store unique categories
        self.categories = df["category"].unique().tolist()

        # Create and train pipeline
        self.model = Pipeline([
            ("vectorizer", CountVectorizer(max_features=1000, stop_words="english")),
            ("classifier", MultinomialNB())
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
