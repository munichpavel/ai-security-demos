"""
Demo 1: Bag-of-Words + Logistic Regression Chatter Detector
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

class SklearnChatterDetector:
    def __init__(self):
        # Training data - examples of chatter/spam vs legitimate messages
        self.train_texts = [
            # Chatter/Spam examples
            "Click here for free prizes",
            "Buy now limited time offer",
            "Congratulations you won",
            "Urgent verify your account",
            "Make money fast",
            "Free gift click here",
            "Act now don't miss out",
            "Special discount today only",
            "You have been selected winner",
            "Claim your prize now",
            # Legitimate examples
            "Can we schedule a meeting",
            "Thanks for your help",
            "Let me know your thoughts",
            "I'll send the report tomorrow",
            "Please review the document",
            "Looking forward to our discussion",
            "I appreciate your feedback",
            "Could you clarify this point",
            "The project is on track",
            "I have a question about the task"
        ]

        self.train_labels = [1] * 10 + [0] * 10  # 1 = chatter, 0 = legitimate

        # Create and train the pipeline
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                lowercase=True,
                max_features=100,
                ngram_range=(1, 2)
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])

        self.pipeline.fit(self.train_texts, self.train_labels)

    def detect(self, text):
        """
        Detect if text is chatter/spam

        Args:
            text: Input text to analyze

        Returns:
            dict with 'label' and 'confidence'
        """
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]

        label = "Chatter/Spam" if prediction == 1 else "Legitimate"
        confidence = probabilities[prediction]

        return {
            'label': label,
            'confidence': float(confidence)
        }