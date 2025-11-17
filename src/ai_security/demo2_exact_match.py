"""
Demo 2: Exact/Simple Text Matching Chatter Detector
"""
import re

class ExactMatchChatterDetector:
    def __init__(self):
        # Define suspicious keywords/patterns
        self.spam_keywords = [
            'free', 'prize', 'winner', 'congratulations', 'click here',
            'buy now', 'limited time', 'urgent', 'verify', 'account',
            'make money', 'act now', 'special offer', 'discount',
            'claim', 'selected', 'gift', 'fast', 'guarantee'
        ]

        self.spam_patterns = [
            r'!!+',  # Multiple exclamation marks
            r'\$\d+',  # Dollar amounts
            r'100%',  # Percentage claims
            r'[A-Z]{5,}',  # Excessive capitals
        ]

    def detect(self, text):
        """
        Detect if text is chatter/spam using exact matching

        Args:
            text: Input text to analyze

        Returns:
            dict with 'label' and 'confidence'
        """
        text_lower = text.lower()

        # Count keyword matches
        keyword_matches = sum(1 for keyword in self.spam_keywords if keyword in text_lower)

        # Count pattern matches
        pattern_matches = sum(1 for pattern in self.spam_patterns if re.search(pattern, text))

        # Simple scoring: if we find matches, it's likely spam
        total_matches = keyword_matches + pattern_matches

        # Determine label and confidence
        if total_matches >= 2:
            label = "Chatter/Spam"
            confidence = min(0.95, 0.6 + (total_matches * 0.1))
        elif total_matches == 1:
            label = "Chatter/Spam"
            confidence = 0.65
        else:
            label = "Legitimate"
            confidence = 0.85

        return {
            'label': label,
            'confidence': float(confidence)
        }