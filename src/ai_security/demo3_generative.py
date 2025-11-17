"""
Demo 3: Generative AI Model Chatter Detector
Uses a truly generative model (text generation) to analyze and classify
"""
import requests
import json
import re

class GenerativeChatterDetector:
    def __init__(self):
        # Use a generative language model that can reason about text
        # Using HuggingFace's free Inference API with a small generative model
        self.api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    def detect(self, text):
        """
        Detect if text is chatter/spam using a generative model
        The model generates a classification by reasoning about the text

        Args:
            text: Input text to analyze

        Returns:
            dict with 'label' and 'confidence'
        """
        try:
            # Craft a prompt that asks the generative model to classify
            prompt = f"""Analyze this message and determine if it is spam/chatter or a legitimate message.
Message: "{text}"

Question: Is this spam or legitimate?
Answer:"""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 20,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()

                # Parse the generated text
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '').lower().strip()

                    # Look for spam/legitimate indicators in generated response
                    is_spam = 'spam' in generated_text or 'chatter' in generated_text
                    is_legit = 'legitimate' in generated_text or 'legit' in generated_text or 'not spam' in generated_text

                    if is_spam and not is_legit:
                        label = "Chatter/Spam"
                        confidence = 0.75
                    elif is_legit and not is_spam:
                        label = "Legitimate"
                        confidence = 0.75
                    else:
                        # Model was uncertain, use heuristic
                        return self._fallback_detection(text)

                    return {
                        'label': label,
                        'confidence': confidence
                    }

            # Fallback if API fails
            return self._fallback_detection(text)

        except Exception as e:
            # Fallback to simple heuristic if API is unavailable
            return self._fallback_detection(text)

    def _fallback_detection(self, text):
        """
        Fallback detection using simple heuristics
        This simulates what a generative model might consider
        """
        # Count "suspicious" features
        text_lower = text.lower()

        suspicious_score = 0

        # Check for urgency words
        urgency_words = ['urgent', 'now', 'immediately', 'hurry', 'fast']
        suspicious_score += sum(2 for word in urgency_words if word in text_lower)

        # Check for promotional language
        promo_words = ['free', 'win', 'prize', 'offer', 'discount', 'buy']
        suspicious_score += sum(1.5 for word in promo_words if word in text_lower)

        # Check for excessive punctuation
        if '!!!' in text or '???' in text:
            suspicious_score += 2

        # Check for all caps words
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        suspicious_score += len(caps_words) * 1.5

        # Normalize score to confidence
        confidence = min(0.95, max(0.55, suspicious_score / 10))

        if suspicious_score >= 3:
            return {
                'label': "Chatter/Spam",
                'confidence': confidence
            }
        else:
            return {
                'label': "Legitimate",
                'confidence': 1 - confidence + 0.3
            }