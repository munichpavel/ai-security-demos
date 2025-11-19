import os
from huggingface_hub import InferenceClient # Import the new client


SCOPE_CATEGORIES = {
    'blackbriar-only': ['harmless', 'blackbriar'],
    'all': ['harmless', 'blackbriar', 'treadstone', 'ultra']
}

SCOPE_PROMPT_TEMPLATES={
    'blackbriar-only': """Classify this text as exactly one of the following categories: '{labels}'
Message: "{{text}}"
Answer:""".format(labels="', '".join(SCOPE_CATEGORIES['blackbriar-only'])),
    'all': """Analyze this message and determine if it is
Message: "{text}"

Question: Is this spam or legitimate?
Answer:"""
}


class GenerativeChatterDetector:
    def __init__(self, scope: str):
        self.prompt_template = SCOPE_PROMPT_TEMPLATES[scope]
        self.categories = SCOPE_CATEGORIES[scope]
        self.client = InferenceClient(model='google/gemma-2-2b-it', token=os.environ['HF_INFERENCE_TOKEN'])


    def predict(self, text):
        try:
            prompt = self.prompt_template.format(text=text)
            print(f"Prompt: {prompt}")

            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,       # Max *new* tokens
                temperature=1.0,
            )

            generated_text = response.choices[0].message.content.lower().strip()
            print(f"Generated Text: {generated_text}")

            # Hacky response parsing (your original logic)
            predicted_categories = []
            for a_cat in self.categories:
                if a_cat in generated_text:
                    predicted_categories.append(a_cat)

            print(f"Predicted Categories: {predicted_categories}")

            if predicted_categories:
                # Use -1 to safely get the last item
                label = predicted_categories[-1]
                return {
                    'label': label,
                    'confidence': None
                }
            else:
                print("Warning: Model response did not contain a known category.")
                return dict(label=None, confidence=None)

        except Exception as e:
            print(f"An error occurred: {e}")
