import gradio as gr
from ai_security.demo1_sklearn import SklearnChatterDetector
from ai_security.demo2_exact_match import ExactMatchChatterDetector
from ai_security.demo3_generative import GenerativeChatterDetector

# Initialize all detectors
detector_a = SklearnChatterDetector()
detector_b = ExactMatchChatterDetector()
detector_c = GenerativeChatterDetector()

def detect_chatter_a(text):
    """Model A detection"""
    result = detector_a.detect(text)
    return f"**Prediction:** {result['label']}\n\n**Confidence:** {result['confidence']:.2%}"

def detect_chatter_b(text):
    """Model B detection"""
    result = detector_b.detect(text)
    return f"**Prediction:** {result['label']}\n\n**Confidence:** {result['confidence']:.2%}"

def detect_chatter_c(text):
    """Model C detection"""
    result = detector_c.detect(text)
    return f"**Prediction:** {result['label']}\n\n**Confidence:** {result['confidence']:.2%}"

# Create Gradio interface with custom CSS
custom_css = """
button.primary {
    background-color: #CD222A !important;
    border-color: #CD222A !important;
}
button.primary:hover {
    background-color: #A51C22 !important;
    border-color: #A51C22 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # Chatter Detection Demo

    Test three different AI models for detecting chatter/spam in text.
    Each model uses a different approach - can you figure out which technique each one uses?

    **Your task:** Try various inputs and observe how each model behaves differently.
    """)

    with gr.Row():
        input_text = gr.Textbox(
            label="Enter text to analyze",
            placeholder="Type your message here...",
            lines=3
        )

    gr.Markdown("### Compare All Three Models")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Model A")
            output_a = gr.Markdown()
            btn_a = gr.Button("Analyze with Model A", variant="primary")

        with gr.Column():
            gr.Markdown("#### Model B")
            output_b = gr.Markdown()
            btn_b = gr.Button("Analyze with Model B", variant="primary")

        with gr.Column():
            gr.Markdown("#### Model C")
            output_c = gr.Markdown()
            btn_c = gr.Button("Analyze with Model C", variant="primary")

    # Example inputs
    gr.Examples(
        examples=[
            ["Click here for free prizes!!!"],
            ["Can we schedule a meeting for tomorrow?"],
            ["BUY NOW LIMITED TIME OFFER"],
            ["Thanks for your help with the project"],
            ["URGENT: Verify your account immediately"],
            ["Let me know if you have any questions"]
        ],
        inputs=input_text
    )

    # Connect buttons to functions
    btn_a.click(fn=detect_chatter_a, inputs=input_text, outputs=output_a)
    btn_b.click(fn=detect_chatter_b, inputs=input_text, outputs=output_b)
    btn_c.click(fn=detect_chatter_c, inputs=input_text, outputs=output_c)

    gr.Markdown("""
    ---
    ### Questions to Consider:
    - Which model seems most consistent?
    - Which model is most sensitive to specific words or phrases?
    - Which model provides the most nuanced analysis?
    - Can you craft inputs that fool one model but not others?
    """)

if __name__ == "__main__":
    demo.launch()