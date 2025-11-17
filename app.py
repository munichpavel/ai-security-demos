import gradio as gr
from ai_security.demo1_sklearn import SklearnChatterDetector
from ai_security.demo2_exact_match import ExactMatchChatterDetector
from ai_security.demo3_generative import GenerativeChatterDetector
from ai_security.rules_chatter_detector import  simple_normalized_blacbriar_chatter_detector

# Initialize all detectors
detector_a = SklearnChatterDetector()
detector_b = ExactMatchChatterDetector()
detector_c = GenerativeChatterDetector()

def detect_chatter_a(text):
    """Model A detection"""
    result = detector_a.detect(text)
    return f"**Prediction:** {result['label']}"

def detect_chatter_b(text):
    """Model B detection"""
    # result = detector_b.detect(text)
    # return f"**Prediction:** {result['label']}\n\n**Confidence:** {result['confidence']:.2%}"
    result = simple_normalized_blacbriar_chatter_detector(transcript=text)
    return f"**Prediction:** {result}"


def detect_chatter_c(text):
    """Model C detection"""
    result = detector_c.detect(text)
    return f"**Prediction:** {result['label']}"


hwr_theme = gr.themes.Default(primary_hue=gr.themes.colors.rose, secondary_hue=gr.themes.colors.red)

with gr.Blocks(theme=hwr_theme) as demo:

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

     # Source - https://stackoverflow.com/a
    # Posted by Vaishnav
    # Retrieved 2025-11-17, License - CC BY-SA 4.0

    def clear_all_outputs():
        return None, None, None, None


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
            [" Bourne's just the tip of the iceberg. Have you heard of an 'Operation Blackbriar'?"],
            ["The files on Blackbriar. Everything."],
            ["Piscataguog Land Conservancy first acquired Blackbriar Woods in 2011 and then Black Brook Preserve in 2015."],
            ["Whether you're a mother or whether you're a brother you're a stayin' alive. Stayin' alive."]
        ],
        inputs=input_text
    )
    clear_btn = gr.Button("Let's try again.")

    # Connect buttons to functions
    btn_a.click(fn=detect_chatter_a, inputs=input_text, outputs=output_a)
    btn_b.click(fn=detect_chatter_b, inputs=input_text, outputs=output_b)
    btn_c.click(fn=detect_chatter_c, inputs=input_text, outputs=output_c)
    clear_btn.click(fn=clear_all_outputs, inputs=None, outputs=[input_text, output_a, output_b, output_c])


if __name__ == "__main__":
    demo.launch()