
import gradio as gr

from .generative_chatter_detector import GenerativeChatterDetector
from .rules_chatter_detector import  simple_normalized_blacbriar_chatter_detector
from .discriminative_chatter_detector import DatasetName, DiscriminativeChatterDetector



def update_dataset(selected_dataset):
    """Reinitialize discriminative detector with new dataset"""
    global discriminative_detector
    discriminative_detector = DiscriminativeChatterDetector(dataset_name=selected_dataset)
    return f"✓ Classic ML model now trained on: {selected_dataset}"


generative_detector = GenerativeChatterDetector(scope='blackbriar-only')


def detect_chatter_a(text):
    """Model A detection"""
    result = generative_detector.predict(text)
    return f"**Prediction:** {result['label']}"


def detect_chatter_b(text):
    """Model B detection"""
    # result = detector_b.detect(text)
    # return f"**Prediction:** {result['label']}\n\n**Confidence:** {result['confidence']:.2%}"
    result = simple_normalized_blacbriar_chatter_detector(transcript=text)
    return f"**Prediction:** {result}"


def detect_chatter_c(text):
    """Model C detection"""
    result = discriminative_detector.predict(text)
    return f"**Prediction:** {result['label']}"


def clear_all_outputs():
    """
    Source - https://stackoverflow.com/a
    Posted by Vaishnav
    Retrieved 2025-11-17, License - CC BY-SA 4.0
    """
    return None, None, None, None



hwr_theme = gr.themes.Default(primary_hue=gr.themes.colors.rose, secondary_hue=gr.themes.colors.red)


with gr.Blocks(theme=hwr_theme) as demo:

    gr.Markdown("""
    # Chatter Detection Demo

    Test three different AI models for detecting (fake) security-relevant chatter in text.
    One demo uses a rules-based model, one uses predictive ML ('discriminative') and one uses generative ML.

    Can you guess which is which?

    **Your task:** Try various inputs and track the responses to guess which demo is using which model type.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type your message here...",
                lines=3
            )
        with gr.Column(scale=1):
            dataset_name = gr.Dropdown(
            choices=[a_name.value for a_name in DatasetName],
            value='blackbriar',  # Set default value
            label="(Optional) Change training Dataset for Classic ML"
        )
        dataset_status = gr.Markdown("Currently using: blackbriar")


    gr.Markdown("### Compare All Three Models")

    # with gr.Row():


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

    btn_a.click(fn=detect_chatter_a, inputs=input_text, outputs=output_a)
    btn_b.click(fn=detect_chatter_b, inputs=input_text, outputs=output_b)
    btn_c.click(fn=detect_chatter_c, inputs=input_text, outputs=output_c)
    clear_btn.click(fn=clear_all_outputs, inputs=None, outputs=[input_text, output_a, output_b, output_c])

    dataset_name.change(
        fn=update_dataset,
        inputs=dataset_name,
        outputs=dataset_status
    )
