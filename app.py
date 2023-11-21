import os
import cv2
import torch

import gradio as gr
import numpy as np
import supervision as sv

from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import postprocess_masks, Visualizer
from gpt4v import prompt_image

HOME = os.getenv("HOME")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SAM_CHECKPOINT = os.path.join(HOME, "app/weights/sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"

MARKDOWN = """
[![arXiv](https://img.shields.io/badge/arXiv-1703.06870v3-b31b1b.svg)](https://arxiv.org/pdf/2310.11441.pdf)

<h1 style='text-align: center'>
    <img 
        src='https://som-gpt4v.github.io/website/img/som_logo.png' 
        style='height:50px; display:inline-block'
    />  
    Set-of-Mark (SoM) Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
</h1>

## 🚧 Roadmap

- [ ] Support for alphabetic labels
- [ ] Support for Semantic-SAM (multi-level)
- [ ] Support for interactive mode
- [ ] Support for result highlighting
"""

SAM = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)


def inference(
    image: np.ndarray,
    annotation_mode: List[str],
    mask_alpha: float
) -> np.ndarray:
    visualizer = Visualizer(mask_opacity=mask_alpha)
    mask_generator = SamAutomaticMaskGenerator(SAM)
    result = mask_generator.generate(image=image)
    detections = sv.Detections.from_sam(result)
    detections = postprocess_masks(
        detections=detections)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = visualizer.visualize(
        image=bgr_image,
        detections=detections,
        with_box="Box" in annotation_mode,
        with_mask="Mask" in annotation_mode,
        with_polygon="Polygon" in annotation_mode,
        with_label="Mark" in annotation_mode)
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


def prompt(message, history, image: np.ndarray, api_key: str) -> str:
    if api_key == "":
        return "⚠️ Please set your OpenAI API key first"
    if image is None:
        return "⚠️ Please generate SoM visual prompt first"
    return prompt_image(
        api_key=api_key,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        prompt=message
    )


image_input = gr.Image(
    label="Input",
    type="numpy",
    height=512)
checkbox_annotation_mode = gr.CheckboxGroup(
    choices=["Mark", "Polygon", "Mask", "Box"],
    value=['Mark'],
    label="Annotation Mode")
slider_mask_alpha = gr.Slider(
    minimum=0,
    maximum=1,
    value=0.05,
    label="Mask Alpha")
image_output = gr.Image(
    label="SoM Visual Prompt",
    type="numpy",
    height=512)
openai_api_key = gr.Textbox(
    show_label=False,
    placeholder="Before you start chatting, set your OpenAI API key here",
    lines=1,
    type="password")
chatbot = gr.Chatbot(
    label="GPT-4V + SoM",
    height=256)
run_button = gr.Button("Run")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input.render()
            with gr.Accordion(
                    label="Detailed prompt settings (e.g., mark type)",
                    open=False):
                with gr.Row():
                    checkbox_annotation_mode.render()
                with gr.Row():
                    slider_mask_alpha.render()
        with gr.Column():
            image_output.render()
            run_button.render()
    with gr.Row():
        openai_api_key.render()
    with gr.Row():
        gr.ChatInterface(
            chatbot=chatbot,
            fn=prompt,
            additional_inputs=[image_output, openai_api_key])

    run_button.click(
        fn=inference,
        inputs=[image_input, checkbox_annotation_mode, slider_mask_alpha],
        outputs=image_output)

demo.queue().launch(debug=False, show_error=True)
