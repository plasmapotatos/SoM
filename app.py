import os
import cv2
import torch

import gradio as gr
import numpy as np
import supervision as sv

from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import postprocess_masks, Visualizer

HOME = os.getenv("HOME")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MINIMUM_AREA_THRESHOLD = 0.01

SAM_CHECKPOINT = os.path.join(HOME, "app/weights/sam_vit_h_4b8939.pth")
# SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

MARKDOWN = """
<h1 style='text-align: center'>
    <img 
        src='https://som-gpt4v.github.io/website/img/som_logo.png' 
        style='height:50px; display:inline-block'
    />  
    Set-of-Mark (SoM) Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
</h1>
"""

SAM = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
VISUALIZER = Visualizer()


def inference(image: np.ndarray, annotation_mode: List[str]) -> np.ndarray:
    mask_generator = SamAutomaticMaskGenerator(SAM)
    result = mask_generator.generate(image=image)
    detections = sv.Detections.from_sam(result)
    detections = postprocess_masks(
        detections=detections,
        area_threshold=MINIMUM_AREA_THRESHOLD)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = VISUALIZER.visualize(
        image=bgr_image,
        detections=detections,
        with_box="Box" in annotation_mode,
        with_mask="Mask" in annotation_mode,
        with_polygon="Polygon" in annotation_mode,
        with_label="Mark" in annotation_mode)
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)


image_input = gr.Image(
    label="Input",
    type="numpy",
    height=512)
checkbox_annotation_mode = gr.CheckboxGroup(
    choices=["Mark", "Polygon", "Mask", "Box"],
    value=['Mark'],
    label="Annotation Mode")
image_output = gr.Image(
    label="SoM Visual Prompt",
    type="numpy",
    height=512)
run_button = gr.Button("Run")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input.render()
            with gr.Accordion(label="Detailed prompt settings (e.g., mark type)", open=False):
                checkbox_annotation_mode.render()
        with gr.Column():
            image_output.render()
            run_button.render()

    run_button.click(
        fn=inference,
        inputs=[image_input, checkbox_annotation_mode],
        outputs=image_output)

demo.queue().launch(debug=False, show_error=True)
