import os
from typing import List, Dict, Tuple, Any, Optional

import cv2
import gradio as gr
import numpy as np
import som
import supervision as sv
import torch
from segment_anything import sam_model_registry
from flask import Flask, request, jsonify

from sam_utils import sam_interactive_inference, sam_inference
from utils import postprocess_masks, Visualizer
from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from typing import Dict, List, Tuple, Any

app = Flask(__name__)

ANNOTATED_IMAGE_KEY = "annotated_image"
DETECTIONS_KEY = "detections"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HOME = os.getenv("HOME")

SAM_CHECKPOINT = os.path.join(HOME, "app/weights/sam_vit_h_4b8939.pth")
# SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

ANNOTATED_IMAGE_KEY = "annotated_image"
DETECTIONS_KEY = "detections"
SAM = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)

def inference(
    image_and_mask: Dict[str, np.ndarray],
    annotation_mode: List[str],
    mask_alpha: float
) -> Tuple[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]], Dict[str, Any]]:
    print("Inference")
    print(image_and_mask)
    image = image_and_mask['image']
    mask = cv2.cvtColor(image_and_mask['mask'], cv2.COLOR_RGB2GRAY)
    is_interactive = not np.all(mask == 0)
    visualizer = Visualizer(mask_opacity=mask_alpha)
    if is_interactive:
        detections = sam_interactive_inference(
            image=image,
            mask=mask,
            model=SAM)
    else:
        detections = sam_inference(
            image=image,
            model=SAM
        )
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
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    state = {
        ANNOTATED_IMAGE_KEY: annotated_image,
        DETECTIONS_KEY: detections
    }
    return (annotated_image, []), state

def decode_image(image_str):
    nparr = np.frombuffer(base64.b64decode(image_str), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/api/inference', methods=['POST'])
def inference_api():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        image = decode_image(data['image'])
        
        # Decode the image and mask from base64
        image_and_mask = {
            'image': image,
            'mask': np.zeros_like(image)
        }
        
        annotation_mode = data['annotation_mode']
        mask_alpha = data['mask_alpha']
        
        # Call the inference function
        (annotated_image, _), _ = inference(image_and_mask, annotation_mode, mask_alpha)        
        # Encode the result image back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        result_image_str = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare the response
        response = {
            'result_image': result_image_str,
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # image = cv2.imread('test.png')
    # image_and_mask = {
    #     'image': image,
    #     'mask': np.zeros_like(image)
    # }
    # mask_alpha = 0.05
    # annotation_mode = ["Mark", "Polygon"]
    # (annotated_image, _), _ = inference(image_and_mask, annotation_mode, mask_alpha)

    # cv2.imwrite('annotated_image.png', annotated_image)
    app.run(debug=True)
