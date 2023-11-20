import gradio as gr

from detectron2.data import MetadataCatalog
from segment_anything import SamAutomaticMaskGenerator


metadata = MetadataCatalog.get('coco_2017_train_panoptic')
print(metadata)


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


demo = gr.Blocks()
image = ImageMask(
    label="Input",
    type="pil",
    brush_radius=20.0,
    brush_color="#FFFFFF")
slider = gr.Slider(
    minimum=1,
    maximum=3,
    value=2,
    label="Granularity",
    info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]")
mode = gr.Radio(
    choices=['Automatic', 'Interactive', ],
    value='Automatic',
    label="Segmentation Mode")
image_out = gr.Image(label="Auto generation", type="pil")
slider_alpha = gr.Slider(
    minimum=0,
    maximum=1,
    value=0.1,
    label="Mask Alpha",
    info="Choose in [0, 1]")
label_mode = gr.Radio(
    choices=['Number', 'Alphabet'],
    value='Number',
    label="Mark Mode")
anno_mode = gr.CheckboxGroup(
    choices=["Mask", "Box", "Mark"],
    value=['Mask', 'Mark'],
    label="Annotation Mode")
runBtn = gr.Button("Run")

title = "Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V"
description = "This is a demo for SoM Prompting to unleash extraordinary visual grounding in GPT-4V. Please upload an image and them click the 'Run' button to get the image with marks. Then try it on <a href='https://chat.openai.com/'>GPT-4V<a>!"

with demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
    gr.Markdown("<h3 style='text-align: center; margin-bottom: 1rem'>project: <a href='https://som-gpt4v.github.io/'>link</a>, arXiv: <a href='https://arxiv.org/abs/2310.11441'>link</a>, code: <a href='https://github.com/microsoft/SoM'>link</a></h3>")
    gr.Markdown(f"<h3 style='margin-bottom: 1rem'>{description}</h3>")
    with gr.Row():
        with gr.Column():
            image.render()
            slider.render()
            with gr.Row():
                mode.render()
                anno_mode.render()
            with gr.Row():
                slider_alpha.render()
                label_mode.render()
        with gr.Column():
            image_out.render()
            runBtn.render()
    # with gr.Row():
    #     example = gr.Examples(
    #         examples=[
    #             ["examples/ironing_man.jpg"],
    #         ],
    #         inputs=image,
    #         cache_examples=False,
    #     )
    #     example = gr.Examples(
    #         examples=[
    #             ["examples/ironing_man_som.png"],
    #         ],
    #         inputs=image,
    #         cache_examples=False,
    #         label='Marked Examples',
    #     )

    # runBtn.click(inference, inputs=[image, slider, mode, slider_alpha, label_mode, anno_mode],
    #           outputs = image_out)

demo.queue().launch()
