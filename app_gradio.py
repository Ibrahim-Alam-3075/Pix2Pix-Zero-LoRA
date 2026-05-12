import os
# Set HuggingFace Cache to D Drive
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
import pdb
from PIL import Image
import gradio as gr
import pillow_avif

from src.utils.gradio_utils import *
from src.utils.huggingface_utils import *
# from utils.generate_synthetic import *
import os

def get_available_loras():
    lora_dir = os.path.join("models", "lora")
    if not os.path.exists(lora_dir):
        return ["None"]
    # return list of directories in models/lora
    loras = [d for d in os.listdir(lora_dir) if os.path.isdir(os.path.join(lora_dir, d))]
    return ["None"] + loras


if __name__=="__main__":
    with gr.Blocks(css=CSS_main, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        # Make the header of the demo website
        gr.HTML(HTML_header)

        with gr.Row():
            # col A: Input Image
            with gr.Column(scale=2):
                gr.HTML("<center><p style='font-size:150%;'>1. Upload Source Image</p></center>")
                img_in_real = gr.Image(type="pil", label="Original Photo", elem_id="input_image")
                gr.Examples(examples="assets/test_images/cats", inputs=[img_in_real])
                
                real_caption = gr.Textbox(label="Source Image Description (Override BLIP):", placeholder="e.g. A majestic snowy mountain peak (Leave blank for Auto-Caption)", interactive=True)
                
                # Synthetic hidden for now as per "LoRA only" request
                img_in_synth = gr.Image(type="pil", visible=False)
                fpath_z_gen = gr.Textbox(value="placeholder", visible=False)
                prompt = gr.Textbox(visible=False)
                seed = gr.Number(value=42, visible=False)
                negative_guidance = gr.Number(value=5, visible=False)

            # col B: Style Settings
            with gr.Column(scale=2):
                gr.HTML("<center><p style='font-size:150%;'>2. Apply Artistic Style</p></center>")
                
                lora_model = gr.Dropdown(get_available_loras(), label="Selected LoRA Style", value=get_available_loras()[1] if len(get_available_loras()) > 1 else "None", interactive=True)
                
                target_style = gr.Textbox(value="ghibli style", label="Describe the target style:", placeholder="e.g. ghibli style, oil painting, cyberpunk", interactive=True)
                
                with gr.Accordion("Optimization Sliders", open=True):
                    num_ddim = gr.Slider(20, 200, 50, label="DDIM Steps (Quality)", interactive=True, step=10)
                    xa_guidance = gr.Slider(0, 0.25, 0.15, label="Structure Preservation (Cross-Attn)", interactive=True, step=0.01)
                    edit_mul = gr.Slider(0, 2, 1.0, label="Style Strength", interactive=True, step=0.05)

                btn_edit = gr.Button(" Transform Image", variant="primary")
                
                gr.HTML("<center><p style='font-size:150%;'>3. Result</p></center>")
                img_out = gr.Image(type="pil", label="Stylized Output", visible=True)

        # Hidden fields to satisfy launch_main signature without cluttering UI
        src_preset = gr.Textbox(value="make your own!", visible=False)
        dest_preset = gr.Textbox(value="make your own!", visible=False)
        src_custom = gr.Textbox(value="photo", visible=False)
        rad_type = gr.Textbox(value="fixed-template", visible=False)
        empty_key = gr.Textbox(value="", visible=False)

        btn_edit.click(
            fn=launch_main,
            inputs=[
                img_in_real, img_in_synth,
                src_preset, src_custom, dest_preset,
                target_style, num_ddim,
                xa_guidance, edit_mul,
                fpath_z_gen, prompt,
                rad_type, rad_type,
                empty_key, empty_key,
                empty_key, empty_key,
                lora_model, real_caption
            ],
            outputs=[img_out]
        )
        
        gr.HTML("<hr><center>Developed for the Pix2Pix-Zero-Ghibli Research Paper</center>")

    gr.close_all()
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=True, debug=True)
