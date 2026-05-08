import os, pdb
from glob import glob
import argparse
import numpy as np
import torch
import requests
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='assets/test_images/cat_a.png')
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_float_16', action='store_true')
    args = parser.parse_args()

    # make the output folders
    os.makedirs(os.path.join(args.results_folder, "inversion"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "prompt"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32


    # load the BLIP model
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # make the DDIM inversion pipeline    
    pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)


    # if the input is a folder, collect all the images as a list
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]


    for img_path in l_img_paths:
        bname = os.path.basename(img_path).split(".")[0]
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        # generate the caption
        _inputs = processor_blip(img, return_tensors="pt").to(device)
        out = model_blip.generate(**_inputs)
        prompt_str = processor_blip.decode(out[0], skip_special_tokens=True)
        x_inv, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype
        )
        # save the inversion
        torch.save(x_inv[0], os.path.join(args.results_folder, f"inversion/{bname}.pt"))
        # save the prompt string
        with open(os.path.join(args.results_folder, f"prompt/{bname}.txt"), "w") as f:
            f.write(prompt_str)
