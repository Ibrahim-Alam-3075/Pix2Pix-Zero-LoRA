import os
# Set HuggingFace Cache to D Drive
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface_cache"
import argparse
import torch
import numpy as np
from PIL import Image
import pillow_avif  # Ensure AVIF support
from src.utils.edit_pipeline import EditingPipeline
from src.utils.ddim_inv import DDIMInversion
from src.utils.scheduler import DDIMInverseScheduler
from diffusers import DDIMScheduler
from peft import PeftModel

def run_edit():
    parser = argparse.ArgumentParser(description="Pix2Pix-Zero Terminal Interface with LoRA support")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the output")
    parser.add_argument("--source", type=str, required=True, help="Source concept (e.g., 'photo' or 'cat')")
    parser.add_argument("--target", type=str, required=True, help="Target concept (e.g., 'ghibli style' or 'dog')")
    parser.add_argument("--lora", type=str, default="ghibli", help="Name of the LoRA folder in models/lora")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--multiplier", type=float, default=1.1, help="Edit multiplier")
    parser.add_argument("--guidance", type=float, default=0.1, help="Cross-attention guidance scale")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using float16 for memory efficiency (prevents crash)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} with {torch_dtype} (Memory Optimized Mode)")

    # 1. Load Image
    image = Image.open(args.input).convert("RGB").resize((512, 512))

    # 2. Inversion Stage
    print(f"Stage 1: Inverting image...")
    inv_pipe = DDIMInversion.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch_dtype
    ).to(device)
    inv_pipe.scheduler = DDIMInverseScheduler.from_config(inv_pipe.scheduler.config)
    
    # CRITICAL: We don't move VAE to float32 here, we handle it in the pipe if needed
    # But to avoid the "abstract colors", we'll just use the pipe normally in fp16
    
    prompt_str = f"a photo of a {args.source}"
    x_inv, _, _ = inv_pipe(
        prompt=prompt_str,
        img=image,
        num_inversion_steps=args.steps,
        torch_dtype=torch_dtype,
        lambda_ac=0.0,
        lambda_kl=0.0
    )
    
    del inv_pipe
    torch.cuda.empty_cache()

    # 3. Load Editing Pipeline
    print("Stage 2: Loading Editing pipeline...")
    edit_pipe = EditingPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch_dtype
    ).to(device)
    
    # 4. Load LoRA
    if args.lora and args.lora != "None":
        lora_path = os.path.join("models", "lora", args.lora)
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}...")
            edit_pipe.unet = PeftModel.from_pretrained(edit_pipe.unet, lora_path)
        else:
            print(f"Warning: LoRA path {lora_path} not found.")

    # 5. Calculate Edit Direction
    print(f"Calculating edit direction: {args.source} -> {args.target}...")
    source_sentences = [f"a photo of a {args.source}", f"a portrait of a {args.source}", f"a picture of a {args.source}"]
    target_sentences = [f"a {args.target}", f"a painting of a {args.target}", f"a beautiful {args.target}"]
    
    with torch.no_grad():
        source_input = edit_pipe.tokenizer(source_sentences, padding="max_length", max_length=edit_pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        source_embeddings = edit_pipe.text_encoder(source_input.input_ids)[0].mean(dim=0, keepdim=True)
        
        target_input = edit_pipe.tokenizer(target_sentences, padding="max_length", max_length=edit_pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        target_embeddings = edit_pipe.text_encoder(target_input.input_ids)[0].mean(dim=0, keepdim=True)
        
        edit_dir = (target_embeddings - source_embeddings) * args.multiplier

    # 6. Run Edit
    print("Redrawing image with Ghibli style...")
    gen_prompt = f"a Ghibli style painting of a {args.target}"
    
    # To prevent "abstract colors" in fp16, we ensure the guidance scale isn't extreme
    # and the VAE receives the correct types.
    _, edited_image = edit_pipe(
        prompt=gen_prompt,
        num_inference_steps=args.steps,
        x_in=x_inv,
        edit_dir=edit_dir,
        guidance_amount=args.guidance,
        guidance_scale=5.0, # Lower guidance is more stable in fp16
        negative_prompt=""
    )

    # 7. Save Result
    edited_image[0].save(args.output)
    print(f"Done! Result saved to {args.output}")

if __name__ == "__main__":
    run_edit()
