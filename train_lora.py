import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, LoraModel
from accelerate import Accelerator
try:
    import pillow_avif
except ImportError:
    pass

def train():
    parser = argparse.ArgumentParser(description="Simple LoRA training script")
    parser.add_argument("--instance_data_dir", type=str, default="data/ghibli", help="Path to training images")
    parser.add_argument("--output_dir", type=str, default="models/lora/ghibli", help="Where to save the model")
    parser.add_argument("--instance_prompt", type=str, default="in studio ghibli style", help="The prompt to use for training")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze VAE and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Add LoRA to UNet
    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lora_config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Dataset
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Support multiple image formats including webp and avif
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
    image_files = [
        os.path.join(args.instance_data_dir, f) 
        for f in os.listdir(args.instance_data_dir) 
        if f.lower().endswith(valid_extensions)
    ]
    
    def collate_fn(examples):
        pixel_values = [train_transforms(Image.open(f).convert("RGB")) for f in examples]
        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        
        input_ids = tokenizer(
            args.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Training loop
    unet, optimizer = accelerator.prepare(unet, optimizer)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    global_step = 0
    while global_step < args.max_train_steps:
        unet.train()
        for _ in range(len(image_files)):
            if global_step >= args.max_train_steps: break
            
            batch = collate_fn([image_files[global_step % len(image_files)]])
            
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(accelerator.device)
                
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_description(f"Loss: {loss.detach().item():.4f}")
        
    # Save the lora weights
    unet = unet.to(torch.float32)
    unet.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    train()
