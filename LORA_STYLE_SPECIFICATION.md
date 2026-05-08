# 🎨 LoRA Style Specification: The Artistic Engine of Pix2Pix-Zero

This document provides a deep-dive into the **Low-Rank Adaptation (LoRA)** implementation in this project. It explains why we added it, how it transforms the standard Pix2Pix-Zero algorithm, and the exact steps used to train and run the Ghibli style.

---

## 1. Why do we need LoRA?

### The Limitation of Standard Pix2Pix-Zero
Pix2Pix-Zero is a "Zero-shot" method. It is incredible at semantic changes (e.g., turning a **Horse** into a **Zebra**) because it understands the *concept* of those animals. However, it struggles with **Artistic Style**.
*   If you ask Pix2Pix-Zero to make something "Ghibli style" using just a text prompt, it relies on the base Stable Diffusion model's generic knowledge of Ghibli.
*   The result is often "anime-like" but lacks the specific colors, lighting, and brushwork that make a real Ghibli movie special.

### The LoRA Solution
LoRA acts as a **surgical style injection**. It teaches the U-Net the specific "aesthetic weights" of Studio Ghibli. By combining this with Pix2Pix-Zero's structural guidance, we get the best of both worlds:
1.  **Pix2Pix-Zero** keeps the original image's shape and layout perfectly.
2.  **LoRA** replaces the pixels with the exact textures and colors of a professional animation studio.

---

## 2. How LoRA Works (The Math)

Normally, training an AI model requires updating millions of parameters. This is slow and requires massive GPUs. LoRA changes this by using **Low-Rank Decomposition**.

### The Formula
For any weight matrix $W_0$ in the U-Net, we don't change it. Instead, we add a tiny update $\Delta W$:
$$W_{new} = W_0 + \Delta W$$
Where $\Delta W$ is decomposed into two smaller matrices:
$$\Delta W = B \times A$$
*   If $W_0$ is $1024 \times 1024$ ($1,048,576$ parameters).
*   $A$ is $1024 \times 4$ and $B$ is $4 \times 1024$.
*   We only train **$8,192$** parameters!
*   This is why LoRA is so fast and the files are so small (~50MB vs 5GB).

---

## 3. Training Our Ghibli LoRA

We used the `train_lora.py` script in this repository to create the Ghibli style.

### A. The Dataset
*   **Images**: 20 high-resolution stills from films like *Spirited Away*, *Howl's Moving Castle*, and *Princess Mononoke*.
*   **Preprocessing**: All images were cropped and resized to $512 \times 512$ to match the Stable Diffusion latent space.

### B. Training Hyperparameters
These are the exact settings found in our training logs:
*   **Base Model**: `stable-diffusion-v1-4`
*   **Learning Rate**: `1e-4` (A standard rate for LoRA style training).
*   **Rank (r)**: `4` (Keeps the style flexible and prevents overfitting).
*   **Alpha ($\alpha$)**: `32` (Ensures the style has a strong visual impact).
*   **Max Training Steps**: `2000` (Enough for the AI to "memorize" the Ghibli palette).
*   **Precision**: `fp16` (Used during training to save memory, though we run inference in `fp32` for stability).

### C. Why such a small dataset (20 images)?
One of the most common questions is why we used only 20 images instead of hundreds. There are three technical reasons for this:
1.  **Style vs. Content**: If you use 500 images from a movie, the AI starts to learn specific characters and specific rooms. If you use 20 diverse, high-quality stills, the AI is forced to learn the *general aesthetic* (the "vibe") rather than specific objects.
2.  **Rank Efficiency**: Our LoRA has a Rank of 4. This is a very small mathematical space. A small dataset fits perfectly into this small space, preventing "Overfitting" (where the AI just copies the training data exactly).
3.  **Training Speed**: Using 20 images allows the LoRA to be trained in under 10 minutes on a standard 3060 GPU. This allows for rapid experimentation.

### D. Parameters & Technicalities (The "Shit" you need to know)
*   **Rank (r=4)**: Think of this as the "intelligence" of the LoRA. 4 is small but smart enough for color and texture. If we used 64, the file would be 10x larger and wouldn't be any better for style transfer.
*   **Alpha (32)**: This is the "volume" knob. We set it high so the Ghibli style really pops.
*   **Injection Point**: We only inject the LoRA into the **Cross-Attention** layers. This is where the text "Studio Ghibli" meets the pixels. By focusing only on these layers, we keep the rest of the U-Net stable.

---

## 4. Implementation: How it Runs

We use the **`peft` (Parameter-Efficient Fine-Tuning)** library to merge the style into our pipeline.

### The Loading Flow
When you select "ghibli" in the UI:
1.  **Load Base Model**: The standard U-Net is loaded into the GPU.
2.  **Inject LoRA**: The `PeftModel.from_pretrained` function is called. It finds the "Cross-Attention" layers in the U-Net and "wraps" them with our LoRA matrices.
3.  **Forward Pass**: Every time the U-Net "thinks," the original weights and the LoRA weights are added together.

### Why this is better than "Fine-Tuning"
If we fine-tuned the whole model, we would have to save a 5GB file for every single style (Cyberpunk, Pixar, Ghibli). With LoRA, we just save a 50MB "adapter" file and swap it in seconds without restarting the system.

---

## 5. Parameter Tuning for Styles

In our implementation, two sliders in the UI directly affect how the LoRA behaves:

1.  **Edit Multiplier**: This scales the text embedding. Because our LoRA is trained on the prompt "Studio Ghibli style," increasing the multiplier tells the LoRA to "shout" its artistic opinion louder.
2.  **Cross-Attention Guidance**: Since LoRA can be very aggressive with its colors, this slider ensures that the LoRA doesn't "break" the original image's structure while it's painting.

---

## 🏁 Summary: The "Secret Sauce"
The reason this project produces such high-quality results is the **Stacking Effect**:
*   **CLIP** provides the semantic concept.
*   **LoRA** provides the artistic texture.
*   **Pix2Pix-Zero SGD** provides the structural cage.

**Without LoRA, it's just an edit. With LoRA, it's a masterpiece.**
