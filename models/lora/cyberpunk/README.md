# Cyberpunk 2077 Style LoRA for Pix2Pix-Zero

This LoRA (Low-Rank Adaptation) model is designed to transform real-world photos into a neon-drenched, high-contrast **Cyberpunk** aesthetic. It was trained on 20 purely in-game screenshots from _Cyberpunk 2077_ to ensure maximum engine-fidelity.

## Model Details

- **Base Model**: Stable Diffusion v1-4
- **Training Data**: 20 high-quality in-game screenshots (Night City cityscapes, character models, interior environments).
- **Style Characteristics**: Teal and orange color palette, neon lighting, wet pavement reflections, gritty digital textures.

## Usage with Pix2Pix-Zero-LoRA

Select `cyberpunk` from the LoRA dropdown in the Gradio UI or use the `--lora cyberpunk` flag in the CLI.

### Recommended Parameters:

- **Edit Multiplier**: 1.4
- **Cross-Attention Guidance**: 0.4
- **DDIM Steps**: 170
- **Target Prompt**: "cyberpunk style", "neon lights", "night city aesthetic"

## Example Results

| Source City | Cyberpunk Output | Source Street | Cyberpunk Output |
| :---: | :---: | :---: | :---: |
| <img src="../../../data/inputs/city.avif" width="200"> | <img src="../../../results/cyberpunk/city.png" width="200"> | <img src="../../../data/inputs/street.png" width="200"> | <img src="../../../results/cyberpunk/street.png" width="200"> |

### 🔬 Research Case: Hallucination
*Example of the model hallucinating a person due to automated captioning errors:*
| Source Mountain | Snowboarder Hallucination |
| :---: | :---: |
| <img src="../../../data/inputs/mountain.png" width="300"> | <img src="../../../results/cyberpunk/mountain(erroneous).png" width="300"> |

---

_Part of the Pix2Pix-Zero-LoRA Framework._
