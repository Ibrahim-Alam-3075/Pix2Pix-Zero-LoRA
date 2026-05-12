import pytest
import torch
from PIL import Image
import numpy as np
from src.utils.gradio_utils import clean_l_sentences

def test_clean_l_sentences():
    sentences = ["1. A cat on a mat.", "2. A dog in a fog-", "3) A bird (flying)"]
    cleaned = clean_l_sentences(sentences)
    assert cleaned[0] == "A cat on a mat"
    assert cleaned[1] == "A dog in a fog"
    assert cleaned[2] == "A bird flying"

def test_image_resize_logic():
    # Simulate the resize logic in launch_main
    width, height = 1000, 500
    if width > height:
        scale_factor = 512 / width
    else:
        scale_factor = 512 / height
    new_size = (int(width * scale_factor), int(height * scale_factor))
    assert new_size == (512, 256)

def test_cuda_availability():
    # Core check for this project
    assert torch.cuda.is_available(), "CUDA must be available for Pix2Pix-Zero-Ghibli"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_latent_shape():
    z = torch.randn((1, 4, 64, 64), device="cuda")
    assert z.shape == (1, 4, 64, 64)
