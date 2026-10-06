import os
import torch
import open_clip

def get_device():
    """Returns the optimal device available."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(model_name="ViT-B-32", pretrained="openai"):
    """
    Loads and returns the CLIP model, tokenizer and preprocessor.
    """
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def set_clip_device(model, target_device: str):
    """Mueve el modelo CLIP a la CPU o GPU para gestionar la VRAM."""
    if target_device == "cpu":
        if next(model.parameters()).device.type == 'cuda':
            model.to('cpu')
            torch.cuda.empty_cache()
    elif target_device == "cuda":
        if next(model.parameters()).device.type == 'cpu':
            model.to('cuda')
