from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from BiRefNet.models.birefnet import BiRefNet

_IMAGE_SIZE = (1024, 1024)


def load_model(device: str = "cuda") -> BiRefNet:
    model = BiRefNet.from_pretrained("ZhengPeng7/BiRefNet")
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device).half()
        torch.set_float32_matmul_precision("high")
    else:
        model = model.to(device)
    model.eval()
    print("[MODEL] Loaded model on", device)
    return model


def extract_object(birefnet: BiRefNet, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    transform_image = transforms.Compose([
        transforms.Resize(_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = transform_image(image).unsqueeze(0)
    if next(birefnet.parameters()).device.type == "cuda":
        input_tensor = input_tensor.to("cuda").half()

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    mask = preds[0].squeeze()
    mask_image = transforms.ToPILImage()(mask)
    mask_image = mask_image.resize(image.size)
    image.putalpha(mask_image)

    return image, mask_image
