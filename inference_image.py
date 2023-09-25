import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from pathlib import Path

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model


def main():
    ROOT = Path().resolve()  # D:\SKZ\GEO_AI\deeplabv3
    # imgs_path = ROOT / 'dataset' / 'crops' / 'test' / 'images'
    imgs_path = r'D:\SKZ\GEO_AI\deeplabv3\data2009\crops'
    outs_dir = ROOT / 'outputs' / 'runs' / 'r100_190923'
    # out_dir = outs_dir / 'inference_results'
    # mask_dir = out_dir / 'masks'
    mask_dir = r'D:\SKZ\GEO_AI\deeplabv3\res2009\masks'
    out_dir = r'D:\SKZ\GEO_AI\deeplabv3\res2009\segments'

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Set computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = prepare_model(len(ALL_CLASSES))
    ckpt = torch.load(outs_dir / 'model.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)

    all_image_paths = os.listdir(imgs_path)
    for i, image_path in enumerate(all_image_paths):
        print(f"Image {i + 1}")
        # Read the image.
        image = Image.open(os.path.join(imgs_path, image_path))

        # RGBA to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize very large images (if width > 1024.) to avoid OOM on GPUs.
        if image.size[0] > 1280:
            image = image.resize((1280, 1280))

        # Do forward pass and get the output dictionary.
        outputs = get_segment_labels(image, model, device)
        # Get the data from the `out` key.
        outputs = outputs['out']
        segmented_image, mask = draw_segmentation_map(outputs)
        final_image = image_overlay(image, segmented_image)

        cv2.imwrite(os.path.join(out_dir, image_path), final_image)
        cv2.imwrite(os.path.join(mask_dir, image_path), mask)


if __name__ == '__main__':
    main()
