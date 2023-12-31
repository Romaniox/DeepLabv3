import torch
import torch.nn as nn
import cv2
import os
from pathlib import Path

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model


def main():
    ROOT = Path().resolve()
    imgs_path = ROOT / 'dataset' / 'crops' / 'test' / 'images'
    outs_dir = ROOT / 'outputs' / 'runs' / 'r50_30082023'
    out_dir = outs_dir / 'inference_results'
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

        # Resize very large images (if width > 1024.) to avoid OOM on GPUs.
        if image.size[0] > 1024:
            image = image.resize((800, 800))

        # Do forward pass and get the output dictionary.
        outputs = get_segment_labels(image, model, device)
        # Get the data from the `out` key.
        outputs = outputs['out']
        segmented_image = draw_segmentation_map(outputs)

        final_image = image_overlay(image, segmented_image)
        cv2.imshow('Segmented image', final_image)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(out_dir, image_path), final_image)


if __name__ == '__main__':
    main()
