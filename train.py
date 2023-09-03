import torch
import torch.nn as nn
import pandas as pd
import os
from pathlib import Path

from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from model import prepare_model
from config import *
from utils import save_model, SaveBestModel, save_plots, get_save_path


def main():
    # Create a directory with the model name for outputs.
    ROOT = Path().resolve()
    out_dir = get_save_path(ROOT / 'outputs')
    out_dir_valid_preds = out_dir / 'valid_preds'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)

    # If multi GPU mode
    if MULTI_GPU_MODE and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    dataset_path = ROOT / 'dataset' / dataset_name
    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=dataset_path
    )

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=IMG_SIZE
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=BATCH
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=0.8 * EPOCHS, gamma=0.1
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    train_loss, train_metrics_all = [], []
    valid_loss, valid_metrics_all = [], []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_metrics = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_metrics = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=out_dir_valid_preds
        )
        train_loss.append(train_epoch_loss)
        train_metrics_all.append(train_metrics)
        valid_loss.append(valid_epoch_loss)
        valid_metrics_all.append(valid_metrics)

        scheduler.step()

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir
        )

        print(f"\nTrain Epoch Loss: {train_epoch_loss:.4f}, "
              f"Train Epoch Dice: {train_metrics['dice']:.4f}, "
              f"Train Epoch IoU: {train_metrics['iou']:.4f}")
        print(f"\nValid Epoch Loss: {valid_epoch_loss:.4f}, "
              f"Valid Epoch Dice: {valid_metrics['dice']:.4f}, "
              f"Valid Epoch IoU: {valid_metrics['iou']:.4f}")
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, out_dir)
    # Save the loss and accuracy plots.
    save_plots(
        train_metrics_all, valid_metrics_all, train_loss, valid_loss, out_dir
    )

    df_train = pd.DataFrame(train_metrics_all)
    df_valid = pd.DataFrame(valid_metrics_all)
    df_loss = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss})

    df_train.to_csv(out_dir / 'train_metrics.csv', index=True, sep=';')
    df_valid.to_csv(out_dir / 'valid_metrics.csv', index=True, sep=';')
    df_loss.to_csv(out_dir / 'loss.csv', index=True, sep=';')

    print('TRAINING COMPLETE')


if __name__ == '__main__':
    main()
