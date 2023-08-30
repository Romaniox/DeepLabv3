import torch
import torch.nn as nn
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
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    dataset_path = ROOT / 'dataset' / 'crops'
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

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    train_loss, train_pix_acc = [], []
    valid_loss, valid_pix_acc = [], []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc = validate(
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
        train_pix_acc.append(train_epoch_pixacc.cpu())
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc.cpu())

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir
        )

        print(f"\nTrain Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
        print(f"\nValid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, out_dir)
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, train_loss, valid_loss, out_dir
    )
    print('TRAINING COMPLETE')


if __name__ == '__main__':
    main()
