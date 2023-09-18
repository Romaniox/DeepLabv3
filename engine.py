import torch
import numpy as np

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import pix_acc, get_metrics, get_tp_tn_fp_fn


def train(
        model,
        train_dataset,
        train_dataloader,
        device,
        optimizer,
        criterion,
        classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct, train_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(train_dataset) / train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0  # to keep track of batch counter
    num_classes = len(classes_to_train)

    iou_g, dice_g, precision_g, recall_g = 0, 0, 0, 0
    for i, data in enumerate(prog_bar):
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)['out']

        ##### BATCH-WISE LOSS #####
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        ###########################

        # For pixel accuracy.
        # iou, dice, precision, recall = get_metrics(target, outputs, num_classes)

        iou, dice, precision, recall = [], [], [], []
        for class_num in range(1, num_classes):
            tp, tn, fp, fn = get_tp_tn_fp_fn(target, outputs, class_num)
            iou_, dice_, precision_, recall_ = get_metrics(tp, tn, fp, fn)

            iou.append(iou_)
            dice.append(dice_)
            precision.append(precision_)
            recall.append(recall_)


            pass

        iou, dice, precision, recall = map(lambda x: np.mean(x), [iou, dice, precision, recall])
        iou_g, dice_g, precision_g, recall_g = map(lambda x, y: x + y,
                                                   [iou_g, dice_g, precision_g, recall_g],
                                                   [iou, dice, precision, recall])

        if iou is None or dice is None or precision is None or recall is None:
            pass

        #############################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        prog_bar.set_description(
            desc=f"Loss: {loss.detach().cpu().numpy():.4f} | IoU: {iou * 100:.2f}, Dice: {dice * 100:.2f},"
                 f" Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}")

    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy
    # pixel_acc = ((1.0 * train_running_correct) / (np.spacing(1) + train_running_label)) * 100
    # iou, dice, precision, recall = get_metrics(tps, tns, fps, fns)
    iou, dice, precision, recall = map(lambda x: x / counter, [iou_g, dice_g, precision_g, recall_g])
    metrics = {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}
    ##############################
    return train_loss, metrics


def validate(
        model,
        valid_dataset,
        valid_dataloader,
        device,
        criterion,
        classes_to_train,
        label_colors_list,
        epoch,
        all_classes,
        save_dir
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct, valid_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(valid_dataset) / valid_dataloader.batch_size)
    num_classes = len(classes_to_train)

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0  # To keep track of batch counter.
        tps, tns, fps, fns = 0, 0, 0, 0
        for i, data in enumerate(prog_bar):
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']

            # Save the validation segmentation maps every
            # last batch of each epoch
            if i == num_batches - 1:
                draw_translucent_seg_maps(
                    data,
                    outputs,
                    epoch,
                    i,
                    save_dir,
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = criterion(outputs, target)
            valid_running_loss += loss.item()
            ###########################

            # For pixel accuracy.
            # For pixel accuracy.
            # iou, dice, precision, recall = get_metrics(target, outputs, num_classes)
            tp, tn, fp, fn = get_tp_tn_fp_fn(target, outputs)
            tps += tp
            tns += tn
            fps += fp
            fns += fn

            iou, dice, precision, recall = get_metrics(tp, tn, fp, fn)

            #############################
            #############################

            prog_bar.set_description(
                desc=f"Loss: {loss.detach().cpu().numpy():.4f} | IoU: {iou * 100:.2f}, Dice: {dice * 100:.2f},"
                     f" Precision: {precision * 100:.2f}, Recall: {recall * 100:.2f}")

    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy
    # pixel_acc = ((1.0 * train_running_correct) / (np.spacing(1) + train_running_label)) * 100
    iou, dice, precision, recall = get_metrics(tps, tns, fps, fns)
    metrics = {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}
    ##############################

    return valid_loss, metrics
