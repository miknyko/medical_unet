# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
main script to train and evaluate the model
"""

from pydicom import dcmread
from pathlib import Path
import random
from model import SimpleUNet
from dataset import DicomDataset
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAccuracy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="eval the model")
parser.add_argument(
    "--eval_on_trainset", action="store_true", help="eval the model on testset"
)
parser.add_argument(
    "--model_weights",
    type=str,
    help="path to the model weights. If not provided, load the pretrained weight",
)
args = parser.parse_args()

MODEL_WEIGHTS_PATH = "./weights/saved_weights.pth"

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = SimpleUNet(in_channels=1, out_channels=1)
# load pretrained weights
if not args.model_weights:
    model.load_state_dict(torch.load("./weights/SimpleUNet_v3.pt", map_location=device))
else:
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
model.to(device)

# Image Data list
data_list = list(Path("Dataset/Images").rglob("*.dcm"))

# Mask Data
# load all mask into memory
all_masks = {}
for mask_file in Path("Dataset/Segmentations").rglob("*.npz"):
    all_masks[mask_file.stem] = np.load(mask_file)["masks"]

# Data Split
train_data, test_data = train_test_split(data_list, test_size=1 / 3, random_state=42)

# Dataset
train_dataset = DicomDataset(images_list=train_data, masks=all_masks)
test_dataset = DicomDataset(images_list=test_data, masks=all_masks)

# Dataloader

if args.train:
    bs = 3
else:
    bs = 1

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


# Loss
# custom loss function combining binary cross entropy and the soft Dice loss
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    loss = 1 - (
        (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    )
    return loss.mean()


# compute the Dice Similarity Coefficient(DSC)
# DSC = 2 * |A ∩ B| / |A| + |B|
def dice_similarity_coefficient(pred, target):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2.0 * intersection) / (pred.sum(dim=1) + target.sum(dim=1) + 1e-5)


# metric
metric = BinaryAccuracy()


def bce_dice_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target.float()) + dice_loss(
        nn.Sigmoid()(pred), target.float()
    )


all_records = {}
if args.train:
    # Train loop
    for epoch in range(10):
        model.train()
        for images, masks, _ in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            prediction = model(images)
            loss_value = bce_dice_loss(prediction, masks)
            loss_value.backward()
            optimizer.step()

            # print the train loss
            print(f"Epoch {epoch}, Loss {loss_value.item()}")

        # Test loop
        # Record and accumulate the loss and accuracy for each batch
        # After the loop, calculate the average loss and accuracy, and plot
        model.eval()
        records = []
        with torch.no_grad():
            for images, masks, _ in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                out = model(images)
                loss_value = bce_dice_loss(out, masks).item()

                # threshold the prediction to get the mask
                prediction = torch.sigmoid(out)
                prediction = prediction > 0.5
                metric.update(prediction, masks)

                # accumulate the loss for each batch
                records.append(loss_value)

            # print the loss and acc for each epoch
            print(f"Epoch {epoch}, Loss {np.mean(records)}, Acc {metric.compute()}")
            # record the loss and acc for each epoch
            all_records[epoch] = (np.mean(records), metric.compute())

    # save the model weights
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

    # plot the loss and acc on different png, and save the plot
    losses = [x[0] for x in all_records.values()]
    accs = [x[1] for x in all_records.values()]
    plt.plot(losses)
    plt.savefig("loss.png")
    plt.close()
    plt.plot(accs)
    plt.savefig("acc.png")
    plt.close()


elif args.test:
    model.eval()

    with torch.no_grad():

        best_dsc = 0
        worst_dsc = 1
        import heapq

        dscs = {
            "Case_011": [],
            "Case_010": [],
            "Case_009": [],
        }

        # eval on testset
        save_dir = Path("./result") / "testset"
        save_dir.mkdir(exist_ok=True, parents=True)
        print("\n" + f"[INFO] >>>>>>>>>>> Eval on testset ..." + "\n")
        for images, masks, filestem in tqdm(test_loader):
            metric.reset()
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)

            # threshold the prediction to get the mask
            prediction = torch.sigmoid(out)
            prediction = prediction > 0.5
            metric.update(prediction, masks)

            # save the mask as png, and write the accuracy and DSC value on the left corner of the image
            dsc = dice_similarity_coefficient(prediction, masks).mean()

            # record the best and worst DSC
            case_name, name = filestem[0].split("+")
            if case_name in dscs:
                heapq.heappush(dscs[case_name], (dsc, name))

            # plot
            # set the figure size to the size of prediction
            plt.figure(figsize=(prediction.shape[-1] / 100, prediction.shape[-2] / 100))
            plt.imshow(prediction.cpu().numpy().squeeze(), cmap="gray")
            plt.text(0, 0, f"Acc {metric.compute()}", color="red")
            plt.text(
                0,
                20,
                f"DSC {dsc}",
                color="red",
            )

            # and save the mask as png
            sub_dir, name = filestem[0].split("+")
            (save_dir / sub_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(save_dir / sub_dir / f"{name}_pred.png")
            plt.close()

        if args.eval_on_trainset:
            # eval on trainset
            save_dir = Path("./result") / "trainset"
            save_dir.mkdir(exist_ok=True, parents=True)

            print("\n" + f"[INFO] >>>>>>>>>>> Eval on train set ..." + "\n")
            for images, masks, filestem in tqdm(train_loader):
                print(images.shape, masks.shape)
                metric.reset()
                images = images.to(device)
                masks = masks.to(device)
                out = model(images)

                # threshold the prediction to get the mask
                prediction = torch.sigmoid(out)
                prediction = prediction > 0.5
                metric.update(prediction, masks)

                print(prediction.sum(), masks.sum())

                # save the mask as png, and write the accuracy and DSC value on the left corner of the image
                dsc = dice_similarity_coefficient(prediction, masks).mean()

                plt.imshow(prediction.cpu().numpy().squeeze(), cmap="gray")
                plt.text(0, 0, f"Acc {metric.compute()}", color="red")
                plt.text(0, 20, f"DSC {dsc}", color="red")
                # and save the mask as png
                sub_dir, name = filestem[0].split("+")
                (save_dir / sub_dir).mkdir(exist_ok=True, parents=True)
                plt.savefig(save_dir / sub_dir / f"{name}_pred.png")
                plt.close()

            # print the loss and acc for each epoch
            print(f"Acc {metric.compute()}")

    # Illustrate at least 3 examples
    # of each category in a figure showing the real image (left), the ground truth
    # segmentation (middle) and the prediction (right).
    # dscs = {
    #         "Case_011": [(0, "1-3"), (1, "1-4"), (2, "1-5")],
    #         "Case_010": [(0, "1-008"), (1, "1-009"), (2, "1-016")],
    #         "Case_009": [(0, "1-004"), (1, "1-005"), (2, "1-007")],
    #     }

    for case_name, record in dscs.items():
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        worst = heapq.heappop(record)[1]
        best = heapq.nlargest(1, record)[0][1]
        mid = random.choice(record)[1]
        for i, filestem in enumerate([best, worst, mid]):
            # index
            index = int(filestem.split("-")[1].strip("0")) - 1
            # raw image
            image_path = list(
                (Path("Dataset/Images") / case_name).rglob(f"{filestem}.dcm")
            )[0]
            image = dcmread(image_path)
            image = image.pixel_array
            ax[i, 0].imshow(image, cmap="gray")
            ax[i, 0].set_title("Image")
            # gt mask
            mask = all_masks[case_name + "_seg"][index]
            ax[i, 1].imshow(mask, cmap="gray")
            ax[i, 1].set_title("GroundTruth")
            # prediction mask
            pred_path = list(
                (Path("result") / "testset" / case_name).rglob(f"{filestem}_pred.png")
            )[0]
            pred = plt.imread(pred_path)
            ax[i, 2].imshow(pred, cmap="gray")
            ax[i, 2].set_title("Prediction")

        # save the figure
        plt.savefig(f"{case_name}_result.png")
