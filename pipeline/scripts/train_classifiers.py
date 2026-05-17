import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import yaml


@dataclass
class SplitPaths:
    train_csv: str
    val_csv: str
    test_csv: str
    label_map: str


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(root, rel_path):
    return os.path.normpath(os.path.join(root, rel_path))


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ImageCsvDataset(Dataset):
    def __init__(self, csv_path, label_map, transform):
        self.df = pd.read_csv(csv_path)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.label_map[row["label"]]
        return self.transform(image), torch.tensor(label, dtype=torch.float32)


def build_model(model_name, num_classes):
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    for avg in ["micro", "macro", "weighted"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=avg,
            zero_division=0,
        )
        metrics[f"precision_{avg}"] = precision
        metrics[f"recall_{avg}"] = recall
        metrics[f"f1_{avg}"] = f1

    metrics["mAP"] = average_precision_score(y_true, y_prob)
    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    return compute_metrics(y_true, y_prob)


def train_one_epoch(model, dataloader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(1, len(dataloader.dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pipeline/configs/default.yaml")
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = os.path.abspath(args.repo_root)
    outputs_root = resolve_path(repo_root, config["outputs"])
    split_dir = os.path.join(outputs_root, "classification")

    split_paths = SplitPaths(
        train_csv=os.path.join(split_dir, "epill_train.csv"),
        val_csv=os.path.join(split_dir, "epill_val.csv"),
        test_csv=os.path.join(split_dir, "medicalpill_test.csv"),
        label_map=os.path.join(split_dir, "label_map.json"),
    )

    label_map = load_label_map(split_paths.label_map)
    num_classes = len(label_map)

    img_size = config["classification"]["img_size"]
    batch_size = config["classification"]["batch_size"]
    num_workers = config["classification"]["num_workers"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageCsvDataset(split_paths.train_csv, label_map, train_transform)
    val_ds = ImageCsvDataset(split_paths.val_csv, label_map, eval_transform)
    test_ds = ImageCsvDataset(split_paths.test_csv, label_map, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in config["classification"]["models"]:
        model = build_model(model_name, num_classes)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["classification"]["lr"])

        for _ in range(config["classification"]["epochs"]):
            train_one_epoch(model, train_loader, device, criterion, optimizer)

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        model_dir = os.path.join(outputs_root, "classification", model_name)
        os.makedirs(model_dir, exist_ok=True)

        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump({
                "val": val_metrics,
                "medicalpill_test": test_metrics,
            }, handle, indent=2)

        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
