import argparse
import json
import os

import yaml
from ultralytics import YOLO


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(root, rel_path):
    return os.path.normpath(os.path.join(root, rel_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pipeline/configs/default.yaml")
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = os.path.abspath(args.repo_root)
    dataset_root = resolve_path(repo_root, config["dataset_root"])
    outputs_root = resolve_path(repo_root, config["outputs"])

    yolo_dir = os.path.join(outputs_root, "yolo")
    os.makedirs(yolo_dir, exist_ok=True)

    data_yaml_path = os.path.join(yolo_dir, "medicalpill.yaml")
    data_yaml = {
        "path": dataset_root,
        "train": "medicalPill/train/images",
        "val": "medicalPill/valid/images",
        "names": ["pill"],
        "nc": 1,
    }

    with open(data_yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    model = YOLO(f"{config['yolo']['variant']}.pt")

    model.train(
        data=data_yaml_path,
        imgsz=config["yolo"]["img_size"],
        epochs=config["yolo"]["epochs"],
        batch=config["yolo"]["batch"],
        workers=config["yolo"].get("workers", 2),
        project=yolo_dir,
        name="train",
        exist_ok=True,
    )

    results = model.val(
        data=data_yaml_path,
        imgsz=config["yolo"]["img_size"],
        batch=config["yolo"]["batch"],
        workers=config["yolo"].get("workers", 2),
        project=yolo_dir,
        name="val",
        exist_ok=True,
    )

    metrics = {"raw": {}}
    if hasattr(results, "results_dict"):
        metrics["raw"] = {k: float(v) for k, v in results.results_dict.items()}

    metrics["mAP50"] = metrics["raw"].get("metrics/mAP50(B)")
    metrics["mAP50_95"] = metrics["raw"].get("metrics/mAP50-95(B)")
    metrics["precision"] = metrics["raw"].get("metrics/precision(B)")
    metrics["recall"] = metrics["raw"].get("metrics/recall(B)")

    metrics_path = os.path.join(yolo_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved YOLO metrics: {metrics_path}")


if __name__ == "__main__":
    main()
