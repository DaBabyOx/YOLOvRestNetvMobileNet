import argparse
import json
import os
from glob import glob

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(root, rel_path):
    return os.path.normpath(os.path.join(root, rel_path))


def build_epill_dataframe(dataset_root, epill_cfg, label_map):
    csv_path = resolve_path(dataset_root, epill_cfg["csv"])
    image_base = resolve_path(dataset_root, epill_cfg["image_base"])

    df = pd.read_csv(csv_path)
    df = df[["image_path"]].copy()
    df["image_path"] = df["image_path"].apply(lambda p: os.path.join(image_base, p))
    df["label"] = list(label_map.keys())[0]
    return df


def build_medicalpill_dataframe(dataset_root, medical_cfg, label_name):
    image_paths = []
    for rel_dir in medical_cfg["images"]:
        abs_dir = resolve_path(dataset_root, rel_dir)
        image_paths.extend(glob(os.path.join(abs_dir, "*.jpg")))
        image_paths.extend(glob(os.path.join(abs_dir, "*.png")))

    df = pd.DataFrame({"image_path": sorted(image_paths)})
    df["label"] = label_name
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pipeline/configs/default.yaml")
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = os.path.abspath(args.repo_root)
    dataset_root = resolve_path(repo_root, config["dataset_root"])

    label_map = config.get("label_map", {"pill": 0})
    label_name = list(label_map.keys())[0]

    epill_df = build_epill_dataframe(dataset_root, config["epill"], label_map)
    train_df, val_df = train_test_split(
        epill_df,
        train_size=config["epill"]["train_split"],
        random_state=config["epill"]["seed"],
        shuffle=True,
        stratify=None,
    )

    medical_df = build_medicalpill_dataframe(dataset_root, config["medicalpill"], label_name)

    output_root = resolve_path(repo_root, config["outputs"])
    output_dir = os.path.join(output_root, "classification")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "epill_train.csv")
    val_path = os.path.join(output_dir, "epill_val.csv")
    test_path = os.path.join(output_dir, "medicalpill_test.csv")
    label_path = os.path.join(output_dir, "label_map.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    medical_df.to_csv(test_path, index=False)

    with open(label_path, "w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    print("Saved classification splits:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()
