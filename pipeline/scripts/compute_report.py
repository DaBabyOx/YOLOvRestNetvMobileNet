import argparse
import json
import os
from itertools import combinations

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(root, rel_path):
    return os.path.normpath(os.path.join(root, rel_path))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_degradation(baseline, candidate):
    degradation = {}
    for metric, base_value in baseline.items():
        cand_value = candidate.get(metric)
        if cand_value is None:
            continue
        delta = cand_value - base_value
        if base_value == 0:
            percent = None
        else:
            percent = (delta / base_value) * 100.0
        degradation[metric] = {
            "baseline": base_value,
            "candidate": cand_value,
            "delta": delta,
            "percent": percent,
        }
    return degradation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pipeline/configs/default.yaml")
    parser.add_argument("--repo-root", default=os.getcwd())
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = os.path.abspath(args.repo_root)
    outputs_root = resolve_path(repo_root, config["outputs"])

    classification_dir = os.path.join(outputs_root, "classification")
    yolo_dir = os.path.join(outputs_root, "yolo")

    report = {
        "classification": {},
        "detection": {},
        "comparisons": {},
        "notes": [
            "Classification uses a single class label for ePill and medicalPill.",
            "YOLO detection metrics are reported separately from classification metrics.",
        ],
    }

    classification_models = {}
    for model_name in config["classification"]["models"]:
        metrics_path = os.path.join(classification_dir, model_name, "metrics.json")
        if os.path.exists(metrics_path):
            classification_models[model_name] = load_json(metrics_path)

    report["classification"]["models"] = classification_models

    yolo_metrics_path = os.path.join(yolo_dir, "metrics.json")
    if os.path.exists(yolo_metrics_path):
        report["detection"]["yolo"] = load_json(yolo_metrics_path)

    comparisons = {"classification": {}, "detection": {}}

    for model_a, model_b in combinations(classification_models.keys(), 2):
        comparisons["classification"][f"{model_a}_vs_{model_b}"] = {
            "val": compute_degradation(
                classification_models[model_a]["val"],
                classification_models[model_b]["val"],
            ),
            "medicalpill_test": compute_degradation(
                classification_models[model_a]["medicalpill_test"],
                classification_models[model_b]["medicalpill_test"],
            ),
        }

    report["comparisons"] = comparisons

    report_path = os.path.join(outputs_root, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    markdown_path = os.path.join(outputs_root, "report.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Model Report\n\n")
        handle.write("## Classification\n\n")
        for model_name, metrics in classification_models.items():
            handle.write(f"### {model_name}\n\n")
            for split_name, split_metrics in metrics.items():
                handle.write(f"**{split_name}**\n\n")
                for key, value in split_metrics.items():
                    handle.write(f"- {key}: {value:.6f}\n")
                handle.write("\n")

        handle.write("## Detection (YOLO)\n\n")
        if "yolo" in report["detection"]:
            for key, value in report["detection"]["yolo"].items():
                if key == "raw":
                    continue
                handle.write(f"- {key}: {value}\n")
            handle.write("\n")

        handle.write("## Performance Degradation\n\n")
        for comp_name, comp_metrics in comparisons["classification"].items():
            handle.write(f"### {comp_name}\n\n")
            for split_name, split_metrics in comp_metrics.items():
                handle.write(f"**{split_name}**\n\n")
                for metric, values in split_metrics.items():
                    percent = values["percent"]
                    percent_str = "n/a" if percent is None else f"{percent:.2f}%"
                    handle.write(
                        f"- {metric}: delta={values['delta']:.6f}, percent={percent_str}\n"
                    )
                handle.write("\n")

        handle.write("## Notes\n\n")
        for note in report["notes"]:
            handle.write(f"- {note}\n")

    print(f"Saved report: {report_path}")
    print(f"Saved report: {markdown_path}")


if __name__ == "__main__":
    main()
