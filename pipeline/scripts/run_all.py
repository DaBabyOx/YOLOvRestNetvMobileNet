import os
import subprocess
import sys


def run_step(args):
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    repo_root = os.getcwd()
    config_path = os.path.join("pipeline", "configs", "default.yaml")

    run_step([sys.executable, "pipeline/scripts/prepare_classification_data.py", "--config", config_path, "--repo-root", repo_root])
    run_step([sys.executable, "pipeline/scripts/train_classifiers.py", "--config", config_path, "--repo-root", repo_root])
    run_step([sys.executable, "pipeline/scripts/run_yolo.py", "--config", config_path, "--repo-root", repo_root])
    run_step([sys.executable, "pipeline/scripts/compute_report.py", "--config", config_path, "--repo-root", repo_root])


if __name__ == "__main__":
    main()
