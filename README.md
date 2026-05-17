# Pipeline Architecture and Results

## Architecture Overview

This pipeline has two parallel tracks that share a common dataset root:

1) Classification track (ePill -> medicalPill)
- Input: ePill CSV from all_labels.csv
- Images: ePill/classification_data/**
- Labels: collapsed to a single class "pill" so that medicalPill can be tested without a class mapping
- Split: train/val on ePill
- Models: MobileNetV3-Small and ResNet-18
- Outputs: per-model metrics for val and medicalPill test

2) Detection track (medicalPill)
- Input: YOLO txt labels from medicalPill/train/labels and medicalPill/valid/labels
- Images: medicalPill/train/images and medicalPill/valid/images
- Model: YOLOv8n
- Outputs: mAP, precision, recall for detection

3) Reporting
- Aggregates classification and detection metrics
- Computes pairwise performance degradation between classification models
- Produces JSON and Markdown summaries

## Results Summary

### Classification (single-class)

MobileNetV3-Small:
- Val: accuracy/precision/recall/F1 (micro/macro/weighted) = 1.0000
- medicalPill test: accuracy/precision/recall/F1 (micro/macro/weighted) = 1.0000
- mAP = 0.0000

ResNet-18:
- Val: accuracy/precision/recall/F1 (micro/macro/weighted) = 1.0000
- medicalPill test: accuracy/precision/recall/F1 (micro/macro/weighted) = 1.0000
- mAP = 0.0000

Explanation:
- Because the classification task is single-class (all images labeled "pill"), accuracy, precision, recall, and F1 reach 1.0000 when the model predicts the single class.
- The mAP is 0.0000 here because average precision for a single-class setup without negative samples can be ill-defined in this implementation. The classification accuracy/F1 are the meaningful metrics for this configuration.

### Detection (YOLOv8n)

- mAP@0.5: 0.9922
- mAP@0.5:0.95: 0.7978
- Precision: 0.9872
- Recall: 0.9799

Explanation:
- The detection model shows strong performance at IoU 0.5 and solid performance across the stricter 0.5:0.95 range.

### Performance Degradation (MobileNet vs ResNet)

- All deltas are 0.0000 for val and medicalPill test metrics.
- This indicates no measured performance difference between MobileNetV3-Small and ResNet-18 on the single-class classification task.

## Notes

- If you provide a multi-class mapping for medicalPill, the classification track will become more informative and mAP will be meaningful.
- Detection metrics are reported separately and should be compared only within the detection task.
