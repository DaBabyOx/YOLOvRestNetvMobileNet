# Model Report

## Classification

### mobilenet_v3_small

**val**

- accuracy: 1.000000
- precision_micro: 1.000000
- recall_micro: 1.000000
- f1_micro: 1.000000
- precision_macro: 1.000000
- recall_macro: 1.000000
- f1_macro: 1.000000
- precision_weighted: 1.000000
- recall_weighted: 1.000000
- f1_weighted: 1.000000
- mAP: 0.000000

**medicalpill_test**

- accuracy: 1.000000
- precision_micro: 1.000000
- recall_micro: 1.000000
- f1_micro: 1.000000
- precision_macro: 1.000000
- recall_macro: 1.000000
- f1_macro: 1.000000
- precision_weighted: 1.000000
- recall_weighted: 1.000000
- f1_weighted: 1.000000
- mAP: 0.000000

### resnet18

**val**

- accuracy: 1.000000
- precision_micro: 1.000000
- recall_micro: 1.000000
- f1_micro: 1.000000
- precision_macro: 1.000000
- recall_macro: 1.000000
- f1_macro: 1.000000
- precision_weighted: 1.000000
- recall_weighted: 1.000000
- f1_weighted: 1.000000
- mAP: 0.000000

**medicalpill_test**

- accuracy: 1.000000
- precision_micro: 1.000000
- recall_micro: 1.000000
- f1_micro: 1.000000
- precision_macro: 1.000000
- recall_macro: 1.000000
- f1_macro: 1.000000
- precision_weighted: 1.000000
- recall_weighted: 1.000000
- f1_weighted: 1.000000
- mAP: 0.000000

## Detection (YOLO)

- mAP50: 0.9922203685576584
- mAP50_95: 0.7977740691347306
- precision: 0.9871870562961016
- recall: 0.9799498746867168

## Performance Degradation

### mobilenet_v3_small_vs_resnet18

**val**

- accuracy: delta=0.000000, percent=0.00%
- precision_micro: delta=0.000000, percent=0.00%
- recall_micro: delta=0.000000, percent=0.00%
- f1_micro: delta=0.000000, percent=0.00%
- precision_macro: delta=0.000000, percent=0.00%
- recall_macro: delta=0.000000, percent=0.00%
- f1_macro: delta=0.000000, percent=0.00%
- precision_weighted: delta=0.000000, percent=0.00%
- recall_weighted: delta=0.000000, percent=0.00%
- f1_weighted: delta=0.000000, percent=0.00%
- mAP: delta=0.000000, percent=n/a

**medicalpill_test**

- accuracy: delta=0.000000, percent=0.00%
- precision_micro: delta=0.000000, percent=0.00%
- recall_micro: delta=0.000000, percent=0.00%
- f1_micro: delta=0.000000, percent=0.00%
- precision_macro: delta=0.000000, percent=0.00%
- recall_macro: delta=0.000000, percent=0.00%
- f1_macro: delta=0.000000, percent=0.00%
- precision_weighted: delta=0.000000, percent=0.00%
- recall_weighted: delta=0.000000, percent=0.00%
- f1_weighted: delta=0.000000, percent=0.00%
- mAP: delta=0.000000, percent=n/a

## Notes

- Classification uses a single class label for ePill and medicalPill.
- YOLO detection metrics are reported separately from classification metrics.
