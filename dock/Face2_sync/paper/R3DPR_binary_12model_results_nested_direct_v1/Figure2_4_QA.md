# Figure 2-4 QA Notes

## Scope

- Source package: the validated four-view, three-backbone nested-direct result set (12 models, 500 pooled outer-held-out OOF predictions per model).
- Exclusions: no participant-level records were excluded. Exploratory eye2 ROI models are outside the prespecified four-view main analysis and are not included.
- Export: PNG only at 600 dpi, as specified for this project.

## Figure 2: Pooled OOF ROC Curves

- Claim: pooled outer-held-out discrimination can be compared across the three backbones within each facial input view.
- Mapping: `binary_label` is the true Control (0) / Patient (1) label; `prob_patient` is the patient probability; ROC/AUC use all 500 OOF rows for each model.
- Check file: `Figure2_AUC_check.csv`.

## Figure 3: Outer-Fold AUC Stability

- Claim: the five outer-test Macro-AUC values describe fold-to-fold stability only; they are not the primary pooled performance estimate.
- Mapping: one `fold` and one outer-test `macro_auc` per model and outer fold; `selected_epoch` is retained in the check file for traceability.
- Check files: `Figure3_fold_auc_check.csv` and `Figure3_AUC_summary_check.csv`.

## Figure 4: Pooled OOF Confusion Matrices

- Claim: hard-label error patterns are reported for the prespecified Whole-face ResNet-18 panel and the pooled-Macro-AUC-leading backbone within the Eye, Cheek, and Lip views.
- Selection rule: Whole-face is fixed to ResNet-18; Eye, Cheek, and Lip use the highest pooled `macro_auc` within their respective input views. No additional threshold tuning was performed.
- Mapping: rows are true `binary_label`, columns are softmax-argmax `pred_class`; cell colors are row-normalized proportions and annotations retain raw counts.
- Check file: `Figure4_confusion_matrix_check.csv`.

## Static and Rendered QA

- Python source parsing, fonts, color maps, dpi, and data-integrity guards passed.
- The static publication validator reports only the expected SVG/PDF/TIFF export warnings because the required deliverable is PNG only.
- Visual inspection confirmed all four panels, labels, legends, and the Figure 4 color bar render without clipping or overlap.
