# SO-R2-X3-D Read-Only OOF Diagnostic

All findings are descriptive diagnostics on fixed internal OOF predictions. No model forward, training, calibration, threshold modification, winner selection, or OOF rewrite occurred.

B2 patient-probability mean delta vs RGB (all cases): 0.042347; upward 0.5 crossings: 51; downward crossings: 31.

AUC measures ranking across thresholds, whereas Macro-F1, balanced accuracy, sensitivity and specificity here use the frozen argmax/0.5 rule; therefore a small ranking change need not improve threshold-dependent metrics. See CSV artifacts for full class-specific transitions, probability shifts, calibration bins, fold heterogeneity, and error overlap.
