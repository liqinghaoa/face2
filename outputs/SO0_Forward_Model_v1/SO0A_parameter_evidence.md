# SO-0A Parameter Evidence

The MVP parameters are frozen in `configs/so0/forward_model_mvp.yaml`.
They define synthetic melanin-sensitive and hemoglobin-sensitive controls only.
No patient image, ROI, disease label, EXIF field, fold assignment, or classifier
result is used to define any optical formula, parameter, or threshold.

The Virtual ColorChecker path is fixed as a no-intercept linear 3x3 least-squares
matrix. There is no ridge term, polynomial expansion, MLP, intercept, or
trainable camera/color degree of freedom.
