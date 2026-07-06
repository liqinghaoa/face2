你现在在 Windows + PyCharm 项目中工作，项目根目录为：

E:/projects/face2

本次任务不是重新训练模型，而是整理和诊断“多 Backbone 模型探索实验”的结果，生成一份可以反馈给科研助手进一步分析的完整报告。

请不要修改训练代码，不要重新训练模型，不要移动或删除任何实验结果文件。

========================================
一、分析目标
========================================

当前已经完成多 backbone 模型探索实验，实验固定：

- preprocessing = hybrid_imagenet_meanbg
- split_dir = data/processed/splits_500
- image_root = data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images
- loss = weighted_cross_entropy
- optimizer = AdamW
- lr = 1e-4
- weight_decay = 1e-4
- epochs = 50
- early_stopping = macro_auc
- augmentation 不变
- 五折划分不变

已完成模型包括：

1. resnet18_meanbg baseline
2. densenet121
3. efficientnet_b0
4. convnext_tiny
5. swin_t
6. mobilenet_v3_large

本次需要重点诊断：

1. 为什么 Swin-Tiny macro-AUC 和 balanced accuracy 最高，但 macro-F1 低；
2. Swin-Tiny 的 mild 类是否被大量误判；
3. EfficientNet-B0 是否适合作为轻量候选；
4. DenseNet121 是否只是 AUC 提升但 hard classification 没改善；
5. ConvNeXt-Tiny 是否只是 severe recall 高但整体不平衡；
6. MobileNetV3-Large 是否可以停止推进；
7. 下一步是否应围绕 Swin-Tiny 做阈值扫描或轻量调参。

========================================
二、需要读取的文件
========================================

请自动扫描以下目录：

E:/projects/face2/experiments/model_exploration_500Data/

读取：

1. model_exploration_summary.csv
2. model_exploration_summary.md
3. model_exploration_job_queue.csv

并扫描每个模型实验目录下的：

summary/fold_metrics_all.csv
summary/mean_metrics.csv
summary/oof_metrics.csv
summary/oof_predictions.csv
summary/summary_report.md

同时读取 ResNet18 meanbg 基线目录：

E:/projects/face2/experiments/preprocess_ablation_500Data/PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg

读取其：

summary/fold_metrics_all.csv
summary/mean_metrics.csv
summary/oof_metrics.csv
summary/oof_predictions.csv
summary/summary_report.md

如果某些文件不存在，请在报告中明确列出缺失路径，不要静默跳过。

========================================
三、输出目录
========================================

请输出到：

E:/projects/face2/experiments/model_exploration_500Data/diagnostic_analysis/

生成以下文件：

1. model_exploration_diagnostic_report.md
2. model_exploration_diagnostic_tables.xlsx
3. model_exploration_confusion_summary.csv
4. model_exploration_argmax_metrics.csv
5. model_exploration_per_class_metrics.csv
6. model_exploration_oof_auc_summary.csv

如果可以，也生成图片：

figures/<model_name>_confusion_count.png
figures/<model_name>_confusion_row_normalized.png

图片用 matplotlib，不要使用 seaborn。

========================================
四、需要计算的内容
========================================

请对以下模型全部计算：

- resnet18_meanbg
- densenet121
- efficientnet_b0
- convnext_tiny
- swin_t
- mobilenet_v3_large

类别定义固定为：

0 = normal
1 = mild
2 = severe

----------------------------------------
1. OOF argmax 三分类指标
----------------------------------------

对每个模型读取 oof_predictions.csv。

要求字段至少包括：

label_3class
pred_class
prob_normal
prob_mild
prob_severe

如果 pred_class 不存在，则使用：

argmax(prob_normal, prob_mild, prob_severe)

重新生成 pred_class。

计算每个模型的 OOF 三分类指标：

accuracy
macro_precision
macro_recall
macro_f1
balanced_accuracy
weighted_f1
recall_normal
recall_mild
recall_severe
precision_normal
precision_mild
precision_severe
f1_normal
f1_mild
f1_severe

输出：

model_exploration_argmax_metrics.csv

----------------------------------------
2. OOF AUC 指标
----------------------------------------

基于 oof_predictions.csv 计算：

macro_auc_ovr
weighted_auc_ovr
auc_normal
auc_mild
auc_severe
normal_vs_abnormal_auc
severe_vs_rest_auc

注意：

normal_vs_abnormal:
y_true = 1 if label_3class == 0 else 0
score = prob_normal

severe_vs_rest:
y_true = 1 if label_3class == 2 else 0
score = prob_severe

输出：

model_exploration_oof_auc_summary.csv

----------------------------------------
3. 混淆矩阵
----------------------------------------

对每个模型计算 OOF argmax 混淆矩阵：

rows = true label
cols = predicted label

类别顺序：

normal, mild, severe

输出两种：

1. count confusion matrix
2. row-normalized confusion matrix

重点统计以下误判：

true normal -> pred mild
true normal -> pred severe
true mild -> pred normal
true mild -> pred severe
true severe -> pred normal
true severe -> pred mild

输出：

model_exploration_confusion_summary.csv

字段至少包括：

model_name
n_total
normal_total
mild_total
severe_total

normal_to_normal
normal_to_mild
normal_to_severe
normal_recall

mild_to_normal
mild_to_mild
mild_to_severe
mild_recall

severe_to_normal
severe_to_mild
severe_to_severe
severe_recall

normal_to_severe_rate
mild_to_severe_rate
severe_to_normal_rate
severe_to_mild_rate

----------------------------------------
4. 专门分析 Swin-Tiny
----------------------------------------

请重点分析 Swin-Tiny 的混淆矩阵和类别指标。

需要回答：

1. Swin-Tiny 的 macro-AUC 是否最高？
2. Swin-Tiny 的 balanced accuracy 是否最高？
3. Swin-Tiny 的 macro-F1 为什么低？
4. Swin-Tiny 的 mild recall 和 mild F1 是多少？
5. Swin-Tiny 的 mild 类主要被误判为 normal 还是 severe？
6. Swin-Tiny 是否牺牲 mild 类换取 normal/severe recall？
7. Swin-Tiny 的 severe recall 是否优于 ResNet18 meanbg？
8. Swin-Tiny 的 severe-vs-rest AUC 是否优于 ResNet18 meanbg？
9. Swin-Tiny 是否值得进入第二轮调参？
10. 如果值得，优先调哪些参数？

----------------------------------------
5. 专门分析 EfficientNet-B0
----------------------------------------

需要回答：

1. EfficientNet-B0 参数量是多少？
2. EfficientNet-B0 是否明显轻于 ResNet18？
3. EfficientNet-B0 的 macro-AUC、BA、macro-F1、severe recall 是否接近 ResNet18 meanbg？
4. EfficientNet-B0 是否适合作为轻量候选？
5. 是否值得进入第二轮轻量调参？

----------------------------------------
6. 专门分析 DenseNet121
----------------------------------------

需要回答：

1. DenseNet121 的 macro-AUC 是否高于 ResNet18 meanbg？
2. DenseNet121 的 BA、macro-F1、severe recall 是否下降？
3. DenseNet121 是否只是概率排序提升，但 hard classification 未改善？
4. 是否建议继续调参？

----------------------------------------
7. 专门分析 ConvNeXt-Tiny
----------------------------------------

需要回答：

1. ConvNeXt-Tiny 的 severe recall 是否最高？
2. 它的 macro-F1、BA 是否明显下降？
3. 它是否属于 severe-sensitive but unbalanced model？
4. 是否建议作为主模型推进？
5. 是否只适合保留为 severe-vs-rest 探索模型？

----------------------------------------
8. 专门分析 MobileNetV3-Large
----------------------------------------

需要回答：

1. MobileNetV3-Large 是否整体低于 ResNet18 meanbg？
2. 它的 severe recall 是否明显偏低？
3. 是否建议停止推进？
4. 如果考虑轻量化，是否 EfficientNet-B0 优先于 MobileNetV3-Large？

----------------------------------------
5-fold 稳定性分析
----------------------------------------

基于 fold_metrics_all.csv，计算每个模型：

macro_auc_mean
macro_auc_std
balanced_accuracy_mean
balanced_accuracy_std
macro_f1_mean
macro_f1_std
recall_severe_mean
recall_severe_std

并回答：

1. 哪个模型 fold 间波动最大？
2. 哪个模型最稳定？
3. Swin-Tiny 的提升是否稳定，还是由个别 fold 拉高？
4. ConvNeXt-Tiny 的 severe recall 是否在各 fold 稳定？
5. EfficientNet-B0 是否稳定但上限有限？

========================================
五、报告格式要求
========================================

请生成 Markdown 报告：

model_exploration_diagnostic_report.md

报告必须包含以下结构：

# Model Exploration Diagnostic Report

## 1. Files Checked
列出读取到的文件和缺失文件。

## 2. Experiment Status
根据 model_exploration_job_queue.csv 总结每个模型是否 SUCCESS，训练耗时，参数量。

## 3. Overall Result Table
给出总表，包含：

model_name
total_params
macro_auc_mean
balanced_accuracy_mean
macro_f1_mean
recall_severe_mean
oof_macro_auc
oof_balanced_accuracy
oof_macro_f1
oof_recall_severe
recommendation

## 4. Comparison with ResNet18 MeanBG
以 ResNet18 meanbg 为基线，列出：

delta_macro_auc
delta_balanced_accuracy
delta_macro_f1
delta_recall_severe
delta_oof_macro_auc
delta_oof_balanced_accuracy
delta_oof_macro_f1

## 5. Per-class Performance
重点展示 normal、mild、severe 的 precision、recall、F1。

## 6. Confusion Matrix Analysis
逐模型说明主要误判方向。

必须重点回答：

- Swin-Tiny 的 mild 类被误判到哪里？
- ConvNeXt-Tiny 是否增加了 severe 预测倾向？
- MobileNetV3-Large 是否漏判 severe 严重？

## 7. Swin-Tiny Focused Analysis
详细解释 Swin-Tiny 为什么 AUC/BA 好但 macro-F1 不高。

## 8. Candidate Models for Next Stage
按优先级给出：

1. 主线候选
2. 轻量候选
3. CNN 对照
4. 不推荐继续推进模型

## 9. Suggested Next Experiments
给出下一步实验建议，但不要直接实现。

建议包括：

- Swin-Tiny OOF threshold scan
- Swin-Tiny lr=5e-5
- Swin-Tiny label smoothing 0.05
- EfficientNet-B0 label smoothing 0.05
- 若 backbone 改进有限，转向 ordinal classification / two-stage classification / ROI-global fusion

## 10. Final Conclusion
用中文总结：

1. 是否有模型全面超过 ResNet18 meanbg；
2. Swin-Tiny 是否值得继续；
3. 当前瓶颈是否仍主要在 hard decision / mild 类边界；
4. 下一步应优先做什么。

========================================
六、额外输出：给科研助手看的摘要
========================================

请在报告最后添加一个专门板块：

## Key Information for Further Analysis

请用非常直接的条目列出以下信息：

1. 最优 macro-AUC 模型：
2. 最优 BA 模型：
3. 最优 macro-F1 模型：
4. 最优 severe recall 模型：
5. Swin-Tiny mild recall：
6. Swin-Tiny mild F1：
7. Swin-Tiny mild -> normal 数量和比例：
8. Swin-Tiny mild -> severe 数量和比例：
9. Swin-Tiny severe -> mild 数量和比例：
10. Swin-Tiny severe -> normal 数量和比例：
11. Swin-Tiny severe-vs-rest AUC：
12. ResNet18 meanbg severe-vs-rest AUC：
13. 是否推荐 Swin-Tiny 进入二轮调参：
14. 是否推荐 EfficientNet-B0 作为轻量候选：
15. 是否建议停止 MobileNetV3-Large：
16. 下一步最推荐的 3 个实验：

这个板块是我之后反馈给科研助手分析用的，必须清楚、完整、不要省略。

========================================
七、实现方式建议
========================================

可以新增脚本：

scripts/evaluate/diagnose_model_exploration_results.py

运行命令：

cd /d E:\projects\face2

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\evaluate\diagnose_model_exploration_results.py

脚本运行后输出：

experiments/model_exploration_500Data/diagnostic_analysis/model_exploration_diagnostic_report.md

请最终在 Codex 回复中直接给出：

1. 新增脚本路径；
2. 输出报告路径；
3. 是否成功读取所有模型；
4. 是否发现缺失文件；
5. Key Information for Further Analysis 板块全文。

注意：
本任务只做结果诊断，不重新训练，不修改训练结果。