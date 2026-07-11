你现在在 Windows + PyCharm 项目中工作，项目根目录为：

E:/projects/face2

本次任务目标：
实现并运行 Global + selected ROI feature-level fusion 实验，用于验证全局人脸图像与关键局部 ROI 是否存在互补信息，尤其是否改善 NYHA 三分类中的 mild 类边界和整体 macro-F1。

本轮实验不是继续做 roi_single，也不是继续做 ROI-only multiroi5_fusion，而是新增：

Global face + Eye ROI
Global face + Cheek ROI
Global face + Eye ROI + Cheek ROI

三组特征级融合实验。

请严格遵守以下原则：

1. 不重新划分数据。
2. 不重新生成标签。
3. 不修改已有实验结果。
4. 不覆盖已有 ROI_500、ROI_Fusion_500、model_exploration_500Data、preprocess_ablation_500Data 结果。
5. 不安装新依赖。
6. 不使用 timm。
7. 第一版只使用 ResNet18。
8. 不使用 Swin。
9. 不加入 label smoothing。
10. 不加入 focal loss。
11. 不加入 ColorJitter、RandomCrop、RandomRotation。
12. 不加入阈值优化作为训练结果。
13. 所有训练严格串行。
14. 必须保持 patient_group_id 折间独立。
15. 必须保证同一样本的 global、eye、cheek 使用一致的随机水平翻转决策。
16. 优先新增独立代码，不要破坏已有 global-only、roi_single、roi_fusion 实验。

========================================
一、实验背景
========================================

当前已有实验结论：

1. Global ResNet18 meanbg 是当前 hard-decision 强基线：
   - macro-F1 = 0.5111
   - BA = 0.5353
   - mild recall = 0.4770
   - severe recall = 0.4333

2. ROI single 中，cheek 和 eye 是最有价值的两个 ROI：
   - cheek / resnet34:
     Macro-F1 = 0.5076
     BA = 0.5447
     severe recall = 0.4382
   - eye / resnet18:
     Macro-F1 = 0.5072
     mild recall = 0.5238

3. 已有 ROI-only multiroi5_fusion 没有超过同 backbone 下的最佳单 ROI。

因此本轮目标是验证：

global face + selected ROI 是否比 global-only 和 ROI-only 更好。

核心假设：

global face 提供整体面部状态；
eye ROI 补充 mild 类边界信息；
cheek ROI 补充面色、皮肤状态和整体/severe 相关信息；
三者特征级融合可能改善 macro-F1、BA、mild recall 和 severe recall 的平衡。

========================================
二、本轮要实现的三个实验
========================================

请实现并运行以下 3 个实验：

1. GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold

输入：
- global face
- eye ROI

2. GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold

输入：
- global face
- cheek ROI

3. GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold

输入：
- global face
- eye ROI
- cheek ROI

统一设置：

backbone = resnet18
pretrained = imagenet
loss = weighted_cross_entropy
optimizer = AdamW
lr = 0.0001
weight_decay = 0.0001
epochs = 50
early_stopping_patience = 10
monitor_metric = macro_auc
random_seed = 2026
batch_size = 8
num_workers = 0
pin_memory = false
use_amp = false

========================================
三、固定数据与路径
========================================

项目根目录：

E:/projects/face2

五折划分目录：

data/processed/splits_500

全局人脸图像目录：

data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images

类别定义：

0 = normal
1 = mild
2 = severe

标签列：

label_3class

图像文件名模板：

{ID}.png

本轮输出根目录：

experiments/global_roi_fusion_500Data

配置输出目录：

config/train/global_roi_fusion

注意：
eye ROI 和 cheek ROI 的真实图像目录不要凭空假设。请先从已有 ROI single / ROI fusion 配置或实验输出中自动识别。

请优先检查以下位置是否存在 ROI 配置或保存的 config：

config/train/roi/
config/train/roi_single/
config/train/roi_fusion/
experiments/ROI_500/*/config.yaml
experiments/ROI_Fusion_500/*/config.yaml

请查找 ROI 为 eye 和 cheek 的实验配置，读取其 image_root 或 roi_root。

如果找不到明确路径，请终止并报告：
Could not locate eye/cheek ROI image root.
不要随意猜路径。

========================================
四、需要先阅读的已有代码
========================================

实现前请先阅读并理解：

scripts/train/train_nyha_3class_5fold.py
scripts/run/run_nyha3class_5fold_with_config.py
models/nyha_backbone_factory.py
models/resnet_nyha_3class.py
datasets/nyha_3class_face_dataset.py
metrics/classification_metrics.py
losses/classification_losses.py
scripts/evaluate/summarize_nyha_3class_5fold.py

同时重点查找并阅读已有 ROI 相关代码，例如：

datasets/*roi*.py
models/*roi*.py
trainers/*roi*.py
evaluators/*roi*.py
scripts/train/*roi*.py
scripts/run/*roi*.py
scripts/evaluate/*roi*.py

如果已有 ROI-only fusion 代码可复用，请尽量复用其数据读取、指标计算和汇总逻辑。但不要直接破坏旧代码。

========================================
五、模型实现
========================================

请新增模型文件：

models/global_roi_fusion_model.py

实现一个多输入特征级融合模型：

class GlobalROIFusionModel(nn.Module):
    ...

建议接口：

class GlobalROIFusionModel(
    backbone: str = "resnet18",
    num_classes: int = 3,
    pretrained: bool | str = True,
    enabled_inputs: list[str] = ["global", "eye", "cheek"],
    projection_dim: int = 256,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
)

支持 enabled_inputs：

["global", "eye"]
["global", "cheek"]
["global", "eye", "cheek"]

第一版只要求 backbone=resnet18，但代码可保留扩展性。

模型结构：

每个输入一个独立 backbone，不共享权重。

global image → ResNet18 feature extractor → 512-d feature
eye image → ResNet18 feature extractor → 512-d feature
cheek image → ResNet18 feature extractor → 512-d feature

每个分支接 projection：

Linear(512, 256)
BatchNorm1d(256)
ReLU
Dropout(0.3)

然后 concat：

Global + Eye:
[global_256, eye_256] → 512-d

Global + Cheek:
[global_256, cheek_256] → 512-d

Global + Eye + Cheek:
[global_256, eye_256, cheek_256] → 768-d

fusion classifier：

Linear(fusion_dim, 256)
BatchNorm1d(256)
ReLU
Dropout(0.3)
Linear(256, num_classes)

输出 logits，shape = [B, 3]。

feature extractor 实现建议：

对于 ResNet18：
- 使用 torchvision.models.resnet18(weights=...)
- 记录 model.fc.in_features = 512
- 将 model.fc = nn.Identity()

或者复用现有 resnet_nyha_3class / nyha_backbone_factory 中的 ResNet 构建逻辑，但需要去掉分类头，输出特征。

pretrained 规则：
- pretrained=True 或 "imagenet" 使用 torchvision DEFAULT weights。
- pretrained=False 使用 weights=None。

需要实现：

def count_parameters(model) -> dict:
    total_params
    trainable_params

forward 输入建议：

def forward(self, batch_or_inputs):
    ...

支持以下两种调用方式之一即可，但训练脚本必须匹配：

方式 A：
model(
    global_image=batch["global_image"],
    eye_image=batch.get("eye_image"),
    cheek_image=batch.get("cheek_image")
)

方式 B：
model(batch)

建议采用方式 A，清晰稳定。

模型必须检查：
- enabled_inputs 中必须包含 global。
- forward 时 enabled_inputs 对应图像不能为空。
- 不支持的 input name 抛出清晰错误。

========================================
六、Dataset 实现
========================================

请新增：

datasets/global_roi_fusion_dataset.py

实现：

class GlobalROIFusionDataset(Dataset):
    ...

每个样本返回：

{
    "global_image": Tensor[3, 224, 224],
    "eye_image": Tensor[3, 224, 224],       # 如果 enabled_inputs 包含 eye
    "cheek_image": Tensor[3, 224, 224],     # 如果 enabled_inputs 包含 cheek
    "label": int,
    "ID": str,
    "patient_group_id": str,
    "NYHA": int,
    "label_3class": int,
    "label_3class_name": str,
    "SEX": int,
    "sex_name": str,
    "fold": int,
    "global_image_path": str,
    "eye_image_path": str,
    "cheek_image_path": str,
}

如果某字段在 CSV 中不存在，可不返回，但必须至少返回：

global_image
ROI images according to enabled_inputs
label
ID
patient_group_id

读取方式：

PIL.Image.open(path).convert("RGB")

路径规则：

global image:
global_image_root / image_filename_template.format(ID=ID)

ROI image:
roi_roots[roi_name] / image_filename_template.format(ID=ID)

如果某张图像缺失：
训练前 preflight 直接报错；
Dataset 中也应抛出 FileNotFoundError，不能静默跳过。

----------------------------------------
同步随机水平翻转
----------------------------------------

这是本轮 Dataset 最重要的细节。

同一样本的 global、eye、cheek 必须使用相同的随机水平翻转决策。

不能让 global 翻转而 ROI 不翻转，也不能让不同 ROI 各自独立随机翻转。

建议实现方式：

1. 不直接用 torchvision.transforms.RandomHorizontalFlip。
2. 自己在 __getitem__ 中根据 random.random() < 0.5 决定 do_flip。
3. 如果 do_flip=True，对 global 和所有 enabled ROI 同时执行 ImageOps.mirror(img) 或 torchvision.transforms.functional.hflip(img)。
4. 然后统一 Resize、ToTensor、Normalize。

训练 transform：

Resize((224, 224))
synchronized horizontal flip p=0.5
ToTensor()
Normalize(ImageNet mean/std)

验证 transform：

Resize((224, 224))
ToTensor()
Normalize(ImageNet mean/std)

ImageNet mean/std：

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

第一版不要加入 ColorJitter、RandomCrop、RandomRotation。

========================================
七、训练脚本
========================================

请新增独立训练脚本：

scripts/train/train_global_roi_fusion_5fold.py

不要直接大改 train_nyha_3class_5fold.py。

训练脚本功能：

1. 解析参数：
   --config
   --output-dir，可选
   --fold，可选
   --epochs，可选
   --num-workers，可选

2. 读取 YAML 配置。

3. 设置随机种子。

4. 选择 cuda/cpu。

5. 保存 config.yaml 到实验目录。

6. 遍历 fold 0..4，或只训练指定 fold。

7. 每折读取：
   data.split_dir / fold_{fold}_train.csv
   data.split_dir / fold_{fold}_val.csv

8. 构建 GlobalROIFusionDataset。

9. 构建 DataLoader：
   train shuffle=True
   val shuffle=False

10. 构建 GlobalROIFusionModel。

11. 根据当前训练折标签计算 class weights。

12. 构建 Weighted Cross Entropy。

13. 构建 AdamW optimizer。

14. 训练并早停，monitor macro_auc。

15. 保存每折 best model。

16. 使用 best model 对 val fold 评估，保存：
   fold metrics
   oof predictions
   confusion matrix
   training curves

可以复用现有 NYHA3ClassTrainer / NYHA3ClassEvaluator，但需要确认它们是否支持 batch 中 image 字段名称不同。如果它们只支持 batch["image"]，可以：

方案 A：
新增 GlobalROIFusionTrainer / GlobalROIFusionEvaluator。

方案 B：
在训练循环中手动调用：
logits = model(
    global_image=batch["global_image"],
    eye_image=batch.get("eye_image"),
    cheek_image=batch.get("cheek_image"),
)

建议如果旧 Trainer 改动成本高，就新增独立 Trainer/Evaluator，避免破坏旧实验。

输出结构尽量保持与已有实验一致：

experiments/global_roi_fusion_500Data/<experiment_name>/
  config.yaml
  fold_0/
    checkpoints/
    metrics.csv
    predictions.csv
  fold_1/
  ...
  summary/
    fold_metrics_all.csv
    mean_metrics.csv
    oof_metrics.csv
    oof_predictions.csv
    summary_report.md

oof_predictions.csv 必须包含：

ID
patient_group_id
NYHA
label_3class
label_3class_name
SEX
sex_name
fold
y_true
y_pred
pred_class
prob_normal
prob_mild
prob_severe
correct

如果字段不存在，至少保存 ID、fold、y_true、pred_class、prob_*、correct。

========================================
八、评价指标
========================================

请与已有 NYHA 三分类实验保持一致，计算：

accuracy
balanced_accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
macro_auc
auc_normal
auc_mild
auc_severe
normal_vs_abnormal_auc
severe_vs_rest_auc
precision_normal
precision_mild
precision_severe
recall_normal
recall_mild
recall_severe
f1_normal
f1_mild
f1_severe

每折和 OOF 都要计算。

macro_auc 使用 one-vs-rest。

severe_vs_rest:
y_true = 1 if label_3class == 2 else 0
score = prob_severe

normal_vs_abnormal:
y_true = 1 if label_3class == 0 else 0
score = prob_normal

========================================
九、配置生成脚本
========================================

请新增：

scripts/run/generate_global_roi_fusion_configs.py

输出目录：

config/train/global_roi_fusion/

生成 3 个 YAML：

1. global_roi_fusion_global_eye_resnet18.yaml
2. global_roi_fusion_global_cheek_resnet18.yaml
3. global_roi_fusion_global_eye_cheek_resnet18.yaml

配置内容示例：

experiment:
  name: GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold
  output_dir: experiments/global_roi_fusion_500Data

data:
  split_dir: data/processed/splits_500
  global_image_root: data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images
  roi_roots:
    eye: <自动识别 eye ROI image root>
    cheek: <自动识别 cheek ROI image root>
  enabled_inputs:
    - global
    - eye
  image_filename_template: "{ID}.png"
  n_folds: 5
  image_size: 224
  num_classes: 3
  label_col: label_3class
  train_csv_pattern: fold_{fold}_train.csv
  val_csv_pattern: fold_{fold}_val.csv

model:
  backbone: resnet18
  pretrained: imagenet
  num_classes: 3
  freeze_backbone: false
  fusion_type: concat
  projection_dim: 256
  dropout: 0.3

train:
  batch_size: 8
  epochs: 50
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.0001
  loss: weighted_cross_entropy
  early_stopping_patience: 10
  monitor_metric: macro_auc
  random_seed: 2026
  num_workers: 0
  pin_memory: false
  use_amp: false

注意：
每个配置的 enabled_inputs 不同。

Global + Eye:
enabled_inputs = [global, eye]

Global + Cheek:
enabled_inputs = [global, cheek]

Global + Eye + Cheek:
enabled_inputs = [global, eye, cheek]

生成 manifest：

config/train/global_roi_fusion/global_roi_fusion_config_manifest.csv

字段：

job_id
experiment_key
config_path
experiment_name
output_root
enabled_inputs
global_image_root
eye_root
cheek_root
backbone
batch_size
status
error_message

三个 job：

global_eye
global_cheek
global_eye_cheek

========================================
十、预检脚本/预检逻辑
========================================

请在 generate config 或 run script 中实现 preflight。

至少检查：

1. split_dir 存在。
2. global_image_root 存在。
3. enabled ROI roots 存在。
4. fold_0_train.csv 到 fold_4_train.csv 存在。
5. fold_0_val.csv 到 fold_4_val.csv 存在。
6. CSV 包含 ID、patient_group_id、label_3class。
7. label_3class 只包含 0、1、2。
8. 若 CSV 包含 NYHA，则检查映射：
   NYHA 0 -> label_3class 0
   NYHA 1/2 -> label_3class 1
   NYHA 3/4 -> label_3class 2
9. train 和 val patient_group_id 无交集。
10. 每个验证 fold 的 fold 字段正确。
11. 所有 val ID 合并后无重复。
12. 每个样本的 global 图像存在。
13. enabled ROI 图像存在。
14. OOF 验证总数应为 500。

如果缺图数量较多，请输出前 30 个缺失项，并终止。

========================================
十一、批量运行脚本
========================================

请新增：

scripts/run/run_global_roi_fusion_5fold.py

功能：
读取：

config/train/global_roi_fusion/global_roi_fusion_config_manifest.csv

按顺序串行运行：

1. global_eye
2. global_cheek
3. global_eye_cheek

每个实验调用：

E:/resarch/Anaconda3/envs/face_heart/python.exe scripts/train/train_global_roi_fusion_5fold.py --config <config_path>

要求：

1. 使用 subprocess.run 阻塞执行。
2. 禁止并行。
3. 一个实验完整五折训练和 summary 完成后，才能进入下一个实验。
4. 每个实验保存 stdout/stderr。
5. 每个实验完成后检查 summary 文件是否存在。
6. 如果 summary 缺失，不标记 SUCCESS。

命令行参数：

--manifest
默认：
config/train/global_roi_fusion/global_roi_fusion_config_manifest.csv

--only
可选：
--only global_eye
--only global_cheek
--only global_eye_cheek

--dry-run
只打印命令，不训练。

--resume
跳过 SUCCESS，继续未完成。

--rerun-failed
只重跑 FAILED。

--skip-existing
如果目标实验已有完整 summary，则跳过。

--continue-on-error
失败后继续后续实验。

--allow-cpu
默认不允许 CPU 训练。如果 CUDA 不可用且没有 --allow-cpu，停止并报错。

输出：

experiments/global_roi_fusion_500Data/global_roi_fusion_job_queue.csv

字段：

job_id
experiment_key
config_path
experiment_name
status
start_time
end_time
duration_minutes
output_dir
exit_code
error_message

status：

PENDING
RUNNING
SUCCESS
FAILED
SKIPPED

日志目录：

experiments/global_roi_fusion_500Data/logs/

每个实验保存：

<experiment_key>_stdout.log
<experiment_key>_stderr.log

每个实验开始前打印并记录 GPU 信息：

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

如果 CUDA 不可用：
CUDA is not available.

若未设置 --allow-cpu，则停止训练。

========================================
十二、结果汇总脚本
========================================

请新增：

scripts/evaluate/summarize_global_roi_fusion_results.py

功能：
汇总并比较以下实验：

1. ResNet18 meanbg global baseline：
experiments/preprocess_ablation_500Data/PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg

2. roi_single / eye / resnet18：
请从 experiments/ROI_500 中自动找到对应实验目录。

3. roi_single / cheek / resnet34：
请从 experiments/ROI_500 中自动找到对应实验目录。

4. roi_fusion / multiroi5_fusion / resnet18：
请从 experiments/ROI_Fusion_500 中自动找到对应实验目录。

5. Global + Eye / ResNet18：
experiments/global_roi_fusion_500Data/GlobalROIFusion_GlobalEye_ResNet18_WeightedCE_5Fold

6. Global + Cheek / ResNet18：
experiments/global_roi_fusion_500Data/GlobalROIFusion_GlobalCheek_ResNet18_WeightedCE_5Fold

7. Global + Eye + Cheek / ResNet18：
experiments/global_roi_fusion_500Data/GlobalROIFusion_GlobalEyeCheek_ResNet18_WeightedCE_5Fold

如果目录因为时间戳变化，请自动扫描前缀匹配的最新目录。

输出：

experiments/global_roi_fusion_500Data/global_roi_fusion_summary.xlsx
experiments/global_roi_fusion_500Data/global_roi_fusion_summary.csv
experiments/global_roi_fusion_500Data/global_roi_fusion_summary.md

xlsx sheet 至少包含：

1. experiment_summary

字段：

experiment_key
experiment_name
source_type
output_dir
macro_auc_mean
macro_auc_std
balanced_accuracy_mean
balanced_accuracy_std
macro_f1_mean
macro_f1_std
macro_recall_mean
macro_recall_std
accuracy_mean
accuracy_std
recall_normal_mean
recall_mild_mean
recall_severe_mean
f1_normal_mean
f1_mild_mean
f1_severe_mean
severe_vs_rest_auc_mean
normal_vs_abnormal_auc_mean
oof_macro_auc
oof_balanced_accuracy
oof_macro_f1
oof_accuracy
oof_recall_normal
oof_recall_mild
oof_recall_severe
oof_f1_normal
oof_f1_mild
oof_f1_severe
oof_severe_vs_rest_auc
oof_normal_vs_abnormal_auc

2. fold_metrics_all

所有可读实验的每折长表。

3. oof_metrics_all

所有可读实验的 OOF 指标。

4. confusion_summary

所有实验 OOF 混淆矩阵关键误判：

normal_to_normal
normal_to_mild
normal_to_severe
mild_to_normal
mild_to_mild
mild_to_severe
severe_to_normal
severe_to_mild
severe_to_severe
mild_to_normal_rate
mild_to_severe_rate
severe_to_normal_rate
severe_to_mild_rate

5. delta_vs_global_resnet18_meanbg

新实验相对 ResNet18 meanbg 的差值：

delta_macro_auc_mean
delta_balanced_accuracy_mean
delta_macro_f1_mean
delta_recall_mild_mean
delta_f1_mild_mean
delta_recall_severe_mean
delta_f1_severe_mean
delta_oof_balanced_accuracy
delta_oof_macro_f1
delta_oof_recall_mild
delta_oof_recall_severe

6. recommendation

自动判断每个新实验：

candidate_main_model
improves_mild_only
improves_severe_only
fusion_not_helpful
overfit_or_unbalanced
needs_threshold_scan
next_step

推荐规则：

如果某 Global+ROI 模型满足：
macro_f1_mean > ResNet18 meanbg macro_f1_mean
balanced_accuracy_mean >= ResNet18 meanbg balanced_accuracy_mean
recall_mild_mean >= ResNet18 meanbg recall_mild_mean
recall_severe_mean >= ResNet18 meanbg recall_severe_mean
则标记 candidate_main_model。

如果 macro-F1 接近 ResNet18 meanbg，BA 高于 ResNet18，且 mild/severe 之一改善、另一个下降不超过 0.03，则标记 candidate_secondary_model。

如果低于 global-only 和最佳单 ROI，则标记 fusion_not_helpful。

summary.md 必须包含：

1. 实验目的；
2. 为什么选择 eye 和 cheek；
3. Global + Eye 是否改善 mild；
4. Global + Cheek 是否改善 severe / BA；
5. Global + Eye + Cheek 是否优于 global-only；
6. 是否优于最佳单 ROI；
7. 是否优于旧 ROI-only fusion；
8. 主要误判方向；
9. 是否建议继续加入 lip/forehead；
10. 是否建议做 ResNet34 版本；
11. 是否建议转向 ordinal/two-stage。

========================================
十三、诊断脚本
========================================

请新增：

scripts/evaluate/diagnose_global_roi_fusion_results.py

功能：
读取 global_roi_fusion_summary.xlsx/csv 以及各实验 oof_predictions.csv，生成更详细诊断。

输出目录：

experiments/global_roi_fusion_500Data/diagnostic_analysis/

输出：

global_roi_fusion_diagnostic_report.md
global_roi_fusion_diagnostic_tables.xlsx
global_roi_fusion_core_comparison.csv
global_roi_fusion_confusion_summary.csv

报告必须包含：

# Global ROI Fusion Diagnostic Report

## 1. Files Checked
列出读取到的文件和缺失文件。

## 2. Experiment Status
三个 Global+ROI 实验是否 SUCCESS。

## 3. Core Result Table
总表。

## 4. Comparison with Global ResNet18 MeanBG
是否超过全局基线。

## 5. Comparison with ROI Single and ROI-only Fusion
是否超过最佳单 ROI 和旧 ROI-only fusion。

## 6. Per-class Analysis
重点分析 mild 和 severe。

## 7. Confusion Matrix Analysis
重点分析：
- mild -> normal
- mild -> severe
- severe -> normal
- severe -> mild

## 8. Recommendation for Next Step
给出下一步建议。

## Key Information for Further Analysis

必须列出：

1. ResNet18 meanbg macro-F1：
2. ResNet18 meanbg BA：
3. ResNet18 meanbg mild recall：
4. ResNet18 meanbg severe recall：
5. Eye single ResNet18 macro-F1：
6. Eye single ResNet18 mild recall：
7. Cheek single ResNet34 macro-F1：
8. Cheek single ResNet34 severe recall：
9. Old ROI-only fusion ResNet18 macro-F1：
10. Global+Eye macro-F1：
11. Global+Eye BA：
12. Global+Eye mild recall：
13. Global+Eye severe recall：
14. Global+Cheek macro-F1：
15. Global+Cheek BA：
16. Global+Cheek mild recall：
17. Global+Cheek severe recall：
18. Global+Eye+Cheek macro-F1：
19. Global+Eye+Cheek BA：
20. Global+Eye+Cheek mild recall：
21. Global+Eye+Cheek severe recall：
22. 哪个 Global+ROI 模型最好：
23. 是否超过 ResNet18 meanbg：
24. 是否超过最佳单 ROI：
25. 是否超过旧 ROI-only fusion：
26. 是否建议加入 lip/forehead：
27. 是否建议做 ResNet34 版本：
28. 下一步最推荐实验：

========================================
十四、总控脚本
========================================

请新增：

scripts/run/run_global_roi_fusion_pipeline.py

按顺序执行：

1. generate_global_roi_fusion_configs.py
2. run_global_roi_fusion_5fold.py
3. summarize_global_roi_fusion_results.py
4. diagnose_global_roi_fusion_results.py

命令行参数：

--skip-config-generation
--skip-training
--skip-summary
--skip-diagnostic
--dry-run
--only
--resume
--rerun-failed
--skip-existing
--continue-on-error
--allow-cpu

输出：

experiments/global_roi_fusion_500Data/global_roi_fusion_pipeline_log.txt
experiments/global_roi_fusion_500Data/global_roi_fusion_pipeline_status.json

每个 stage 记录：

stage_name
command
start_time
end_time
duration_minutes
status
return_code
error_message

========================================
十五、测试命令
========================================

实现完成后，请给出并尽量执行以下命令。

进入项目根目录：

cd /d E:\projects\face2

语法检查：

E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile models\global_roi_fusion_model.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile datasets\global_roi_fusion_dataset.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\train\train_global_roi_fusion_5fold.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\generate_global_roi_fusion_configs.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\run_global_roi_fusion_5fold.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\evaluate\summarize_global_roi_fusion_results.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\evaluate\diagnose_global_roi_fusion_results.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\run_global_roi_fusion_pipeline.py

生成配置：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\generate_global_roi_fusion_configs.py

dry-run：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_5fold.py --dry-run

只跑 Global+Eye：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_5fold.py --only global_eye --continue-on-error

只跑 Global+Cheek：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_5fold.py --only global_cheek --continue-on-error

只跑 Global+Eye+Cheek：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_5fold.py --only global_eye_cheek --continue-on-error

正式跑全部三个实验：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_5fold.py --continue-on-error

汇总：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\evaluate\summarize_global_roi_fusion_results.py

诊断：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\evaluate\diagnose_global_roi_fusion_results.py

一键 dry-run：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_pipeline.py --dry-run

一键正式运行全部：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_global_roi_fusion_pipeline.py --continue-on-error

========================================
十六、最终交付说明
========================================

实现完成后，请汇报：

1. 新增了哪些文件；
2. 是否修改了旧文件；
3. eye ROI root 和 cheek ROI root 是如何自动识别的；
4. 是否确认 global、eye、cheek 的图像都完整；
5. 是否确认 OOF 样本数为 500；
6. 是否通过 py_compile；
7. 三个配置文件路径；
8. 三个实验输出路径；
9. job_queue 路径；
10. summary 路径；
11. diagnostic report 路径；
12. Global+Eye 初步结果；
13. Global+Cheek 初步结果；
14. Global+Eye+Cheek 初步结果；
15. 是否超过 ResNet18 meanbg；
16. 是否超过最佳单 ROI；
17. 是否超过旧 ROI-only fusion；
18. 下一步建议。

注意：
本轮实验是 Global + selected ROI feature-level fusion，用于验证全局面部和关键局部区域的互补性。
不要把该实验写成最终临床模型。
所有结论仍然基于 5-fold CV 和 OOF，需要后续独立验证。