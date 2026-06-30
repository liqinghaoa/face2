任务：实现配置驱动型 Multi-ROI 特征级融合 NYHA 三分类实验
请基于项目目录：
E:\projects\face2
实现一个可配置的 Multi-ROI 特征级融合实验框架，用于 NYHA 三分类任务。
当前项目已经有全局人脸图三分类、ROI-only 三分类、ResNet18/34/50、WeightedCE、五折训练和汇总等基础框架。请不要重新实现整套训练系统，而是在现有框架基础上新增 Multi-ROI Dataset、Multi-ROI Fusion Model 和通用 ROI fusion run 入口。
一、实验目标
当前任务是 NYHA 三分类：
0 = normal
1 = mild
2 = severe
标签映射为：
NYHA 0     → normal
NYHA 1、2  → mild
NYHA 3、4  → severe
本次实验目标是实现 feature-level ROI fusion：
多个 ROI 图像
    → 共享 ResNet backbone 提取特征
    → 拼接多个 ROI 特征
    → MLP 分类头
    → 输出 normal / mild / severe 三分类 logits
本次不是做单个固定 ROI 组合，而是实现配置驱动框架，使后续可以通过 YAML 配置自由设置：
1. ROI 组合：eye / lip / cheek / forehead / chin 的任意组合
2. Backbone：resnet18 / resnet34 / resnet50
3. 融合方式：第一版只支持 concat
二、已有框架约束
请先阅读并理解以下现有代码，再做修改：
E:\projects\face2\scripts\train\train_nyha_3class_5fold.py
E:\projects\face2\scripts\evaluate\summarize_nyha_3class_5fold.py
E:\projects\face2\datasets\nyha_3class_face_dataset.py
E:\projects\face2\models\resnet_nyha_3class.py
E:\projects\face2\losses\classification_losses.py
E:\projects\face2\trainers\nyha_3class_trainer.py
E:\projects\face2\evaluators\nyha_3class_evaluator.py
当前 ROI-only 训练框架的基本逻辑是：
run_exp_roi_nyha3class_5fold.py
  -> preflight 检查 ROI 图像与 splits_500
  -> scripts/train/train_nyha_3class_5fold.py
  -> datasets/nyha_3class_face_dataset.py
  -> models/resnet_nyha_3class.py
  -> losses/classification_losses.py
  -> trainers/nyha_3class_trainer.py
  -> evaluators/nyha_3class_evaluator.py
  -> scripts/evaluate/summarize_nyha_3class_5fold.py
Multi-ROI 融合实验应尽量复用：
loss
optimizer
trainer
evaluator
metrics
summarizer
只新增或小幅扩展：
Dataset
Model
train script 分支
run script / preflight
config
三、数据输入约束
ROI 图像根目录固定为：
E:\projects\face2\data\processed\roi_dataset\manual_shift_data
已有 ROI 子目录包括：
cheek_roi
chin_roi
eye_roi
forehead_roi
lip_roi
每个 ROI 子目录下图像文件命名为：
{ID}.png
固定五折划分目录为：
E:\projects\face2\data\processed\splits_500
每折读取：
fold_0_train.csv
fold_0_val.csv
...
fold_4_train.csv
fold_4_val.csv
不要重新划分数据。
不要从 ROI 图像目录自动生成标签。
标签、fold、patient_group_id、NYHA、SEX 等信息全部来自 splits_500 中的 CSV。
四、新增 Multi-ROI Dataset
请新增文件：
E:\projects\face2\datasets\nyha_3class_multi_roi_dataset.py
新增核心类：
class NYHA3ClassMultiROIDataset(Dataset):
    ...
4.1 初始化参数建议
class NYHA3ClassMultiROIDataset(Dataset):
    def __init__(
        self,
        csv_path,
        roi_root,
        roi_names,
        image_filename_template="{ID}.png",
        image_size=224,
        label_col="label_3class",
        train=False,
        horizontal_flip=False,
        same_flip_for_all_rois=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        ...
4.2 Dataset 读取逻辑
对于每一行 CSV，读取 ID，然后按配置中的 roi_names 依次读取：
{roi_root}/{roi_name}/{ID}.png
例如配置：
roi_names:
  - eye_roi
  - lip_roi
  - cheek_roi
  - forehead_roi
则某个样本 ID 为 9300017987 时，读取：
manual_shift_data/eye_roi/9300017987.png
manual_shift_data/lip_roi/9300017987.png
manual_shift_data/cheek_roi/9300017987.png
manual_shift_data/forehead_roi/9300017987.png
4.3 返回格式
__getitem__ 返回：
images, label, meta
其中：
images: Tensor [R, 3, 224, 224]
label: int
meta: dict
R = len(roi_names)。
DataLoader 组 batch 后应为：
images: [B, R, 3, 224, 224]
labels: [B]
meta 至少包含：
ID
patient_group_id
NYHA
label_3class
SEX
fold
如果当前 trainer/evaluator 对 meta 格式有固定要求，请与现有 NYHA3ClassFaceDataset 保持兼容。
4.4 同步水平翻转
训练集仍然只使用水平翻转，不使用 color jitter，不使用 random crop。
但 Multi-ROI Dataset 中必须保证：同一病例的所有 ROI 使用相同的水平翻转决策。
不要对每张 ROI 图像单独使用 RandomHorizontalFlip，否则会出现一个样本内不同 ROI 翻转状态不一致的问题。
推荐在 __getitem__ 中统一生成：
do_flip = self.train and self.horizontal_flip and random.random() < 0.5
然后对该样本所有 ROI：
if do_flip:
    img = TF.hflip(img)
训练集处理：
Resize(224,224)
same RandomHorizontalFlip(p=0.5) for all ROIs
ToTensor
ImageNet Normalize
验证集处理：
Resize(224,224)
ToTensor
ImageNet Normalize
五、新增 Multi-ROI Fusion Model
请新增文件：
E:\projects\face2\models\multi_roi_fusion_nyha_3class.py
新增核心类：
class ConfigurableMultiROIFusionResNet(nn.Module):
    ...
5.1 初始化参数建议
class ConfigurableMultiROIFusionResNet(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained="imagenet",
        num_rois=4,
        num_classes=3,
        shared_backbone=True,
        fusion_method="concat",
        hidden_dim=512,
        dropout=0.3,
        use_batchnorm=True,
        freeze_backbone=False,
    ):
        ...
5.2 第一版支持范围
第一版必须支持：
backbone: resnet18 / resnet34 / resnet50
pretrained: imagenet
shared_backbone: true
fusion_method: concat
num_classes: 3
第一版可以暂不实现：
shared_backbone: false
attention fusion
global + ROI fusion
sex feature fusion
label smoothing
如果配置中出现不支持的选项，请明确报错，不要静默忽略。
5.3 Backbone 构建要求
请支持 torchvision ImageNet 预训练权重：
resnet18 → ResNet18_Weights.IMAGENET1K_V1
resnet34 → ResNet34_Weights.IMAGENET1K_V1
resnet50 → ResNet50_Weights.IMAGENET1K_V1
如果项目已有兼容新旧 torchvision 的写法，请沿用现有写法。
构建 backbone 时，不要保留原始分类头。
应使用：
feature_dim = model.fc.in_features
model.fc = nn.Identity()
注意不要写死 512，因为：
resnet18 / resnet34: feature_dim = 512
resnet50: feature_dim = 2048
5.4 Forward 逻辑
输入：
x: [B, R, 3, 224, 224]
其中：
B = batch size
R = ROI 数量
forward 逻辑：
B, R, C, H, W = x.shape

if R != self.num_rois:
    raise ValueError(...)

x = x.reshape(B * R, C, H, W)
features = self.backbone(x)          # [B*R, D]
features = features.reshape(B, R, -1) # [B, R, D]

fused = features.reshape(B, R * D)    # [B, R*D]
logits = self.classifier(fused)       # [B, 3]

return logits
输出必须是 logits，不加 softmax。
5.5 Fusion Classifier
第一版使用 concat fusion：
fusion_dim = num_rois * feature_dim
分类头建议：
layers = [
    nn.Linear(fusion_dim, hidden_dim),
]

if use_batchnorm:
    layers.append(nn.BatchNorm1d(hidden_dim))

layers.extend([
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes),
])
对于 ResNet50 + 多 ROI，fusion_dim 可能很大，例如 4 ROI 时为 8192。因此必须通过 hidden_dim 降维，不要直接接复杂大头。
5.6 冻结策略
第一版：
freeze_backbone: false
所有参数参与训练。
如果配置中 freeze_backbone=true，可以支持，也可以先报错。为了和现有实验保持一致，建议第一版强制要求 false。
六、训练脚本适配
请小幅修改：
E:\projects\face2\scripts\train\train_nyha_3class_5fold.py
使其支持：
model:
  type: multi_roi_fusion
6.1 Dataset 分支
如果：
cfg["model"]["type"] == "multi_roi_fusion"
则使用：
NYHA3ClassMultiROIDataset
否则继续使用现有：
NYHA3ClassFaceDataset
MultiROI train dataset 构建时从 config 读取：
data.roi_root
data.roi_names
data.image_filename_template
data.image_size
data.label_col
augmentation.horizontal_flip
augmentation.same_flip_for_all_rois
normalize.mean
normalize.std
6.2 Model 分支
如果：
cfg["model"]["type"] == "multi_roi_fusion"
则构建：
ConfigurableMultiROIFusionResNet(
    backbone=cfg["model"]["backbone"],
    pretrained=cfg["model"]["pretrained"],
    num_rois=len(cfg["data"]["roi_names"]),
    num_classes=cfg["model"]["num_classes"],
    shared_backbone=cfg["model"]["shared_backbone"],
    fusion_method=cfg["model"]["fusion_method"],
    hidden_dim=cfg["model"]["fusion_head"]["hidden_dim"],
    dropout=cfg["model"]["fusion_head"]["dropout"],
    use_batchnorm=cfg["model"]["fusion_head"]["use_batchnorm"],
    freeze_backbone=cfg["model"]["freeze_backbone"],
)
否则继续走现有单图 ResNet 构建逻辑。
6.3 Loss 不改
本次实验继续使用：
weighted_cross_entropy
类别权重仍然每个 fold 根据训练集动态计算：
class_weight = N / (C × class_count)
不要改成 label smoothing。
不要使用 focal loss。
不要修改当前 loss factory 中已有逻辑，除非为了兼容 config 的 loss 字段。
6.4 Trainer / Evaluator 尽量不改
如果现有 trainer/evaluator 只是将 images 送入 model(images)，则无需修改。
如果代码中强制要求：
images.ndim == 4
请改成允许：
images.ndim in [4, 5]
因为 Multi-ROI 输入是：
[B, R, 3, 224, 224]
Evaluator 最终仍然只需要模型输出：
logits: [B, 3]
所以 metrics、AUC、confusion matrix、OOF 逻辑都不需要改变。
七、新增通用 ROI Fusion Run 脚本
请新增：
E:\projects\face2\scripts\run\run_exp_roi_fusion_nyha3class_5fold.py
第一版只要求支持：
python scripts/run/run_exp_roi_fusion_nyha3class_5fold.py --config config/train/roi_fusion/nyha_3class_multiroi_shared_resnet34_concat_weightedce.yaml
不强制第一版实现命令行覆盖 --rois 和 --backbone，但可以预留。
7.1 Run 脚本职责
该 run 脚本负责：
1. 读取 config
2. 执行 ROI fusion preflight 检查
3. 创建实验输出目录
4. 避免覆盖旧实验目录
5. 调用 scripts/train/train_nyha_3class_5fold.py
6. 调用 scripts/evaluate/summarize_nyha_3class_5fold.py
7. 写入 run_5fold.log
调用训练脚本形式：
python scripts/train/train_nyha_3class_5fold.py --config <config_path> --output-dir <experiment_dir>
调用汇总脚本形式：
python scripts/evaluate/summarize_nyha_3class_5fold.py --experiment-dir <experiment_dir>
输出目录：
E:\projects\face2\experiments\ROI_Fusion_500\{experiment_name}
如果目录已存在，自动追加时间戳，或清楚提示，不要直接覆盖。
八、Preflight 检查
ROI fusion 的 preflight 必须严格。
请在 run 脚本中实现 preflight(config) 或放在独立 utils 中。
8.1 配置检查
检查：
config 文件存在
experiment.name 非空
data.split_dir == E:\projects\face2\data\processed\splits_500
data.roi_root == E:\projects\face2\data\processed\roi_dataset\manual_shift_data
data.roi_names 存在且长度 >= 2
data.roi_names 无重复
data.image_filename_template == "{ID}.png"
data.n_folds == 5
data.image_size == 224
data.num_classes == 3
model.type == multi_roi_fusion
model.backbone in [resnet18, resnet34, resnet50]
model.pretrained == imagenet
model.shared_backbone == true
model.fusion_method == concat
model.num_classes == 3
model.freeze_backbone == false
loss.name == weighted_cross_entropy
loss.class_weight == true
augmentation.horizontal_flip == true
augmentation.same_flip_for_all_rois == true
augmentation.color_jitter == false
augmentation.random_crop == false
8.2 ROI 目录检查
允许 ROI 名称：
cheek_roi
chin_roi
eye_roi
forehead_roi
lip_roi
检查每个 roi_name：
roi_root / roi_name 目录存在
8.3 Split CSV 检查
检查每折 CSV：
fold_{fold}_train.csv
fold_{fold}_val.csv
必须存在。
必须包含列：
ID
patient_group_id
NYHA
label_3class
SEX
fold
如果存在以下列，也检查一致性：
label_3class_name
sex_name
检查：
label_3class 只能是 0/1/2
NYHA 到 label_3class 映射必须一致：
  NYHA 0 -> 0
  NYHA 1/2 -> 1
  NYHA 3/4 -> 2
val CSV 中 fold 必须等于当前 fold
train CSV 中不能包含当前 fold
train/val 之间不能有 patient_group_id 泄漏
同一 ID 不应在同一 fold 的 train 和 val 同时出现
8.4 多 ROI 图像完整性检查
对所有 split 中出现的 ID，检查每个 ROI 都有对应图像：
{roi_root}/{roi_name}/{ID}.png
检查所有 ROI 子目录中的 split ID 是否一致。
建议检查：
每个 ROI 子目录中的 PNG 数量
每个 ROI 子目录是否包含所有 split ID
是否存在 split 外额外 PNG
如果存在额外 PNG，第一版可以沿用 ROI-only 的严格逻辑，直接报错；或者至少清楚警告。为了与当前 ROI-only 框架一致，建议直接报错。
九、新增配置文件
请新增目录：
E:\projects\face2\config\train\roi_fusion
新增第一版配置文件：
E:\projects\face2\config\train\roi_fusion\nyha_3class_multiroi_shared_resnet34_concat_weightedce.yaml
配置内容建议如下：
experiment:
  name: MultiROI4_ImageNetResNet34_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold
  output_dir: E:\projects\face2\experiments\ROI_Fusion_500

data:
  split_dir: E:\projects\face2\data\processed\splits_500
  roi_root: E:\projects\face2\data\processed\roi_dataset\manual_shift_data
  roi_names:
    - eye_roi
    - lip_roi
    - cheek_roi
    - forehead_roi
  image_filename_template: "{ID}.png"
  n_folds: 5
  image_size: 224
  num_classes: 3
  label_col: label_3class

model:
  type: multi_roi_fusion
  backbone: resnet34
  pretrained: imagenet
  shared_backbone: true
  fusion_method: concat
  num_classes: 3
  freeze_backbone: false
  fusion_head:
    hidden_dim: 512
    dropout: 0.3
    use_batchnorm: true

loss:
  name: weighted_cross_entropy
  class_weight: true

train:
  batch_size: 16
  epochs: 50
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  monitor_metric: macro_auc
  random_seed: 2026
  num_workers: 0
  pin_memory: false
  use_amp: false

augmentation:
  horizontal_flip: true
  same_flip_for_all_rois: true
  color_jitter: false
  random_crop: false

normalize:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

metrics:
  main:
    - macro_auc
    - accuracy
    - macro_precision
    - macro_recall
    - macro_f1
    - balanced_accuracy
  auxiliary:
    - per_class_auc
    - per_class_precision
    - per_class_recall
    - per_class_f1
    - severe_vs_rest_auc
    - normal_vs_abnormal_auc
    - confusion_matrix
十、后续配置示例
实现时要确保后续可以通过复制 YAML 并修改配置完成不同实验。
10.1 改 ROI 组合
例如只融合 eye + lip：
experiment:
  name: MultiROI2_EyeLip_ImageNetResNet34_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold

data:
  roi_names:
    - eye_roi
    - lip_roi
模型应自动识别 num_rois=2，不需要修改代码。
10.2 改 backbone
例如 ResNet18：
experiment:
  name: MultiROI4_ImageNetResNet18_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold

model:
  backbone: resnet18
例如 ResNet50：
experiment:
  name: MultiROI4_ImageNetResNet50_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold

model:
  backbone: resnet50
模型应自动识别：
resnet18/resnet34 feature_dim = 512
resnet50 feature_dim = 2048
不要写死融合维度。
十一、summary_report.md 需要记录的信息
请确保最终 summary report 中除了原有指标，还记录：
Experiment name
Dataset: ROI_Fusion_500
Split dir: splits_500
ROI root
ROI names
Number of ROIs
Model type: multi_roi_fusion
Backbone
Pretrained weights
Shared backbone
Fusion method
Feature dim per ROI
Fusion dim
Fusion head hidden dim
Fusion head dropout
Loss
Class weight rule
Augmentation
same_flip_for_all_rois
例如：
ROI names: eye_roi, lip_roi, cheek_roi, forehead_roi
Number of ROIs: 4
Backbone: resnet34
Feature dim per ROI: 512
Fusion dim: 2048
Fusion method: concat
Shared backbone: true
Loss: weighted_cross_entropy
十二、输出结构
输出目录：
E:\projects\face2\experiments\ROI_Fusion_500\MultiROI4_ImageNetResNet34_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold
目录结构沿用当前五折实验：
MultiROI4_ImageNetResNet34_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold
├── config.yaml
├── run_5fold.log
├── fold_0
│   ├── checkpoints
│   ├── logs
│   ├── predictions
│   ├── metrics
│   └── curves
├── fold_1
├── fold_2
├── fold_3
├── fold_4
└── summary
    ├── fold_metrics_all.csv
    ├── mean_metrics.csv
    ├── oof_predictions.csv
    ├── oof_metrics.csv
    └── summary_report.md
十三、Smoke Test
实现后，请做以下检查。
13.1 Dataset 检查
临时读取一个样本：
dataset = NYHA3ClassMultiROIDataset(...)
images, label, meta = dataset[0]
print(images.shape)
print(label)
print(meta)
期望：
torch.Size([4, 3, 224, 224])
label 为 0/1/2
meta 中包含 ID、NYHA、fold 等信息
13.2 Model forward 检查
model = ConfigurableMultiROIFusionResNet(
    backbone="resnet34",
    pretrained="imagenet",
    num_rois=4,
    num_classes=3,
    shared_backbone=True,
    fusion_method="concat",
)

x = torch.randn(2, 4, 3, 224, 224)
y = model(x)
print(y.shape)
期望：
torch.Size([2, 3])
13.3 ResNet50 检查
model = ConfigurableMultiROIFusionResNet(
    backbone="resnet50",
    pretrained="imagenet",
    num_rois=4,
    num_classes=3,
    shared_backbone=True,
    fusion_method="concat",
)
x = torch.randn(2, 4, 3, 224, 224)
y = model(x)
print(y.shape)
期望：
torch.Size([2, 3])
确认不会因为 ResNet50 feature_dim=2048 而维度错误。
13.4 运行入口检查
运行：
python E:\projects\face2\scripts\run\run_exp_roi_fusion_nyha3class_5fold.py --config E:\projects\face2\config\train\roi_fusion\nyha_3class_multiroi_shared_resnet34_concat_weightedce.yaml
先确认 preflight 能通过并开始训练。
如果支持单 fold 调试，可以先只跑 fold_0；如果现有框架不支持，则直接完整五折。
十四、不要做的事情
本次不要做以下修改：
不要重新划分数据
不要切换到 522Data
不要使用 470Data / 500Data 的非 splits_500 之外划分
不要修改 label 映射
不要修改 sex 映射
不要修改评价指标
不要修改 WeightedCE 逻辑
不要加入 label smoothing
不要加入 focal loss
不要加入 ColorJitter
不要加入 random crop
不要加入 sex 特征
不要加入 global image
不要实现 attention fusion
不要默认使用 independent backbone
不要写死 ROI 数量为 4
不要写死 ROI 名称
不要写死 feature_dim=512
不要把 ResNet50 融合维度写错
不要让同一病例的不同 ROI 随机翻转状态不一致
不要覆盖已有 ROI-only 或 global-only 实验输出目录
十五、最终目标
实现完成后，我可以通过配置文件控制 ROI 组合和 backbone。
第一版运行：
python E:\projects\face2\scripts\run\run_exp_roi_fusion_nyha3class_5fold.py --config E:\projects\face2\config\train\roi_fusion\nyha_3class_multiroi_shared_resnet34_concat_weightedce.yaml
完成实验：
MultiROI4_ImageNetResNet34_SharedBackbone_ConcatFusion_NYHA3Class_WeightedCE_5Fold
该实验应实现：
eye_roi + lip_roi + cheek_roi + forehead_roi
shared ResNet34
concat feature fusion
WeightedCE
splits_500
5-fold training + OOF summary
后续我可以只改 YAML：
roi_names
backbone
experiment.name
来运行不同 ROI 组合和不同 ResNet backbone 的融合实验。

代码实现后，不要直接运行实验，指导我人工运行实验。