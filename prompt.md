你现在在 Windows + PyCharm 项目中工作，项目根目录为：

E:/projects/face2

本次任务目标：
在当前最佳预处理方案 hybrid_imagenet_meanbg 的基础上，开展多 backbone 模型探索实验，评估 ResNet 之外的深度学习分类模型是否能提升 NYHA 三分类性能。

本轮实验定位：
模型架构消融实验。

核心问题：
在固定同一套数据、同一套五折划分、同一套预处理、同一套 loss、optimizer、epoch 和 augmentation 的前提下，DenseNet、EfficientNet、ConvNeXt、Swin、MobileNetV3 等非 ResNet backbone 是否能优于当前 ResNet18 + hybrid_imagenet_meanbg 基线。

请严格遵守以下原则：

1. 本轮只改变 model.backbone。
2. 不改变数据预处理。
3. 不改变五折划分。
4. 不改变 loss。
5. 不改变 optimizer。
6. 不改变 learning rate。
7. 不改变 weight decay。
8. 不改变 epoch。
9. 不改变 early stopping。
10. 不改变数据增强。
11. 不加入 label smoothing。
12. 不加入 focal loss。
13. 不加入 ColorJitter。
14. 不加入 ROI fusion。
15. 不加入阈值优化作为训练结果。
16. 不安装新依赖。
17. 不使用 timm。
18. 只使用 torchvision.models。
19. 所有训练必须严格串行执行，不允许并行训练。
20. 必须保持旧 ResNet 配置完全兼容。

========================================
一、当前项目与环境信息
========================================

项目根目录：

E:/projects/face2

Python 环境：

E:/resarch/Anaconda3/envs/face_heart/python.exe

当前环境已确认：

torch = 2.3.0
torchvision = 0.18.0
cuda = True

当前 torchvision.models 已确认支持以下模型：

densenet121
efficientnet_b0
convnext_tiny
swin_t
mobilenet_v3_large

因此本轮不需要安装 timm，也不要引入其他新依赖。

当前通用 YAML 配置训练入口已存在，并已跑通过：

scripts/run/run_nyha3class_5fold_with_config.py

该脚本已经在 Backbone Check 中成功运行多个 YAML 配置，因此本轮继续复用它，不要重新实现五折训练入口。

当前训练脚本：

scripts/train/train_nyha_3class_5fold.py

当前单图像分支中，_build_model(config) 仍直接调用：

models/resnet_nyha_3class.py::build_resnet_nyha_model

当前逻辑类似：

return build_resnet_nyha_model(
    backbone=model_config["backbone"],
    num_classes=int(model_config["num_classes"]),
    pretrained=_pretrained_enabled(model_config["pretrained"]),
)

本轮需要小幅修改这个模型构建入口，使其支持非 ResNet backbone。

允许修改：

scripts/train/train_nyha_3class_5fold.py

但修改范围必须限制在模型构建入口，不允许改动训练流程、loss、optimizer、scheduler、early stopping、metrics、dataset、split 或 augmentation。

========================================
二、本轮固定数据与基线
========================================

固定五折划分目录：

data/processed/splits_500

固定图像输入目录：

data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images

固定预处理方案：

hybrid_imagenet_meanbg

固定类别定义：

0 = normal
1 = mild
2 = severe

固定标签列：

label_3class

固定图像尺寸：

224 x 224

固定训练基础设置：

loss = weighted_cross_entropy
optimizer = AdamW
lr = 0.0001
weight_decay = 0.0001
epochs = 50
early_stopping_patience = 10
monitor_metric = macro_auc
random_seed = 2026
num_workers = 0
pin_memory = false
use_amp = false

固定 transform：
沿用当前 Dataset 和训练脚本逻辑，不新增 ColorJitter、RandomCrop、RandomRotation 或其他增强。

参考基线：

ResNet18 + hybrid_imagenet_meanbg

基线目录：

E:/projects/face2/experiments/preprocess_ablation_500Data/PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg

该目录已确认包含：

summary/fold_metrics_all.csv
summary/mean_metrics.csv
summary/oof_metrics.csv
summary/oof_predictions.csv
summary/summary_report.md

基线关键指标：

macro-AUC = 0.7094
balanced accuracy = 0.5353
macro-F1 = 0.5111
severe recall = 0.4333

后续 model exploration 汇总时，需要与这个 ResNet18 meanbg 基线做 delta 对比。

========================================
三、本轮候选模型
========================================

第一轮支持 5 个候选模型：

1. densenet121
2. efficientnet_b0
3. convnext_tiny
4. swin_t
5. mobilenet_v3_large

正式训练时需要支持 --only 参数，因此可以先只跑前三个模型：

densenet121
efficientnet_b0
convnext_tiny

但代码和配置生成阶段需要完整支持 5 个模型。

建议 batch size：

densenet121: 16
efficientnet_b0: 16
convnext_tiny: 8
swin_t: 8
mobilenet_v3_large: 16

理由：
当前设备是笔记本 RTX 4060。convnext_tiny 和 swin_t 显存占用相对更高，第一轮使用 batch size 8 更稳。

========================================
四、任务 1：新增通用 backbone factory
========================================

请新增文件：

models/nyha_backbone_factory.py

目标：
构建统一的 NYHA 三分类模型工厂，支持 ResNet 和非 ResNet backbone。

需要提供主函数：

def build_nyha_classification_model(
    backbone: str,
    num_classes: int = 3,
    pretrained: bool | str = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
) -> torch.nn.Module:
    ...

要求支持以下 backbone：

resnet18
resnet34
resnet50
densenet121
efficientnet_b0
convnext_tiny
swin_t
mobilenet_v3_large

实现要求：

1. 优先使用 torchvision.models。
2. 不使用 timm。
3. 不安装新包。
4. pretrained=True 或 pretrained="imagenet" 时，使用 torchvision 对应 DEFAULT weights。
5. pretrained=False 或 pretrained=None 时，weights=None。
6. 对不支持的 backbone 抛出清晰错误。
7. 保持 ResNet18/34/50 与旧逻辑兼容。
8. 每个模型都替换最终分类头为 num_classes=3。
9. 支持 freeze_backbone。
10. 支持 dropout 参数，但默认不启用。
11. 返回标准 torch.nn.Module。
12. 提供参数统计函数。

建议同时实现：

def count_parameters(model: torch.nn.Module) -> dict:
    返回：
    total_params
    trainable_params

def get_supported_backbones() -> list[str]:
    返回当前支持的 backbone 名称。

各模型分类头替换建议：

1. ResNet18/34/50：
可以继续调用旧的 build_resnet_nyha_model，或在 factory 内直接使用 torchvision.models.resnet18/resnet34/resnet50。
为了最大兼容旧实验，建议 factory 对 ResNet 系列优先复用：
models/resnet_nyha_3class.py::build_resnet_nyha_model

2. DenseNet121：
model = torchvision.models.densenet121(weights=...)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)

3. EfficientNet-B0：
model = torchvision.models.efficientnet_b0(weights=...)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

4. ConvNeXt-Tiny：
model = torchvision.models.convnext_tiny(weights=...)
torchvision 中 classifier 结构可能因版本略有差异。
请不要硬编码过死。
建议在 model.classifier 中反向查找最后一个 nn.Linear 并替换。
如果找不到 nn.Linear，抛出清晰错误。

5. Swin-Tiny：
model = torchvision.models.swin_t(weights=...)
in_features = model.head.in_features
model.head = nn.Linear(in_features, num_classes)

6. MobileNetV3-Large：
model = torchvision.models.mobilenet_v3_large(weights=...)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

freeze_backbone 逻辑：
如果 freeze_backbone=True：
先将所有参数 requires_grad=False；
然后将最终分类头参数 requires_grad=True。
需要根据不同模型找到分类头：
resnet: fc
densenet: classifier
efficientnet: classifier
convnext: classifier
swin: head
mobilenet_v3_large: classifier

本轮配置中 freeze_backbone=false，但 factory 仍应支持该参数，保持通用性。

dropout 逻辑：
本轮默认 dropout=None，不启用。
如果配置中给出 dropout，可在分类头前加入 Dropout。
但不要影响默认无 dropout 行为。

请在 factory 中加入清晰注释，说明每个 backbone 的分类头替换逻辑。

========================================
五、任务 2：小幅修改训练脚本模型构建入口
========================================

请修改：

scripts/train/train_nyha_3class_5fold.py

修改目标：
让 _build_model(config) 从通用 factory 构建模型，而不是只能调用 build_resnet_nyha_model。

建议修改为：

from models.nyha_backbone_factory import build_nyha_classification_model, count_parameters

然后在 _build_model(config) 中读取：

model_config = config["model"]
backbone = model_config["backbone"]
num_classes = int(model_config["num_classes"])
pretrained = _pretrained_enabled(model_config["pretrained"])
freeze_backbone = bool(model_config.get("freeze_backbone", False))
dropout = model_config.get("dropout", None)

model = build_nyha_classification_model(
    backbone=backbone,
    num_classes=num_classes,
    pretrained=pretrained,
    freeze_backbone=freeze_backbone,
    dropout=dropout,
)

然后打印或记录：

backbone
total_params
trainable_params

必须保持旧配置兼容：

resnet18
resnet34
resnet50

旧 ResNet 配置必须仍能正常跑。

不要修改：

dataset
transform
loss
optimizer
scheduler
early stopping
metrics
evaluator
fold split
class weights
summary format

如果当前训练脚本没有合适日志位置，可以至少 print 参数统计信息，并把信息写入每个 fold 的 logs/model_summary.txt 或实验根目录下的 model_summary.txt。

建议每个实验输出：

model_summary.txt

内容包括：

backbone
pretrained
freeze_backbone
num_classes
total_params
trainable_params
model_class_name

========================================
六、任务 3：生成多 backbone 配置
========================================

请新增配置目录：

config/train/model_exploration_imagenet_meanbg/

请新增脚本：

scripts/run/generate_model_exploration_configs.py

功能：
自动生成本轮 5 个模型配置。

配置模板：
优先读取当前已有 ResNet18 配置，例如：

config/train/nyha_3class_global224_imagenet_resnet18.yaml

如果该文件不存在，则从其他可用 ResNet 配置复制结构，但必须保持字段完整。

输出 5 个 YAML：

1. nyha_3class_densenet121_imagenet_meanbg.yaml
2. nyha_3class_efficientnet_b0_imagenet_meanbg.yaml
3. nyha_3class_convnext_tiny_imagenet_meanbg.yaml
4. nyha_3class_swin_t_imagenet_meanbg.yaml
5. nyha_3class_mobilenet_v3_large_imagenet_meanbg.yaml

每个配置统一设置：

experiment.output_dir:
experiments/model_exploration_500Data

data.split_dir:
data/processed/splits_500

data.image_root:
data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images

data.image_filename_template:
"{ID}.png"

data.n_folds:
5

data.image_size:
224

data.num_classes:
3

data.label_col:
label_3class

data.train_csv_pattern:
fold_{fold}_train.csv

data.val_csv_pattern:
fold_{fold}_val.csv

model.pretrained:
imagenet

model.num_classes:
3

model.freeze_backbone:
false

train.epochs:
50

train.optimizer:
AdamW

train.lr:
0.0001

train.weight_decay:
0.0001

train.loss:
weighted_cross_entropy

train.early_stopping_patience:
10

train.monitor_metric:
macro_auc

train.random_seed:
2026

train.num_workers:
0

train.pin_memory:
false

train.use_amp:
false

每个模型差异字段：

1. densenet121：
experiment.name = ModelExploration_DenseNet121_ImageNetMeanBG
model.backbone = densenet121
train.batch_size = 16

2. efficientnet_b0：
experiment.name = ModelExploration_EfficientNetB0_ImageNetMeanBG
model.backbone = efficientnet_b0
train.batch_size = 16

3. convnext_tiny：
experiment.name = ModelExploration_ConvNeXtTiny_ImageNetMeanBG
model.backbone = convnext_tiny
train.batch_size = 8

4. swin_t：
experiment.name = ModelExploration_SwinTiny_ImageNetMeanBG
model.backbone = swin_t
train.batch_size = 8

5. mobilenet_v3_large：
experiment.name = ModelExploration_MobileNetV3Large_ImageNetMeanBG
model.backbone = mobilenet_v3_large
train.batch_size = 16

生成 manifest：

config/train/model_exploration_imagenet_meanbg/model_exploration_config_manifest.csv

manifest 字段：

job_id
backbone
config_path
image_root
experiment_name
output_root
batch_size
supported
status
error_message

在生成配置前请检查：

1. data/processed/splits_500 是否存在；
2. data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images 是否存在；
3. torchvision 是否支持每个 backbone；
4. scripts/run/run_nyha3class_5fold_with_config.py 是否存在。

如果某模型当前 torchvision 不支持，则 manifest 中 supported=false，status=UNSUPPORTED，不生成对应训练作业，且不要安装新依赖。

========================================
七、任务 4：批量串行运行多 backbone 五折实验
========================================

请新增脚本：

scripts/run/run_model_exploration_nyha3class_5fold.py

功能：
读取：

config/train/model_exploration_imagenet_meanbg/model_exploration_config_manifest.csv

按顺序串行运行支持的模型：

1. densenet121
2. efficientnet_b0
3. convnext_tiny
4. swin_t
5. mobilenet_v3_large

每个模型调用：

E:/resarch/Anaconda3/envs/face_heart/python.exe scripts/run/run_nyha3class_5fold_with_config.py --config <config_path>

必须使用 subprocess.run 阻塞执行。
禁止使用非阻塞 Popen 后直接启动下一个任务。
禁止并行训练多个模型。

脚本参数：

--manifest
默认：
config/train/model_exploration_imagenet_meanbg/model_exploration_config_manifest.csv

--only
可选，只跑指定 backbone，逗号分隔。
例如：
--only densenet121,efficientnet_b0,convnext_tiny

--start-from
从指定 backbone 开始。

--dry-run
只打印将执行的命令，不真正训练。

--skip-existing
如果目标实验目录下已有完整 summary 文件，则跳过。

--resume
跳过 SUCCESS，继续 PENDING、FAILED 或 RUNNING 状态作业。

--rerun-failed
只重跑 FAILED 作业。

--continue-on-error
某个作业失败后继续后续作业。

--allow-cpu
默认不允许 CPU 长时间训练。
如果 CUDA 不可用且未设置 --allow-cpu，应停止训练并明确提示。

输出根目录：

experiments/model_exploration_500Data/

输出 job queue：

experiments/model_exploration_500Data/model_exploration_job_queue.csv

字段：

job_id
backbone
config_path
image_root
experiment_name
status
start_time
end_time
duration_minutes
output_dir
exit_code
error_message
total_params
trainable_params

status 取值：

PENDING
RUNNING
SUCCESS
FAILED
SKIPPED
UNSUPPORTED

日志目录：

experiments/model_exploration_500Data/logs/

每个模型保存：

<backbone>_stdout.log
<backbone>_stderr.log

每个作业开始前打印并记录 GPU 信息：

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

如果 CUDA 不可用，输出：

CUDA is not available.

如果未设置 --allow-cpu，则停止训练并标记作业 FAILED，避免静默 CPU 长时间训练。

每个作业训练完成后，必须检查以下文件是否存在：

<output_dir>/<experiment_name>/summary/fold_metrics_all.csv
<output_dir>/<experiment_name>/summary/mean_metrics.csv
<output_dir>/<experiment_name>/summary/oof_metrics.csv
<output_dir>/<experiment_name>/summary/oof_predictions.csv
<output_dir>/<experiment_name>/summary/summary_report.md

注意：
具体路径请根据 run_nyha3class_5fold_with_config.py 的实际输出逻辑确定。如果该脚本将 experiment.output_dir 和 experiment.name 拼接为最终实验目录，则按该逻辑检查。
不要凭空假设路径，先阅读该脚本确认。

如果 summary 文件缺失，不能标记 SUCCESS，应标记 FAILED。

========================================
八、任务 5：汇总多 backbone 实验结果
========================================

请新增脚本：

scripts/evaluate/summarize_model_exploration_experiments.py

功能：
扫描：

experiments/model_exploration_500Data/

读取每个模型实验的：

summary/fold_metrics_all.csv
summary/mean_metrics.csv
summary/oof_metrics.csv
summary/oof_predictions.csv
summary/summary_report.md

同时读取 ResNet18 meanbg 参考基线：

experiments/preprocess_ablation_500Data/PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg

并将该基线纳入对比。

输出：

experiments/model_exploration_500Data/model_exploration_summary.xlsx
experiments/model_exploration_500Data/model_exploration_summary.csv
experiments/model_exploration_500Data/model_exploration_summary.md

xlsx 至少包含以下 sheet：

1. experiment_summary

每个模型一行，字段包括：

backbone
experiment_name
output_dir
batch_size
total_params
trainable_params
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
severe_vs_rest_auc_mean
normal_vs_abnormal_auc_mean
recall_normal_mean
recall_mild_mean
recall_severe_mean
f1_normal_mean
f1_mild_mean
f1_severe_mean
oof_macro_auc
oof_balanced_accuracy
oof_macro_f1
oof_macro_recall
oof_accuracy
oof_severe_vs_rest_auc
oof_normal_vs_abnormal_auc

2. fold_metrics_all

所有模型每折指标长表。

3. oof_metrics_all

所有模型 OOF 指标。

4. delta_vs_resnet18_meanbg

以 ResNet18 + hybrid_imagenet_meanbg 为基线，计算每个模型相对差值：

delta_macro_auc_mean
delta_balanced_accuracy_mean
delta_macro_f1_mean
delta_recall_severe_mean
delta_f1_severe_mean
delta_severe_vs_rest_auc_mean
delta_normal_vs_abnormal_auc_mean
delta_oof_macro_auc
delta_oof_balanced_accuracy
delta_oof_macro_f1
delta_oof_severe_vs_rest_auc
delta_oof_normal_vs_abnormal_auc

基线参考值：

macro_auc = 0.7094
balanced_accuracy = 0.5353
macro_f1 = 0.5111
recall_severe = 0.4333

但不要只硬编码这几个值。
优先从基线目录的 summary/mean_metrics.csv 和 summary/oof_metrics.csv 中读取真实值。
如果读取失败，再使用上述数值作为 fallback，并在 summary.md 中注明 fallback。

5. ranking

至少按照以下指标分别排名：

rank_by_macro_auc_mean
rank_by_balanced_accuracy_mean
rank_by_macro_f1_mean
rank_by_recall_severe_mean
rank_by_oof_macro_auc
rank_by_oof_balanced_accuracy
rank_by_oof_macro_f1

6. model_complexity

字段：

backbone
total_params
trainable_params
batch_size
params_vs_resnet18_meanbg
performance_note

7. recommendation

自动给出每个模型是否进入第二轮调参。

推荐规则：

如果某模型相比 ResNet18 meanbg 满足以下任一条件，则标记为 candidate_for_tuning：

1. macro_auc_mean 下降不超过 0.02，且 balanced_accuracy_mean 提升 >= 0.02；
2. macro_auc_mean 下降不超过 0.02，且 macro_f1_mean 提升 >= 0.02；
3. severe recall 提升，且 macro_f1_mean 和 balanced_accuracy_mean 不明显下降；
4. oof_macro_f1 或 oof_balanced_accuracy 明显优于 ResNet18 meanbg。

如果模型 severe recall 很高，但 macro_f1 和 balanced_accuracy 明显下降，则标记为 high_severe_recall_but_unbalanced，不推荐作为主模型。

如果模型总体指标接近 ResNet18 meanbg，但参数明显更少，例如 MobileNetV3-Large，则标记为 lightweight_candidate。

如果模型 macro_auc、BA、macro-F1 均下降，则标记为 not_recommended。

summary.md 内容必须包括：

1. 实验目的；
2. 固定变量说明；
3. 模型清单；
4. 与 ResNet18 meanbg 的对比；
5. 哪个模型 macro-AUC 最好；
6. 哪个模型 balanced accuracy 最好；
7. 哪个模型 macro-F1 最好；
8. 哪个模型 severe recall 最好；
9. 哪些模型推荐进入第二轮调参；
10. 如果所有模型均未明显优于 ResNet18，则说明：
   当前性能瓶颈可能不主要来自 backbone，而可能来自 mild/severe 标签边界、类别不平衡、样本量、决策阈值或 ROI 信息融合不足。
11. 下一步建议：
   - top 2 模型做 lr/weight_decay/dropout/label smoothing 轻量调参；
   - 对 top 模型做混淆矩阵和 OOF 阈值扫描；
   - 若 backbone 改进有限，则转向 ordinal classification、two-stage classification 或 ROI/global fusion。

========================================
九、任务 6：总控脚本
========================================

请新增总控脚本：

scripts/run/run_model_exploration_pipeline.py

功能：
按顺序执行：

1. generate_model_exploration_configs.py
2. run_model_exploration_nyha3class_5fold.py
3. summarize_model_exploration_experiments.py

参数：

--skip-config-generation
--skip-training
--skip-summary
--dry-run
--only
--start-from
--resume
--rerun-failed
--skip-existing
--continue-on-error
--allow-cpu

输出日志：

experiments/model_exploration_500Data/model_exploration_pipeline_log.txt
experiments/model_exploration_500Data/model_exploration_pipeline_status.json

每个阶段记录：

stage_name
command
start_time
end_time
duration_minutes
status
return_code
error_message

如果 --dry-run：
只打印将执行的命令，不真正训练。

如果 --only densenet121,efficientnet_b0,convnext_tiny：
只运行这三个模型。

========================================
十、兼容性与安全检查
========================================

实现时请先阅读以下文件：

scripts/train/train_nyha_3class_5fold.py
scripts/run/run_nyha3class_5fold_with_config.py
models/resnet_nyha_3class.py
datasets/nyha_3class_face_dataset.py
metrics/classification_metrics.py
losses/classification_losses.py
config/train/nyha_3class_global224_imagenet_resnet18.yaml

在修改前先确认：

1. run_nyha3class_5fold_with_config.py 的实际 config 读取逻辑；
2. experiment.output_dir 和 experiment.name 如何组合为最终输出目录；
3. train 脚本 _build_model(config) 的实际位置；
4. 当前 metrics 文件中指标字段命名；
5. mean_metrics.csv 和 oof_metrics.csv 的字段命名；
6. 是否已有类似 model factory，避免重复冲突。

不得破坏旧 ResNet 训练。

请完成后做兼容性 smoke test：

1. 构建 resnet18：
调用 build_nyha_classification_model("resnet18", num_classes=3, pretrained=False)，确认可正常 forward。

2. 构建 densenet121：
确认可正常 forward。

3. 构建 efficientnet_b0：
确认可正常 forward。

4. 构建 convnext_tiny：
确认可正常 forward。

5. 构建 swin_t：
确认可正常 forward。

6. 构建 mobilenet_v3_large：
确认可正常 forward。

forward 输入：

torch.randn(2, 3, 224, 224)

输出 shape 必须是：

[2, 3]

请新增或使用简单测试脚本：

scripts/run/smoke_test_backbone_factory.py

输出每个 backbone：

backbone
status
output_shape
total_params
trainable_params
error_message

========================================
十一、测试命令
========================================

请在实现完成后执行或至少给出以下测试命令。

进入项目根目录：

cd /d E:\projects\face2

语法检查：

E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile models\nyha_backbone_factory.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\train\train_nyha_3class_5fold.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\generate_model_exploration_configs.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\run_model_exploration_nyha3class_5fold.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\evaluate\summarize_model_exploration_experiments.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\run_model_exploration_pipeline.py
E:\resarch\Anaconda3\envs\face_heart\python.exe -m py_compile scripts\run\smoke_test_backbone_factory.py

模型工厂 smoke test：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\smoke_test_backbone_factory.py

生成配置：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\generate_model_exploration_configs.py

检查 manifest：

config\train\model_exploration_imagenet_meanbg\model_exploration_config_manifest.csv

dry-run 批量训练：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --dry-run

只 dry-run 前三个模型：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --only densenet121,efficientnet_b0,convnext_tiny --dry-run

正式只跑前三个模型：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --only densenet121,efficientnet_b0,convnext_tiny --continue-on-error

正式跑全部五个模型：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --continue-on-error

断点续跑：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --resume --continue-on-error

只重跑失败：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_nyha3class_5fold.py --rerun-failed --continue-on-error

汇总结果：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\evaluate\summarize_model_exploration_experiments.py

一键 dry-run：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_pipeline.py --dry-run

一键正式运行前三个模型：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_pipeline.py --only densenet121,efficientnet_b0,convnext_tiny --continue-on-error

一键正式运行全部模型：

E:\resarch\Anaconda3\envs\face_heart\python.exe scripts\run\run_model_exploration_pipeline.py --continue-on-error

========================================
十二、最终交付说明
========================================

实现完成后，请汇报：

1. 新增了哪些文件；
2. 修改了哪些旧文件；
3. 是否保持 ResNet 旧配置兼容；
4. 是否新增 models/nyha_backbone_factory.py；
5. 每个 backbone 的分类头如何替换；
6. 是否通过 smoke_test_backbone_factory.py；
7. 生成了哪些 YAML 配置；
8. manifest 路径；
9. 如何只运行前三个模型；
10. 如何运行全部模型；
11. 如何断点续跑；
12. 如何只重跑失败；
13. 汇总结果输出在哪里；
14. 如何查看 model_exploration_summary.xlsx 和 model_exploration_summary.md；
15. 哪些模型进入第二轮调参；
16. 如果没有模型明显优于 ResNet18 meanbg，请明确说明可能的研究解释。

注意：
本轮实验是模型架构消融，不要把结论写成最终临床模型结论。
所有指标应继续按 5-fold mean ± std 和 OOF 两套结果同时报告。
如果模型结果只提高 severe recall 但 BA/macro-F1 下降，不推荐作为主模型。
如果模型 BA/macro-F1 提升且 severe recall 不下降，推荐进入第二轮调参。