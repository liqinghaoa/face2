你现在需要在当前面部NYHA三分类项目中，正式实现“Global ResNet18 + 六维光学表型融合对照实验”。

本任务包括：

1. 编写完整实验代码；
2. 新增模型、Dataset、特征预处理器、Trainer、Evaluator、runner和summary；
3. 新增配置文件；
4. 新增单元测试和协议测试；
5. 执行只读preflight；
6. 执行CPU可完成的单元测试、协议测试和轻量冒烟测试；
7. 生成服务器正式训练命令；
8. 不在当前CPU-only环境启动完整25次训练。

不得修改或重新运行第一阶段、Stage 2A、Stage 2B。

==================================================
一、项目与实现背景
==================================================

实际项目根目录：

E:\projects\face2

当前仓库存在用户未提交的修改，尤其包括：

- trainers/nyha_3class_trainer.py
- scripts/train/train_nyha_3class_5fold.py
- scripts/evaluate/summarize_nyha_3class_5fold.py
- Stage 1/2A/2B相关代码、配置、测试和报告

因此本任务必须优先新增独立文件，禁止覆盖或重写用户已有修改。

正式数据队列为500例，三分类数量：

- normal：115；
- mild：237；
- severe：148。

类别索引固定为：

- normal = 0；
- mild = 1；
- severe = 2。

原始NYHA映射：

- NYHA 0 → normal；
- NYHA 1、2 → mild；
- NYHA 3、4 → severe。

固定五折：

- fold编号0–4；
- 每折400例train；
- 每折100例val；
- 五折val拼接后为500例唯一ID；
- 按label_3class × SEX分层；
- patient_group_id不跨fold。

==================================================
二、实验目标
==================================================

实现并公平比较以下五个variant：

1. global_only / G0

输入：

- Global meanbg RGB图像；
- ResNet18输出512维Global特征；
- 不读取任何光学特征；
- 不读取forehead_available；
- 分类头输入维度512。

2. global_mask / G-Mask

输入：

- 512维Global特征；
- 一维forehead_available；
- 分类头输入维度513。

3. global_raw / G-Raw

输入：

- 512维Global特征；
- 第一阶段Raw六维光学表型；
- 一维forehead_available；
- 分类头输入维度519。

4. global_stage2a / G-A

输入：

- 512维Global特征；
- Stage 2A Ridge calibrated六维表型；
- 一维forehead_available；
- 分类头输入维度519。

5. global_stage2b / G-B

输入：

- 512维Global特征；
- Stage 2B MLP calibrated六维表型；
- 一维forehead_available；
- 分类头输入维度519。

核心比较：

- G-Mask vs G0；
- G-Raw vs G-Mask；
- G-A vs G-Mask；
- G-B vs G-Mask；
- G-A vs G-Raw；
- G-B vs G-Raw；
- G-B vs G-A。

==================================================
三、先检查仓库状态
==================================================

开始实现前必须：

1. 确认实际project root；
2. 运行git status；
3. 记录branch和commit；
4. 检查计划新增的目标文件是否已经存在；
5. 检查现有未提交修改；
6. 不得执行git reset、checkout、clean或任何破坏性操作；
7. 不得覆盖用户已有文件；
8. 如果建议新增的文件已经存在，先读取并判断来源；
9. 如果存在无法安全合并的冲突，停止并报告，不得静默覆盖。

本任务原则上不修改现有历史文件。

为避免修改__init__.py，允许直接通过完整模块路径导入新增模块。

==================================================
四、必须先阅读的现有代码
==================================================

编码前必须实际读取并理解：

1. models/resnet_nyha_3class.py
2. models/nyha_backbone_factory.py
3. 当前NYHA3ClassFaceDataset所在文件
4. trainers/nyha_3class_trainer.py
5. 当前NYHA evaluator
6. scripts/train/train_nyha_3class_5fold.py
7. scripts/evaluate/summarize_nyha_3class_5fold.py
8. 当前metrics实现
9. 当前weighted CrossEntropy实现
10. 当前build_transforms实现
11. 当前YAML配置读取方式
12. 当前checkpoint和resume格式
13. 当前run manifest风格
14. Stage 1/2A/2B的schema、manifest和相关测试
15. 预实现审计报告：

reports/global_resnet18_optical_fusion_preimplementation/
global_resnet18_optical_fusion_preimplementation_audit.md

必须尽量复用：

- ResNet18构造语义；
- ImageNet权重选择；
- 图像transform；
- weighted CrossEntropy；
- 分类metrics；
- checkpoint选择逻辑；
- 训练日志和曲线风格；
- OOF输出格式；
- YAML解析风格；
- Stage 1/2A/2B的schema和哈希验证风格。

不得仅根据本提示词概念性创建代码。

==================================================
五、正式数据来源
==================================================

A. Global meanbg图像

根目录：

data/processed/global_face/preprocess_ablation/
hybrid_imagenet_meanbg/images/

命名格式：

{ID}.png

要求：

- 总计500张；
- 使用Pillow读取；
- convert("RGB")；
- 输入224×224；
- 不重新生成meanbg；
- 不再次进行背景替换。

B. 固定五折

根目录：

data/processed/splits_500/

正式master split：

data/processed/splits_500/
nyha_3class_sex_stratified_group_5fold.csv

同时读取现有实际逐折train/val文件命名方式，不能只假设文件名。

要求：

- fold=0–4；
- 每折400/100；
- train/val ID交集为0；
- patient_group_id交集为0；
- 五个val恰好覆盖500例；
- split角色由明确的train/val文件路径决定；
- 不重新生成split。

C. 标签

原始标签血缘：

data/raw/label_raw_nyha2_remove22_sex_balanced_500.csv

实际训练优先沿用split CSV中的label_3class。

要求：

- ID按字符串读取；
- 不丢失“-1”等后缀；
- 不按数值类型读取ID；
- 类别顺序固定为normal、mild、severe；
- 不重新生成或修改标签。

==================================================
六、第一阶段Raw特征
==================================================

正式文件：

data/processed/optical_observations_v1/
regional_optical_observations.csv

schema：

data/processed/optical_observations_v1/
feature_schema.json

manifest：

data/processed/optical_observations_v1/
extraction_manifest.json

Raw六维固定顺序：

1. cheek_mean_log2_y
2. cheek_mean_log2_rg
3. cheek_mean_log2_bg
4. forehead_minus_cheek_log2_y
5. forehead_minus_cheek_log2_rg
6. forehead_minus_cheek_log2_bg

availability字段：

forehead_available

统计期望：

- 500行；
- 500唯一ID；
- forehead_available=1：486例；
- forehead_available=0：14例；
- 前三项cheek全部有限；
- 后三项仅14例unavailable为NaN。

G-Mask只允许读取：

- ID；
- forehead_available。

G-Raw只允许读取：

- ID；
- 上述六维Raw；
- forehead_available。

禁止自动读取“所有数值字段”。

==================================================
七、Stage 2A特征
==================================================

正式根目录：

experiments/optical_condition_calibration_stage2a/

schema：

experiments/optical_condition_calibration_stage2a/
summary/calibration_feature_schema.json

run manifest：

experiments/optical_condition_calibration_stage2a/
summary/run_manifest.json

G-A六维固定顺序：

1. calibrated_cheek_mean_log2_y
2. calibrated_cheek_mean_log2_rg
3. calibrated_cheek_mean_log2_bg
4. calibrated_forehead_minus_cheek_log2_y
5. calibrated_forehead_minus_cheek_log2_rg
6. calibrated_forehead_minus_cheek_log2_bg

分类fold k必须读取：

train：

experiments/optical_condition_calibration_stage2a/
fold_k/train_calibrated_features.csv

val：

experiments/optical_condition_calibration_stage2a/
fold_k/val_calibrated_features.csv

只允许读取：

- ID；
- fold；
- split_role；
- forehead_available；
- 六个calibrated字段。

禁止输入：

- raw_*；
- predicted_acquisition_*；
- residual_*；
- camera_id；
- EXIF；
- condition；
- z条件；
- QC；
- NYHA；
- SEX；
- 其他数值字段。

禁止使用：

summary/oof_calibrated_features.csv

作为分类器train输入。

OOF文件只允许用于输入完整性审计，不得重新拆成400/100。

==================================================
八、Stage 2B特征
==================================================

正式根目录：

experiments/optical_condition_calibration_stage2b/

schema：

experiments/optical_condition_calibration_stage2b/
summary/calibration_stage2b_feature_schema.json

run manifest：

experiments/optical_condition_calibration_stage2b/
summary/run_manifest.json

G-B六维固定顺序：

1. calibrated_nn_cheek_mean_log2_y
2. calibrated_nn_cheek_mean_log2_rg
3. calibrated_nn_cheek_mean_log2_bg
4. calibrated_nn_forehead_minus_cheek_log2_y
5. calibrated_nn_forehead_minus_cheek_log2_rg
6. calibrated_nn_forehead_minus_cheek_log2_bg

分类fold k必须读取：

train：

experiments/optical_condition_calibration_stage2b/
fold_k/train_nn_calibrated_features.csv

val：

experiments/optical_condition_calibration_stage2b/
fold_k/val_nn_calibrated_features.csv

只允许读取：

- ID；
- fold；
- split_role；
- forehead_available；
- 六个calibrated_nn字段。

禁止输入：

- raw_*；
- predicted_condition_nn_*；
- residual_nn_*；
- camera_id；
- EXIF；
- condition；
- z条件；
- QC；
- NYHA；
- SEX；
- 其他数值字段。

禁止使用：

summary/oof_nn_calibrated_features.csv

作为分类器train输入。

==================================================
九、正向字段白名单
==================================================

必须在代码中使用显式variant→字段映射，不得通过：

- dtype；
- 正则匹配所有calibrated字段；
- 排除少数字段；
- CSV剩余数值列；
- 列位置；

自动推断输入。

建立不可变常量，例如：

VARIANT_FEATURE_COLUMNS = {
    "global_only": [],
    "global_mask": [],
    "global_raw": [...6 fields...],
    "global_stage2a": [...6 fields...],
    "global_stage2b": [...6 fields...],
}

VARIANT_AUX_DIM = {
    "global_only": 0,
    "global_mask": 1,
    "global_raw": 7,
    "global_stage2a": 7,
    "global_stage2b": 7,
}

要求：

- 字段顺序与schema交叉核验；
- schema顺序与代码顺序不一致时停止；
- 禁止字段意外进入时停止；
- aux最终shape必须严格匹配variant；
- G0不得读取Stage 1、2A或2B特征文件；
- G-Mask不得读取六维数值，只读取availability。

==================================================
十、特征预处理器
==================================================

新增：

utils/optical_feature_preprocessor.py

实现独立、可测试的fit/transform接口。

建议包含：

- variant定义；
- 字段白名单；
- 特征源解析；
- schema验证；
- ID和split验证；
- FeatureScaler数据类；
- fit；
- transform；
- save_json；
- load_json；
- 哈希和manifest生成；
- 禁止OOF路径检查。

A. 计算精度

- mean/std使用float64计算；
- ddof=0；
- transform后转换为float32；
- std阈值为1e-8；
- std<1e-8立即报错；
- 不静默替换std；
- 不winsorize；
- 不clip；
- 不删除异常病例。

B. Cheek三维

前三项使用当前outer fold全部400个train病例计算：

- mean；
- std；
- valid_n。

要求：

- 全部有限；
- 任一NaN或Inf立即报错。

C. Forehead-minus-cheek三维

后三项只使用当前outer train中：

forehead_available == 1

的病例计算mean/std。

要求：

- available病例必须全部有限；
- unavailable病例应为NaN；
- availability与NaN模式严格匹配；
- 不能把unavailable病例的NaN当作0参与均值；
- 不能用val病例拟合。

D. 缺失填充

对于forehead_available=0：

- 保留前三项cheek；
- 后三项在标准化之后填0；
- availability作为最后一维0；
- 不删除病例。

对于forehead_available=1：

- 后三项正常标准化；
- availability作为最后一维1。

E. variant行为

global_only：

- 不拟合scaler；
- aux维度0。

global_mask：

- 不拟合六维scaler；
- aux只有availability；
- aux维度1。

global_raw/global_stage2a/global_stage2b：

- 独立拟合当前fold当前variant的六维scaler；
- aux为6个标准化特征+availability；
- aux维度7。

Raw、2A、2B不得共用scaler。

F. scaler保存

每个fold、每个需要六维特征的variant保存：

feature_scaler.json

至少包含：

- schema_version；
- variant；
- fold；
- feature_names及顺序；
- mean；
- std；
- valid_n；
- forehead_available_train_n；
- forehead_unavailable_train_n；
- ddof=0；
- std_epsilon；
- missing_fill_after_standardization=0；
- availability_position；
- source相对路径；
- source SHA256；
- schema SHA256；
- train ID SHA256；
- split SHA256；
- fit_timestamp；
- code/config SHA256。

checkpoint中保存相同scaler payload或其完整哈希。

Evaluator只允许load和transform，禁止refit。

==================================================
十一、Dataset
==================================================

新增：

datasets/global_optical_fusion_dataset.py

建议实现：

GlobalOpticalFusionDataset

优先采用组合方式复用现有NYHA3ClassFaceDataset的图像加载、标签、ID和transform逻辑，不修改历史Dataset。

要求：

1. split CSV行顺序是唯一主顺序；
2. 特征表通过完整字符串ID进行一对一join；
3. 不按行号或排序位置join；
4. join后保持split CSV原行顺序；
5. 不允许缺失ID；
6. 不允许额外ID；
7. 不允许重复ID；
8. Stage 2A/2B的fold必须等于当前classification fold；
9. Stage 2A/2B的split_role必须与train/val文件角色一致；
10. train与val必须使用各自对应文件；
11. G0不读取任何光学表；
12. G-Mask只读取Stage 1 availability；
13. G-Raw读取Stage 1；
14. G-A读取同fold Stage 2A；
15. G-B读取同fold Stage 2B。

统一batch建议返回：

- image；
- aux_features；
- label；
- ID；
- patient_group_id；
- NYHA；
- SEX；
- sex_name；
- label_3class_name；
- fold；
- split_role。

其中：

- global_only的aux_features形状为[0]；
- DataLoader后为[B,0]；
- global_mask为[B,1]；
- 其余为[B,7]。

aux_features必须为float32且全部有限。

禁止把以下字段放入aux tensor：

- label；
- NYHA；
- SEX；
- patient_group_id；
- fold；
- split_role；
- camera_id；
- EXIF；
- predicted condition；
- residual；
- QC。

image_path可以在Dataset内部使用，但不得写入最终OOF CSV绝对路径。

==================================================
十二、图像transform
==================================================

严格复用正式meanbg配置：

train：

1. Resize(224,224)；
2. RandomHorizontalFlip(p=0.5)；
3. ToTensor；
4. ImageNet Normalize。

val：

1. Resize(224,224)；
2. ToTensor；
3. ImageNet Normalize。

ImageNet：

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

禁止加入：

- RandomCrop；
- RandomResizedCrop；
- Rotation；
- ColorJitter；
- brightness；
- contrast；
- saturation；
- hue；
- gamma；
- RandomErasing；
- MixUp；
- CutMix；
- 任何新增强。

水平翻转不改变六维特征，因为六维只包含脸颊均值和额部−脸颊差值，没有左右方向性。

==================================================
十三、融合模型
==================================================

新增：

models/resnet18_optical_fusion.py

建议实现：

ResNet18OpticalFusion

要求：

1. 使用与现有Global基线相同的torchvision ResNet18；
2. pretrained=ImageNet时使用：
   ResNet18_Weights.IMAGENET1K_V1；
3. ResNet18原fc替换为Identity；
4. forward_features(image)输出[B,512]；
5. backbone全部可训练；
6. 不冻结；
7. 不分阶段解冻；
8. 不设置差异学习率；
9. 无Dropout；
10. 无BatchNorm新增；
11. 无MLP；
12. 无投影层；
13. 无attention；
14. 无gate；
15. 无FiLM；
16. 无特征交互模块；
17. 只做拼接后单Linear分类。

模型结构：

global_only：

global_features [B,512]
→ Linear(512,3)

global_mask：

global_features [B,512]
+
aux [B,1]
→ concat [B,513]
→ Linear(513,3)

global_raw/global_stage2a/global_stage2b：

global_features [B,512]
+
aux [B,7]
→ concat [B,519]
→ Linear(519,3)

分类头参数量期望：

- global_only：1539；
- global_mask：1542；
- 三个融合variant：1560。

forward签名建议统一为：

forward(images, aux_features=None)

或：

forward(images, aux_features)

但必须满足：

- G0只接受None或[B,0]；
- G-Mask严格要求[B,1]；
- 其他严格要求[B,7]；
- batch维必须一致；
- dtype兼容；
- aux必须有限；
- availability必须为0或1；
- 维度错误立即报错；
- 不允许静默截断或padding。

输出为未归一化logits [B,3]。

softmax只在Evaluator和指标计算时使用。

==================================================
十四、模型初始化和公平性
==================================================

五个variant必须：

- 使用相同ImageNet backbone权重；
- 使用相同fold；
- 使用相同训练数据；
- 使用相同train顺序；
- 使用相同val顺序；
- 使用相同图像增强协议；
- 使用相同loss；
- 使用相同训练预算；
- 使用相同checkpoint规则。

不允许：

- 从历史G0 checkpoint初始化融合模型；
- 先训练G0再继续训练融合模型；
- 给某个variant额外epoch；
- 给某个variant不同learning rate；
- 给某个variant不同early stopping；
- 搜索多个seed后选择最好结果。

建议固定：

base_seed = 2026
fold_seed = base_seed + fold

同一fold的五个variant使用相同fold_seed。

至少分离并记录：

- model seed；
- DataLoader shuffle generator seed；
- augmentation seed；
- NumPy seed；
- Python random seed；
- torch CPU seed；
- torch CUDA seed。

为了让不同分类头维度不影响数据随机流：

1. 每个variant开始时重新设置seed；
2. 构建模型；
3. 构建模型后重新设置数据/augmentation seed；
4. DataLoader显式传入torch.Generator；
5. num_workers=0；
6. 每个epoch开始前重置该epoch的augmentation seed；
7. 相同fold、相同epoch的五个variant应产生相同样本顺序和水平翻转序列。

不需要创建复杂的增强系统；应以最小可靠实现为原则。

保存每个run的实际seed值。

==================================================
十五、训练配置
==================================================

新增：

config/train/global_optical_fusion/
global_resnet18_optical_fusion.yaml

正式配置固定：

device: auto
backbone: resnet18
pretrained: imagenet
num_classes: 3
image_size: 224

batch_size: 16
epochs: 50
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 1e-4

scheduler: none
warmup: none
gradient_clipping: none

loss: weighted_cross_entropy
label_smoothing: 0
AMP: false

early_stopping_patience: 10
monitor_metric: macro_auc
monitor_mode: max
minimum_improvement: 0
tie_breaking: earlier_epoch

seed: 2026
num_workers: 0
pin_memory: false

transforms:
  resize: [224,224]
  horizontal_flip_probability: 0.5
  imagenet_normalization: true
  color_jitter: false
  random_crop: false

feature_standardization:
  ddof: 0
  std_epsilon: 1e-8
  missing_fill_after_standardization: 0
  train_only: true

配置支持：

variants:
  - global_only
  - global_mask
  - global_raw
  - global_stage2a
  - global_stage2b

auxiliary_input_dim和fused_input_dim必须由variant推导并断言，不允许用户配置出不一致值。

==================================================
十六、损失函数
==================================================

严格复用现有fold-specific weighted CrossEntropy。

每个fold只使用当前400例train标签计算：

weight_c =
    N_train
    /
    (num_classes * count_c)

要求：

- G0、G-Mask、G-Raw、G-A、G-B使用相同权重；
- 权重只由当前fold train计算；
- val不参与；
- reduction="mean"；
- 不使用label smoothing；
- 不同时使用其他loss；
- 不加入auxiliary loss；
- 不加入feature reconstruction loss；
- 不加入camera adversarial loss；
- 不加入EXIF decorrelation loss；
- 不加入设备不变性loss。

==================================================
十七、Trainer
==================================================

新增：

trainers/global_optical_fusion_trainer.py

优先复用现有Trainer的：

- optimizer创建；
- weighted loss；
- train/val循环；
- Macro-AUC计算；
- early stopping；
- checkpoint；
- CSV日志；
- 曲线图；
- resume语义。

但不得覆盖现有Trainer。

训练时：

logits = model(images, aux_features)

每epoch记录：

- epoch；
- train_loss；
- val_loss；
- train_accuracy；
- val_accuracy；
- train_macro_auc；
- val_macro_auc；
- val_balanced_accuracy；
- val_macro_f1；
- learning_rate；
- elapsed_seconds；
- is_best；
- patience_counter。

checkpoint选择：

- 指标：val Macro-AUC；
- 新值严格大于旧值才算改善；
- 相等时保留较早epoch；
- patience=10；
- 最多50 epochs；
- 保存best_macro_auc.pth；
- 保存last_checkpoint.pth。

必须承认：

outer val每epoch参与early stopping，因此这里是五折held-out validation，不是独立test。

所有variant必须使用相同规则。

==================================================
十八、checkpoint内容
==================================================

每个best/last checkpoint至少保存：

- model_state_dict；
- optimizer_state_dict；
- epoch；
- best_epoch；
- best_macro_auc；
- patience_counter；
- fold；
- variant；
- architecture；
- backbone；
- pretrained weights名称；
- num_classes；
- global_feature_dim；
- auxiliary_input_dim；
- fused_input_dim；
- classifier_head结构；
- parameter_count；
- trainable_parameter_count；
- feature_names及顺序；
- availability位置；
- feature scaler payload或哈希；
- feature source相对路径；
- feature source SHA256；
- feature schema SHA256；
- Stage 1/2A/2B manifest SHA256；
- train ID SHA256；
- val ID SHA256；
- split SHA256；
- config；
- config SHA256；
- class mapping；
- class weights；
- transform定义；
- seed信息；
- Python版本；
- PyTorch版本；
- torchvision版本；
- CUDA版本；
- device；
- git commit；
- clinical fields used=false；
- camera used=false；
- exif used=false；
- outer_validation_tuning=true；
- historical_inputs_modified=false。

resume时必须恢复：

- model；
- optimizer；
- epoch；
- best metric；
- patience；
- RNG状态；
- scaler一致性。

恢复后必须验证variant、feature order、scaler hash和split hash一致，不一致时拒绝resume。

==================================================
十九、Evaluator
==================================================

新增：

evaluators/global_optical_fusion_evaluator.py

要求：

1. 加载best_macro_auc.pth；
2. 恢复模型；
3. 加载并验证feature scaler；
4. 不得重新fit scaler；
5. 对当前fold val进行一次最终推理；
6. softmax得到三类概率；
7. 保存逐例结果；
8. 计算现有全部分类指标；
9. 保存confusion matrix；
10. 验证checkpoint与当前variant/fold一致。

逐折val_predictions.csv至少包含：

- ID；
- patient_group_id；
- fold；
- true_label；
- true_class_name；
- prob_normal；
- prob_mild；
- prob_severe；
- pred_class；
- pred_class_name；
- correct；
- NYHA；
- SEX；
- sex_name；
- forehead_available。

其中：

- G0的forehead_available可以保留为仅评估/QC字段，但不得作为模型输入；
- 不包含绝对image_path；
- 不包含六维具体数值；
- 不包含camera_id；
- 不包含EXIF；
- 不包含predicted condition；
- 不包含residual；
- 不包含QC中间量。

指标至少包括：

- Accuracy；
- Balanced Accuracy；
- Macro Precision；
- Macro Recall；
- Macro F1；
- Macro OvR AUC；
- normal Precision/Recall/F1/AUC；
- mild Precision/Recall/F1/AUC；
- severe Precision/Recall/F1/AUC；
- severe-vs-rest AUC；
- normal-vs-abnormal AUC；
- confusion matrix。

类别顺序固定：

normal, mild, severe

==================================================
二十、五折OOF与汇总
==================================================

新增：

scripts/evaluate/
summarize_global_optical_fusion_5fold.py

每个variant完成五折后拼接五个val_predictions.csv，生成：

- 500行；
- 500唯一ID；
- 每ID恰好出现一次；
- fold=0–4；
- fold映射与固定split完全一致；
- 概率有限；
- 每行概率和为1；
- 无重复；
- 无缺失。

每个variant输出：

summary/{variant}/
    fold_metrics.csv
    aggregate_fold_metrics.csv
    oof_predictions.csv
    oof_metrics.json
    oof_confusion_matrix.csv
    oof_confusion_matrix.png
    training_curve_summary.png

aggregate_fold_metrics至少报告：

- 五折mean；
- 五折sample std，ddof=1；
- median；
- min；
- max；
- valid fold数量。

同时计算500例pooled OOF指标。

五折平均指标和pooled OOF指标必须分别报告，不能混为一项。

==================================================
二十一、配对比较
==================================================

实现以下配对比较：

1. global_mask - global_only
2. global_raw - global_mask
3. global_stage2a - global_mask
4. global_stage2b - global_mask
5. global_stage2a - global_raw
6. global_stage2b - global_raw
7. global_stage2b - global_stage2a

生成：

summary/pairwise_comparison.csv
summary/foldwise_metric_deltas.csv
summary/oof_metrics_all_variants.csv
summary/pairwise_bootstrap_deltas.csv

A. 逐fold差值

对以下指标计算candidate-reference：

- Macro-AUC；
- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- 三类AUC；
- 三类Recall；
- severe-vs-rest AUC；
- normal-vs-abnormal AUC。

每项报告：

- fold 0–4差值；
- 平均差；
- sample std；
- 中位数；
- 最小值；
- 最大值；
- candidate更优的fold数量；
- reference更优的fold数量；
- 完全相等的fold数量。

B. OOF差值

严格按：

- ID；
- patient_group_id；
- fold；
- true_label；

对齐两个variant。

对齐不一致时停止，不得通过排序强行拼接。

C. 患者组级配对bootstrap

由于500张图像对应483个patient_group_id，bootstrap必须以patient_group_id为采样单位，不能把同患者多图当作完全独立样本。

建议固定：

- bootstrap_repetitions=2000；
- bootstrap_seed=2026；
- percentile CI；
- 2.5%和97.5%；
- 同一个bootstrap样本同时用于reference和candidate；
- 按true_label分层抽取patient_group_id；
- 保留被抽中patient group的全部图像；
- patient group被重复抽中时相应重复其全部图像。

至少对以下delta输出95% CI：

- Macro-AUC；
- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- normal AUC；
- mild AUC；
- severe AUC。

不得根据bootstrap p值自动宣布胜负。

如果bootstrap样本计算失败：

- 记录失败原因；
- 不填伪造结果；
- 报告有效重复次数；
- 有效重复次数不足预设阈值时停止summary。

==================================================
二十二、六维特征分布审计
==================================================

生成：

summary/feature_distribution_audit.csv

对global_raw、global_stage2a、global_stage2b的每fold、每个六维特征记录：

- train valid_n；
- val valid_n；
- train mean；
- train std；
- train median；
- train min；
- train max；
- val mean；
- val std；
- val median；
- val min；
- val max；
- standardized mean difference；
- train unavailable数量；
- val unavailable数量；
- train availability比例；
- val availability比例。

同时可记录标准化后的train/val统计。

该审计只用于解释：

- Raw/2A/2B的折间变化；
- Stage 2B train in-sample与val out-of-sample分布偏移。

禁止根据该审计：

- 删除病例；
- 重新拟合Stage 2B；
- 调整loss；
- 调整网络；
- 对某个variant单独标准化到验证集；
- 事后修改特征。

==================================================
二十三、Runner
==================================================

新增：

scripts/run/
run_global_optical_fusion_5fold.py

以及：

scripts/train/
train_global_optical_fusion_5fold.py

runner至少支持：

--config
--variant global_only
--variant global_mask
--variant global_raw
--variant global_stage2a
--variant global_stage2b
--variant all

--fold 0
--fold all

--protocol-only
--smoke-test
--summarize-only
--resume
--skip-completed
--overwrite
--allow-cpu-training

默认行为：

- 不覆盖已有结果；
- 已完成run在skip-completed时跳过；
- 未明确overwrite时不得删除；
- overwrite只能作用于本次新实验目录；
- 不得删除历史实验。

设备规则：

1. protocol-only和单元测试允许CPU；
2. smoke-test允许CPU；
3. 正式all variants × all folds训练时，若CUDA不可用：
   - 默认拒绝启动；
   - 打印明确原因；
   - 只有显式--allow-cpu-training才允许；
4. 不安装或升级PyTorch；
5. 正式服务器训练device=auto时优先CUDA；
6. 使用sys.executable调用子进程；
7. 路径全部project-root相对解析；
8. 必须兼容Windows路径。

smoke-test要求：

- 不下载新依赖；
- 不运行完整fold；
- 可在临时目录或独立smoke目录；
- 使用很小样本；
- 最多1个epoch；
- 验证五个variant的完整forward/backward/checkpoint/evaluator通路；
- 验证三种aux维度；
- 验证保存和加载；
- 不写入正式fold完成标志；
- 不把smoke结果当正式结果。

==================================================
二十四、Preflight
==================================================

正式训练前必须通过preflight。

生成：

experiments/global_resnet18_optical_fusion/
protocol/
    environment_audit.json
    input_audit.csv
    feature_source_audit.csv
    fold_alignment_audit.csv
    schema_audit.json
    protocol_manifest.json

必须验证：

1. meanbg图像500张；
2. 标签500个唯一ID；
3. split 500个唯一ID；
4. Stage 1 500个唯一ID；
5. Stage 2A OOF 500个唯一ID；
6. Stage 2B OOF 500个唯一ID；
7. 六个集合完全一致；
8. 类别数量115/237/148；
9. 每fold train/val为400/100；
10. train/val ID无交集；
11. patient_group_id无交集；
12. 五折val覆盖500例一次；
13. Stage 2A每fold train/val文件存在；
14. Stage 2B每fold train/val文件存在；
15. Stage 2A/B split_role正确；
16. Stage 2A/B fold正确；
17. Stage 2A/B与分类split逐折一致；
18. Stage 1/2A/2B Raw字段最大差不超过允许浮点误差；
19. availability为486/14；
20. cheek字段全部有限；
21. forehead字段缺失与availability严格一致；
22. schema字段顺序正确；
23. run manifest状态为COMPLETE；
24. NYHA未进入Stage 1/2A/2B aux字段；
25. camera/EXIF/condition/predicted/residual/QC未进入正向白名单；
26. OOF文件未被配置为classifier train source；
27. 配置训练参数与锁定协议一致。

任一关键检查失败，停止正式训练。

==================================================
二十五、代码文件计划
==================================================

优先新增：

models/resnet18_optical_fusion.py
datasets/global_optical_fusion_dataset.py
utils/optical_feature_preprocessor.py
trainers/global_optical_fusion_trainer.py
evaluators/global_optical_fusion_evaluator.py
scripts/train/train_global_optical_fusion_5fold.py
scripts/run/run_global_optical_fusion_5fold.py
scripts/evaluate/summarize_global_optical_fusion_5fold.py
config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml
tests/test_global_optical_fusion_model.py
tests/test_global_optical_fusion_dataset.py
tests/test_global_optical_fusion_protocol.py
tests/test_global_optical_fusion_checkpoint.py
tests/test_global_optical_fusion_summary.py

如果当前项目实际目录命名略有不同，可以做最小适配。

原则：

- 不修改历史Global模型；
- 不修改历史Dataset；
- 不修改历史Trainer；
- 不修改历史Evaluator；
- 不修改Stage 1/2A/2B；
- 不修改split、标签、图像；
- 不进行无关重构；
- 不为了减少少量重复代码而重构整个训练框架。

==================================================
二十六、输出结构
==================================================

正式实验输出：

experiments/global_resnet18_optical_fusion/
    protocol/
        environment_audit.json
        input_audit.csv
        feature_source_audit.csv
        fold_alignment_audit.csv
        schema_audit.json
        protocol_manifest.json

    global_only/
        fold_0/
            resolved_config.yaml
            training_log.csv
            training_curves.png
            best_macro_auc.pth
            last_checkpoint.pth
            val_predictions.csv
            metrics.json
            confusion_matrix.csv
            confusion_matrix.png
            feature_distribution.csv
            fold_manifest.json
        ...
        fold_4/

    global_mask/
        fold_0/
        ...
        fold_4/

    global_raw/
        fold_0/
            feature_scaler.json
            ...
        ...
        fold_4/

    global_stage2a/
        fold_0/
            feature_scaler.json
            ...
        ...
        fold_4/

    global_stage2b/
        fold_0/
            feature_scaler.json
            ...
        ...
        fold_4/

    summary/
        global_only/
            fold_metrics.csv
            aggregate_fold_metrics.csv
            oof_predictions.csv
            oof_metrics.json
            oof_confusion_matrix.csv
            oof_confusion_matrix.png

        global_mask/
        global_raw/
        global_stage2a/
        global_stage2b/

        oof_metrics_all_variants.csv
        foldwise_metric_deltas.csv
        pairwise_comparison.csv
        pairwise_bootstrap_deltas.csv
        feature_distribution_audit.csv
        experiment_summary.json
        run_manifest.json

报告输出：

reports/global_resnet18_optical_fusion/
    global_resnet18_optical_fusion_implementation_report.md
    global_resnet18_optical_fusion_report.md
    oof_metrics_all_variants.csv
    pairwise_comparison.csv
    pairwise_bootstrap_deltas.csv
    foldwise_metric_deltas.csv
    feature_distribution_audit.csv
    training_curves/
    confusion_matrices/

在本地仅完成代码实现和测试时：

- 生成implementation_report；
- 不生成虚假的正式实验结果；
- global_resnet18_optical_fusion_report.md可以由summarize-only在服务器五个variant全部完成后生成；
- 未完整训练时不得写EXPERIMENT_STATUS=COMPLETE。

==================================================
二十七、单元测试
==================================================

测试必须能在CPU运行，并避免下载ImageNet权重。测试模型可以使用pretrained=false或mock backbone。

A. 模型测试

- forward_features输出[B,512]；
- G0输出[B,3]；
- G-Mask输出[B,3]；
- G-Raw/G-A/G-B输出[B,3]；
- 分类头输入维度512/513/519；
- 分类头参数量1539/1542/1560；
- 只有一个Linear分类头；
- 无额外MLP；
- 无新增BN；
- 无Dropout；
- 无attention/gate/FiLM；
- 错误aux维度报错；
- batch不一致报错；
- aux NaN/Inf报错；
- mask非0/1报错；
- G0输入非空aux报错；
- 非法variant报错。

B. scaler测试

- ddof=0；
- float64拟合；
- float32输出；
- 只使用train；
- val不影响mean/std；
- cheek使用全部train；
- forehead只使用available train；
- unavailable不参与mean/std；
- 标准化后unavailable后三维为0；
- cheek仍保留；
- mask正确附加；
- std<1e-8报错；
- available但NaN时报错；
- unavailable但有限值时按正式规则检查并明确处理；
- save/load逐元素一致；
- source hash和train ID hash一致。

C. Dataset测试

- ID字符串和后缀保留；
- split顺序保持；
- ID一对一join；
- 重复ID报错；
- 缺失ID报错；
- 额外ID报错；
- 2A/2B fold不一致报错；
- split_role不一致报错；
- train/val文件串用报错；
- 2A/2B串用报错；
- OOF路径报错；
- G0不读取特征；
- G-Mask只读取availability；
- aux shape正确；
- 禁止字段不进入aux。

D. checkpoint测试

- 保存后恢复模型；
- 恢复前后预测一致；
- variant一致；
- feature order一致；
- scaler一致；
- split hash一致；
- source hash一致；
- 不一致时拒绝恢复；
- resume恢复optimizer、epoch、patience和RNG。

E. RNG公平测试

- 相同fold seed下五个variant样本顺序一致；
- 相同fold/epoch下翻转决策一致；
- 重复运行得到相同顺序；
- variant执行顺序不改变单个variant结果；
- 不同fold seed不同且可复现。

F. summary测试

使用合成OOF：

- 五个variant严格ID对齐；
- 500行要求可配置为小型测试数量；
- 概率和为1；
- 重复ID报错；
- fold不一致报错；
- true label不一致报错；
- 指标差值方向正确；
- paired bootstrap使用同一抽样；
- patient group整体抽样；
- 多图患者不会被拆开；
- 固定seed结果一致。

==================================================
二十八、协议测试
==================================================

正式训练前必须使用真实项目元数据执行协议测试，但不需要加载全部图像进入GPU。

至少验证：

- 正式500例队列；
- 115/237/148类别数量；
- 483个patient_group_id；
- 17个多图patient group；
- patient group不跨fold；
- Stage 1/2A/2B ID完全一致；
- 每fold文件完整；
- 每fold 400/100；
- 486/14 availability；
- 三套六维顺序正确；
- G0/G-Mask/G-Raw/G-A/G-B输入维度正确；
- OOF文件未被用作train source；
- camera/EXIF未进入输入；
- scaler只fit train；
- config与锁定协议一致；
- 输出目录不会覆盖历史实验。

协议测试失败时不得启动正式训练。

==================================================
二十九、正式执行顺序
==================================================

本地Codex执行：

1. 检查git状态；
2. 阅读现有实现；
3. 新增代码；
4. 静态检查；
5. 运行新增单元测试；
6. 运行相关Stage 1/2A/2B既有测试；
7. 运行protocol-only；
8. 运行CPU smoke-test；
9. 验证checkpoint恢复；
10. 生成implementation report；
11. 输出服务器正式命令；
12. 不运行完整25次训练。

服务器正式执行顺序：

1. protocol-only；
2. 单元/协议测试；
3. global_only fold 0–4；
4. global_mask fold 0–4；
5. global_raw fold 0–4；
6. global_stage2a fold 0–4；
7. global_stage2b fold 0–4；
8. summarize-only；
9. 拼接五套OOF；
10. 生成逐折比较；
11. 执行paired cluster bootstrap；
12. 生成正式实验报告；
13. 生成run manifest；
14. 验收所有输出。

推荐服务器命令形式：

python scripts/run/run_global_optical_fusion_5fold.py \
  --config config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml \
  --variant all \
  --fold all

Windows终端可以写成单行。

==================================================
三十、正式报告
==================================================

完整服务器训练后生成：

reports/global_resnet18_optical_fusion/
global_resnet18_optical_fusion_report.md

至少包含：

1. 完成状态；
2. 数据和队列；
3. 类别分布；
4. 五折划分；
5. Global meanbg图像来源；
6. Stage 1/2A/2B来源；
7. 为什么2A/2B使用逐折文件；
8. 为什么不使用500行OOF作为train；
9. 五个variant定义；
10. 模型结构；
11. 参数量；
12. 六维字段顺序；
13. 标准化；
14. 缺失处理；
15. availability控制组；
16. 训练配置；
17. loss和类别权重；
18. 图像增强；
19. checkpoint选择；
20. outer val用于early stopping的限制；
21. 每折best epoch；
22. 每折训练曲线；
23. 每折指标；
24. 五折mean±std；
25. pooled OOF指标；
26. G-Mask vs G0；
27. G-Raw/G-A/G-B vs G-Mask；
28. G-A vs G-Raw；
29. G-B vs G-Raw；
30. G-B vs G-A；
31. 更优fold数量；
32. paired patient-group bootstrap 95% CI；
33. 三类AUC和Recall变化；
34. confusion matrix；
35. 六维train/val分布；
36. Stage 2B in-sample/out-of-sample偏移；
37. 是否存在过拟合；
38. 是否存在某类性能牺牲；
39. OOF完整性；
40. 测试结果；
41. 确定性与seed；
42. 历史数据未修改声明；
43. 局限性；
44. 最终候选建议。

报告不得只展示最佳variant。

必须同时报告：

- 所有五个variant；
- 所有五折；
- 正向和负向变化；
- 折间不一致；
- 置信区间；
- Stage 2B可能的分布偏移；
- outer val选模带来的乐观性。

==================================================
三十一、结果解释规则
==================================================

不得使用单一指标自动宣布胜负。

A. 如果G-Mask优于G0

说明availability或额部质量状态本身可能提供分类信息。

不能把G-Raw/G-A/G-B相对G0的全部提升归因于六维表型。

B. 如果G-Raw优于G-Mask

说明Raw六维光学表型包含Global图像之外的增量信息。

C. 如果G-A优于G-Raw

可以谨慎说明：

线性采集条件校准可能提高了光学表型与Global视觉特征的互补性。

不能声称完全去除了设备影响。

D. 如果G-B优于G-A

只有同时满足以下条件，才支持Stage 2B作为主候选：

- pooled OOF Macro-AUC更高；
- 五折平均Macro-AUC更高；
- 至少3/5折方向一致；
- Macro-F1和Balanced Accuracy没有明显下降；
- 三类Recall没有不可接受的牺牲；
- bootstrap结果支持方向稳定；
- 提升不是只来自一个异常fold。

E. 如果G-A与G-B接近

优先Stage 2A，因为：

- Ridge更简单；
- 可解释性更好；
- Stage 2B存在折外泛化和分布偏移风险；
- Stage 2B此前没有整体降低设备差异。

F. 如果G-Raw最好

可能说明：

- 校准删除了与NYHA有关的个体光学信息；
- 或校准分布偏移削弱了分类；
- 不得直接解释为“物理思路完全无效”。

G. 如果三套六维均不优于G-Mask

说明六维表型可能与ResNet18 Global特征冗余。

H. 如果只有G-Mask提升

说明增益更可能来自额部可用性或图像质量信息，而不是六维光学表型。

==================================================
三十二、run manifest
==================================================

生成：

experiments/global_resnet18_optical_fusion/
summary/run_manifest.json

至少记录：

- task；
- status；
- variants；
- completed_variants；
- completed_folds；
- model architecture；
- backbone；
- pretrained weights；
- feature dimensions；
- parameter counts；
- training config；
- class mapping；
- class counts；
- split SHA256；
- Stage 1 SHA256；
- Stage 2A manifest/schema SHA256；
- Stage 2B manifest/schema SHA256；
- 每fold特征源SHA256；
- 每foldtrain/val ID SHA256；
- 每foldscaler SHA256；
- 每foldbest checkpoint SHA256；
- 每variant OOF SHA256；
- config SHA256；
- code SHA256；
- seed规则；
- software versions；
- device；
- CUDA版本；
- git commit；
- tests；
- protocol status；
- smoke-test status；
- full_training_executed；
- outer_validation_tuning=true；
- camera_used=false；
- exif_used=false；
- clinical_features_used=false；
- oof_used_as_classifier_train=false；
- stage1_modified=false；
- stage2a_modified=false；
- stage2b_modified=false；
- split_modified=false；
- labels_modified=false；
- historical_inputs_modified=false。

==================================================
三十三、实现边界
==================================================

本任务禁止：

- 修改历史Global代码；
- 覆盖用户未提交修改；
- 修改Stage 1；
- 修改Stage 2A；
- 修改Stage 2B；
- 重跑Stage 1/2A/2B；
- 修改五折split；
- 修改标签；
- 修改图像；
- 使用522例历史队列；
- 使用strict-blackbg替代meanbg；
- 使用历史0.7025直接充当G0；
- 使用500例2A/2B OOF作为分类train；
- 使用camera或EXIF；
- 使用predicted condition或residual；
- 使用Raw+2A+2B同时拼接；
- 加入临床特征；
- 加入性别作为模型输入；
- 加入新ROI；
- 加入额外MLP；
- 加入attention/gating/FiLM；
- 搜索学习率；
- 搜索batch size；
- 搜索epoch；
- 搜索patience；
- 搜索loss；
- 搜索seed；
- 运行多个seed后选最优；
- 给某个variant单独调参；
- 安装或升级依赖；
- 在CPU环境误启动完整25次训练；
- 根据smoke-test结果选择模型；
- 伪造未完成的正式结果；
- 进行无关重构。

==================================================
三十四、本地实现报告
==================================================

本地代码实现完成后生成：

reports/global_resnet18_optical_fusion/
global_resnet18_optical_fusion_implementation_report.md

至少包括：

1. 实现状态；
2. 新增文件；
3. 是否修改历史文件；
4. 读取的现有基线；
5. 五个variant；
6. 模型结构；
7. Dataset数据流；
8. 六维字段；
9. 逐折2A/2B文件选择；
10. scaler实现；
11. 缺失处理；
12. checkpoint；
13. Trainer；
14. Evaluator；
15. OOF和summary；
16. paired bootstrap；
17. RNG公平；
18. preflight结果；
19. 单元测试结果；
20. 既有Stage测试结果；
21. 协议测试结果；
22. smoke-test结果；
23. 当前CUDA状态；
24. 是否运行正式训练；
25. 未完成事项；
26. 服务器正式命令；
27. 验收条件；
28. 历史输入未修改声明。

本地只完成代码和测试时，状态应为：

READY_FOR_SERVER_TRAINING

不得写：

EXPERIMENT_COMPLETE

==================================================
三十五、最终终端输出
==================================================

本地实现结束时打印：

- GLOBAL_OPTICAL_FUSION_IMPLEMENTATION_STATUS；
- 实际project root；
- git branch；
- git commit；
- 当前CUDA是否可用；
- 新增文件；
- 修改的历史文件数量；
- 五个variant是否实现；
- 模型维度测试；
- Dataset测试；
- scaler测试；
- checkpoint测试；
- RNG公平测试；
- summary测试；
- Stage 1/2A/2B相关既有测试；
- protocol-only状态；
- smoke-test状态；
- 是否启动完整训练；
- 是否修改Stage 1；
- 是否修改Stage 2A；
- 是否修改Stage 2B；
- 是否修改split；
- 是否修改标签；
- 是否修改图像；
- implementation report路径；
- protocol manifest路径；
- 服务器正式训练命令；
- 是否满足READY_FOR_SERVER_TRAINING。

如果实现未完成：

- 不得声称READY；
- 明确停止步骤；
- 输出真实错误；
- 保留日志；
- 不覆盖用户文件；
- 给出准确恢复方式；
- 不估算或伪造测试结果。

服务器完整训练结束时打印：

- GLOBAL_OPTICAL_FUSION_EXPERIMENT_STATUS；
- 完成variant数量；
- 完成fold数量；
- 每variant OOF行数和唯一ID数；
- 每variant pooled OOF Macro-AUC；
- 每variant五折平均Macro-AUC；
- 每variant Accuracy；
- 每variant Balanced Accuracy；
- 每variant Macro-F1；
- G-Mask vs G0差值；
- G-Raw vs G-Mask差值；
- G-A vs G-Mask差值；
- G-B vs G-Mask差值；
- G-A vs G-Raw差值；
- G-B vs G-Raw差值；
- G-B vs G-A差值；
- 每个比较更优fold数量；
- bootstrap有效重复次数；
- Stage 2B分布偏移摘要；
- 测试状态；
- OOF路径；
- pairwise comparison路径；
- bootstrap路径；
- 正式报告路径；
- run manifest路径；
- 是否满足全部验收条件。

只有五个variant、25个fold run、五套OOF、配对比较和正式报告全部完成后，才允许：

GLOBAL_OPTICAL_FUSION_EXPERIMENT_STATUS=COMPLETE