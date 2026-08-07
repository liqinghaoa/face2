你正在 E:\projects\face2 项目中继续执行普通手机人脸照片心功能二分类研究。

本轮任务名称：

Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1

本轮一次性依次完成两个核心子实验：

C0-A：
Xiaomi-only SH093 224×224 Control vs Patient严格patient-group嵌套五折实验。

C0-B：
Patient-only SH093 224×224 Xiaomi vs HONOR相机相关域可预测性实验。

可选但高优先级控制臂：

C0-A-Control：
与SH093 224×224输入几何和背景严格匹配的原始RGB 224×224 Xiaomi-only二分类实验。

不得进入：

- 多SH训练；
- 跨SH测试；
- Exposure/Gamma鲁棒训练；
- R3DPR一致性训练；
- Stage3-B1；
- ROI实验；
- 模型结构搜索；
- 超参数搜索。

======================================================================
一、研究背景
======================================================================

现有完整500例SH093重光照二分类历史结果为：

macro_auc = 0.8281
accuracy = 0.7820
macro_precision = 0.7168
macro_recall = 0.7731
macro_f1 = 0.7314
balanced_accuracy = 0.7731

该历史实验实际使用的输入目录为：

E:/projects/face2/data/processed/global_face/fixedSH_origcam_menafg

其上游R3DPR原始SH093输出目录为：

E:/projects/face2/data/processed/P2_ExampleSH093_FixedLight_v1/images/fixedSH_origcam

R3DPR实际流程为：

原始手机RGB
→ CropPose检测、裁剪和原视角估计
→ DPR估计原始SH
→ 主编码得到planes和superres_ws
→ 保留每例CropPose原视角
→ 将显式光照替换为固定SH093
→ 生成fixedSH_origcam
→ 后续处理为224×224 fixedSH_origcam_menafg

因此，SH093统一的是最终生成阶段的显式球谐光照，不代表：

- 已完全消除原始手机ISP；
- 已完全消除相机相关生成偏差；
- 已获得真实相机无关反射率；
- 已完全排除所有光照残留；
- 已证明分类信息属于医学表型。

Stage3-B0已经得到：

Xiaomi-only原始RGB：
pooled OOF AUC = 0.4964
95% CI = 0.4237–0.5724
Balanced Accuracy = 0.5099
Macro-F1 = 0.5081

说明在固定Xiaomi相机后，原始RGB没有表现出稳定同相机区分信号。

本轮需要回答：

1. 固定为Xiaomi后，SH093生成式统一光照表征是否能恢复稳定的Control–Patient区分信号？
2. SH093 224×224输入是否仍保留能够预测原手机设备域的信息？
3. 完整500例SH093 AUC 0.8281更可能来自同相机疾病相关表征，还是来自设备/采集域残留？
4. SH093是否具备成为后续主要二分类输入的资格？

======================================================================
二、核心实验定义
======================================================================

C0-A：

输入：
SH093 224×224图像。

队列：
Xiaomi M2006J10C全部233例。

标签：
0 = Control
1 = Patient

目标：
判断SH093在固定手机型号后是否具有稳定疾病区分能力。

C0-B：

输入：
同一套SH093 224×224图像。

队列：
仅Patient，预计385例。

标签：
0 = Xiaomi M2006J10C
1 = HONOR BVL-AN00

目标：
判断在全部样本均为Patient后，SH093图像是否仍具有相机相关域可预测性。

两个实验共享：

- 图像资产审计；
- Dataset基础实现；
- 图像读取和标准化；
- patient-group防泄漏规则；
- 指标实现；
- patient-group bootstrap；
- 运行环境；
- 输出完整性审计。

两个实验必须独立：

- 标签；
- split；
- checkpoint；
- OOF；
- 模型；
- 训练日志；
- 统计结论。

禁止联合训练或多任务训练。

======================================================================
三、运行环境
======================================================================

必须使用：

E:\resarch\Anaconda3\envs\face2\python.exe

记录：

- Python版本；
- PyTorch版本；
- TorchVision版本；
- CUDA版本；
- cuDNN版本；
- GPU型号；
- NumPy版本；
- Pillow版本；
- OpenCV版本；
- scikit-learn版本；
- 当前Git commit；
- git status；
- 工作目录；
- 执行命令。

输出：

logs/runtime_environment.txt
logs/git_status.txt
logs/execution_commands.txt

不得使用base环境或系统Python。

======================================================================
四、输入资产
======================================================================

4.1 SH093 224×224正式输入

必须使用用户指定的现有目录，路径字符串按字面使用：

E:/projects/face2/data/processed/global_face/fixedSH_origcam_menafg

不得改用：

- 正式六SH目录中的512×512 relight_neutral_front.png；
- fixedSH_fixedcam；
- 原始512×512 fixedSH_origcam；
- 重新resize得到的新目录；
- 其他SH编号。

4.2 R3DPR上游原图

仅用于资产链审计，不作为本轮分类输入：

E:/projects/face2/data/processed/P2_ExampleSH093_FixedLight_v1/images/fixedSH_origcam

4.3 Stage3-B0冻结输出

实验根目录：

E:/projects/face2/experiments/lighting_confounding/Stage3_B0_XiaomiOnly_ResNet18_ControlVsPatient_Group5Fold_v1

重点读取：

reports/stage3_b0_report.md
reports/stage3_b0_machine_summary.json
oof/xiaomi_b0_oof_predictions.csv
splits/xiaomi_control_patient_group5fold_v1.csv
splits/inner/
training_contract/
fold_0/ 到 fold_4/

C0-A必须复用B0：

- 完全相同的233例；
- 完全相同的outer fold；
- 完全相同的inner train/validation划分；
- 完全相同的patient_group定义。

4.4 相机和标签信息

优先从Stage1冻结主表读取：

E:/projects/face2/experiments/lighting_confounding/Lighting_Confounding_Audit_Stage1_v1/metadata/stage1_master_500.csv

必要时结合：

E:/projects/face2/data/raw/EXIF/Image_Metadata_All.xlsx

4.5 旧SH093完整500例实验

Codex必须搜索项目中：

- 对fixedSH_origcam_menafg的代码引用；
- 训练配置；
- 实验目录；
- OOF；
- 五折split；
- 结果报告；
- 得到macro_auc 0.8281的具体模型。

使用命令行或代码检索：

fixedSH_origcam_menafg
0.8281
fixedSH_origcam
SH093

必须输出旧实验定位报告。

旧实验只作为历史审计，不得复用旧checkpoint作为C0-A正式结果。

4.6 本轮输出目录

E:/projects/face2/experiments/lighting_confounding/Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1

不得覆盖任何已有实验。

======================================================================
五、SH093 224×224资产与处理契约审计
======================================================================

在训练前必须定位：

- fixedSH_origcam_menafg的生成脚本；
- 配置文件；
- 输入目录；
- 输出目录；
- 人脸检测方式；
- 对齐方式；
- crop方式；
- resize方式；
- 插值方法；
- mask方式；
- 背景定义；
- “menafg”在代码中的实际含义；
- 图像格式；
- 图像色彩空间；
- 是否保存PNG；
- 是否进行ImageNet标准化；
- 是否使用标签、NYHA、fold或预测结果参与像素处理。

不得仅根据目录名推断。

必须验证：

- 500个预期病例是否均存在；
- 文件名与sample_id是否一一对应；
- 图像是否严格224×224；
- 通道是否RGB；
- dtype是否uint8；
- 文件是否可解码；
- 图像是否finite；
- 是否存在重复文件hash；
- 是否存在全黑图；
- 是否存在异常纯色图；
- 是否存在严重截断；
- 背景是否符合预处理契约；
- C0-A的233例是否全部存在；
- C0-B的Patient样本是否全部存在。

输出：

shared_asset_audit/sh093_224_asset_inventory.csv
shared_asset_audit/sh093_224_asset_contract.json
shared_asset_audit/sh093_224_asset_audit.json
shared_asset_audit/sh093_224_asset_report.md
shared_asset_audit/sh093_224_hash_inventory.csv
shared_asset_audit/sh093_224_failures.csv
shared_asset_audit/sh093_224_qc_panel.png

若C0-A任一233例缺图，停止整个任务。

若C0-B存在缺图，停止C0-B，不影响已经通过并完成的C0-A，但必须明确报告。

======================================================================
六、匹配原始RGB 224×224控制资产审计
======================================================================

本控制臂用于区分：

- SH093重光照效应；
- 224×224几何、背景和预处理效应。

Codex必须在以下位置检索已有资产和生成代码：

E:/projects/face2/data/processed/global_face
E:/projects/face2/preprocessing
E:/projects/face2/src
E:/projects/face2/scripts
E:/projects/face2/experiments

寻找与fixedSH_origcam_menafg满足以下条件的原始RGB输入：

1. 同样224×224；
2. 同样的人脸几何定义；
3. 同样的对齐或裁剪方式；
4. 同样的mask定义；
5. 同样的背景定义；
6. 仅输入源不同：
   - 一个来自原始RGB；
   - 一个来自fixedSH_origcam。

禁止仅因为都是224×224就认定为匹配。

若找到唯一可信资产：

- 将其冻结为C0-A-Control输入；
- 记录路径、代码、配置和hash；
- 使用与C0-A完全相同的split和训练协议。

若找到多个候选且不能唯一判断：

- 不自行选择；
- 输出候选比较表；
- 将matched_control_status标记为ambiguous；
- 不运行控制臂。

若没有精确匹配资产：

- matched_control_status = unavailable；
- 不在本轮临时生成新数据；
- C0-A与C0-B继续正常执行；
- 原始RGB Stage3-B0仅作描述性历史参照。

输出：

matched_control/preflight/original_rgb_224_candidate_inventory.csv
matched_control/preflight/original_rgb_224_contract_comparison.csv
matched_control/preflight/matched_control_decision.json
matched_control/preflight/matched_control_report.md

======================================================================
七、旧SH093 0.8281实验审计
======================================================================

需要回答：

- 旧实验代码位置；
- 实验输出目录；
- 输入是否确为fixedSH_origcam_menafg；
- 样本数是否500；
- 标签定义；
- split文件；
- split是否patient-group安全；
- 同一patient不同visit是否跨fold；
- 是否使用outer test选epoch；
- 是否存在独立inner validation；
- 模型结构；
- 输入尺寸；
- train/val transform；
- loss；
- optimizer；
- learning rate；
- batch size；
- max epoch；
- early stopping；
- checkpoint选择方式；
- OOF是否完整；
- macro_auc 0.8281如何计算；
- 是否为pooled OOF AUC；
- 是否为五折平均；
- 是否有测试数据泄漏；
- 是否根据分类结果选择SH093。

SH093是依据视觉质量标准预先选择，不是依据分类AUC选择。报告中应如实记录。

输出：

legacy_audit/legacy_sh093_experiment_inventory.csv
legacy_audit/legacy_sh093_training_contract.json
legacy_audit/legacy_sh093_split_audit.json
legacy_audit/legacy_sh093_result_reproduction_audit.json
legacy_audit/legacy_sh093_report.md

本轮不要求重新训练完整500例旧实验。

若无法唯一定位旧实验，记录为not_uniquely_identified，不影响C0-A和C0-B执行。

======================================================================
八、C0-A Xiaomi-only SH093疾病分类
======================================================================

8.1 队列

严格复用Stage3-B0队列：

- 总计233例；
- Control 115；
- Patient 118；
- patient groups 232；
- camera model全部为M2006J10C。

必须逐行核对：

- sample_id；
- patient_group_id；
- binary_label；
- outer_fold；
- image_path；
- SH093图像是否存在。

输出：

c0a_disease/cohort/c0a_xiaomi_sh093_cohort.csv
c0a_disease/cohort/c0a_cohort_audit.json

8.2 Split

必须直接读取并复用：

Stage3-B0外层split；
Stage3-B0每折inner split。

不得：

- 重新生成；
- 更换seed；
- 为SH093优化fold；
- 调整类别分布；
- 删除难例；
- 根据SH093质量筛选病例。

生成复用证明：

c0a_disease/splits/c0a_reused_outer_split.csv
c0a_disease/splits/c0a_reused_inner_split_inventory.csv
c0a_disease/splits/c0a_split_identity_audit.json

要求：

- 外层split与B0逐行一致；
- inner split与B0逐行一致；
- hash一致或内容逐字段一致；
- patient_group无泄漏。

8.3 模型训练契约

优先复用旧SH093 0.8281实验的224×224训练契约，因为输入数据定义相同。

必须保留：

- ResNet具体版本；
- ImageNet初始化；
- 分类头；
- dropout；
- 224×224输入；
- RGB顺序；
- ImageNet normalization；
- train transform；
- val transform；
- optimizer；
- scheduler；
- learning rate；
- weight decay；
- batch size；
- max epoch；
- patience；
- BN策略；
- AMP策略。

但必须采用严格nested评价：

inner validation选择checkpoint；
outer test只在checkpoint冻结后推理一次。

若旧SH093契约无法唯一定位，则使用Stage3-B0的ResNet18基础契约，并只将输入尺寸调整为224×224：

- ResNet18 ImageNet初始化；
- dropout 0.3；
- ImageNet normalization；
- train仅水平翻转；
- AdamW；
- lr=1e-4；
- weight_decay=1e-4；
- batch_size=16；
- max_epoch=30；
- patience=5；
- BN frozen eval；
- AMP=false；
- BCEWithLogitsLoss(pos_weight=1.0)。

必须在training_contract中记录使用了：

legacy_sh093_contract
或
stage3_b0_fallback_contract

不得同时尝试两套契约后选择结果更好的版本。

8.4 禁止项

C0-A禁止：

- 使用HONOR；
- 使用EXIF；
- 使用camera label；
- 使用原始RGB双分支；
- 使用R3DPR中间planes；
- 使用original_camera.npy；
- 使用original_sh；
- 使用其他SH；
- 使用Exposure/Gamma增强；
- 使用颜色归一化搜索；
- 使用类别重采样；
- 使用测试fold选epoch；
- 使用旧完整500例checkpoint；
- 调整阈值。

分类阈值固定0.5。

8.5 checkpoint选择

每个outer fold：

inner train训练；
inner validation选择best checkpoint；
outer test在best checkpoint冻结后只推理一次。

checkpoint主选择指标：

inner validation ROC-AUC。

tie-break：

1. 更高inner validation AUC；
2. 更低inner validation loss；
3. 更早epoch。

保存：

c0a_disease/fold_k/checkpoints/best_auc.pth
c0a_disease/fold_k/checkpoints/last.pth
c0a_disease/fold_k/history/training_history.csv
c0a_disease/fold_k/predictions/outer_test_predictions.csv

8.6 OOF

合并：

c0a_disease/oof/c0a_xiaomi_sh093_oof_predictions.csv

必须满足：

- 233行；
- 233个唯一sample_id；
- 每例仅一次outer test预测；
- fold与B0一致；
- 无缺失；
- probability和logit finite。

字段至少包括：

sample_id
patient_group_id
outer_fold
binary_label
logit_patient
probability_patient
prediction_05
correct
selected_epoch
checkpoint_sha256

8.7 指标

主要终点：

pooled OOF ROC-AUC。

同时计算：

- patient-group cluster bootstrap 95% CI；
- Accuracy；
- Balanced Accuracy；
- Macro-Precision；
- Macro-Recall；
- Macro-F1；
- Sensitivity；
- Specificity；
- Brier score；
- calibration intercept；
- calibration slope；
- confusion matrix；
- 每折AUC；
- 每折BA；
- 每折Sensitivity；
- 每折Specificity；
- 每折Macro-F1。

bootstrap：

iterations=2000
seed=2026
抽样单位=patient_group_id

8.8 与Stage3-B0原始RGB比较

使用完全相同的233例进行配对比较：

SH093-224 C0-A
vs
Original RGB Stage3-B0 256×320

计算：

delta_auc
delta_balanced_accuracy
delta_macro_f1
delta_sensitivity
delta_specificity
delta_brier

使用patient-group cluster paired bootstrap。

必须明确：

该比较同时包含重光照和输入预处理差异，只能作为配对描述性/综合管线比较，不能把全部差异归因于SH093。

8.9 与匹配Original-RGB-224控制比较

仅当matched_control_status=available时运行。

使用：

- 相同233例；
- 相同outer split；
- 相同inner split；
- 相同模型和训练契约；
- 相同随机种子；
- 相同指标；
- 相同bootstrap。

此比较才是主要的预处理匹配对照：

AUC_SH093_224 - AUC_OriginalRGB_224

若控制臂不可用，不得伪造或替代。

8.10 C0-A信号判定

输出：

c0a_same_camera_signal_level:
- none
- weak
- moderate
- strong
- indeterminate

建议判定：

none：
- pooled AUC接近0.5；
- CI跨0.5；
- 少于3/5折AUC>0.5；
- BA接近0.5；
- 存在预测塌缩。

weak：
- AUC点估计约0.55–0.62；
- CI跨0.5或折间不稳定；
- 未明确优于控制。

moderate：
- AUC CI下界高于0.5；
- 至少4/5折AUC>0.5；
- BA>0.55；
- 无单侧类别塌缩；
- 相对B0有明确配对提升。

strong：
- 在moderate基础上；
- AUC较高且折间稳定；
- 明显优于匹配Original-RGB-224；
- 不由单一fold驱动。

这些是项目内部证据等级，不是临床标准。

======================================================================
九、C0-B Patient-only SH093相机相关域分类
======================================================================

9.1 主要队列

只使用Patient。

预计：

Xiaomi Patient = 118
HONOR Patient = 267
总计 = 385

禁止加入Control。

原因：

完整500例中Control全部为Xiaomi，若直接做相机分类，会使疾病标签和相机标签直接混杂。

标签定义：

camera_label = 0：
Xiaomi M2006J10C

camera_label = 1：
HONOR BVL-AN00

必须核对实际标准化相机字符串。

输出：

c0b_camera/cohort/c0b_patient_only_camera_cohort.csv
c0b_camera/cohort/c0b_camera_cohort_audit.json
c0b_camera/cohort/c0b_camera_cohort_report.md

若数量不是118、267、385，停止C0-B并输出差异。

9.2 patient-group外层五折

为C0-B新建专用五折。

固定：

n_splits=5
shuffle=true
seed=2026

优先级：

1. patient_group_id不得跨fold；
2. camera_label分层；
3. 尽量平衡NYHA原始等级或轻重症组；
4. 尽量平衡sex；
5. 尽量平衡age；
6. 各折样本数接近。

每折必须同时包含Xiaomi Patient和HONOR Patient。

输出：

c0b_camera/splits/c0b_patient_camera_group5fold.csv
c0b_camera/splits/c0b_outer_split_audit.csv
c0b_camera/splits/c0b_outer_split_report.md

9.3 inner validation

在每个outer development中建立固定inner split。

建议：

StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=12026 + outer_fold
)

取第一个split为inner validation。

要求：

- outer test不进入inner split；
- patient_group不跨inner train/val；
- inner train和val均包含两种相机；
- 每折确定性可复现。

9.4 模型协议

使用与C0-A相同的224×224 ResNet18基础协议。

唯一变化：

预测标签由疾病变为相机域。

由于类别不平衡，使用：

BCEWithLogitsLoss(pos_weight=n_xiaomi_train/n_honor_train的正确二分类定义)

注意：

若HONOR定义为正类1，则：

pos_weight =
N_negative_Xiaomi / N_positive_HONOR

必须仅根据当前inner train计算。

不得使用完整385例提前计算权重。

不得使用WeightedRandomSampler。

9.5 主要指标

主要终点：

pooled Patient-only camera ROC-AUC。

同时计算：

- Balanced Accuracy；
- Macro-F1；
- Xiaomi recall；
- HONOR recall；
- Accuracy；
- Brier；
- calibration；
- confusion matrix；
- patient-group bootstrap 95% CI；
- 每折AUC；
- 每折BA；
- 每折两类recall。

不能只报告Accuracy。

9.6 相机域证据等级

输出：

camera_domain_predictability_level:
- none
- weak
- moderate
- strong
- indeterminate

建议：

none：
- AUC接近0.5；
- CI跨0.5；
- 折间不稳定。

weak：
- AUC约0.55–0.65；
- CI或折间证据有限。

moderate：
- AUC约0.65–0.80；
- CI下界高于0.5；
- 至少4/5折AUC>0.5。

strong：
- AUC≥0.80；
- CI下界高于0.5；
- 五折稳定；
- 两类recall均不过度塌缩。

不得称为“纯手机传感器指纹”。

正式表述：

camera-associated domain predictability
相机相关域可预测性

因为仍可能包含：

- NYHA分布差异；
- 年龄差异；
- 性别差异；
- 采集时间差异；
- 操作者差异；
- 病房或环境差异；
- 生成质量差异。

======================================================================
十、C0-B人群协变量基线
======================================================================

建立仅使用非图像变量的相机预测基线。

候选变量：

- age；
- sex；
- original NYHA grade；
- mild/severe NYHA grouping；
- 其他明确可用且非图像的基础人口学变量。

禁止使用：

- 图像颜色；
- EXIF；
- BrightnessValue；
- 相机字段本身；
- sample_id；
- 时间戳，除非单独作为采集流程敏感性分析。

使用与C0-B相同outer folds。

每fold：

- 仅在outer development拟合；
- 数值缺失用训练折中位数；
- 分类缺失用训练折众数或显式missing；
- 标准化只在训练折拟合；
- LogisticRegression C=1.0；
- 不调参。

输出：

c0b_camera/baselines/covariate_baseline_oof.csv
c0b_camera/baselines/covariate_baseline_metrics.csv

作用：

判断相机分类能力是否可能仅由Patient人群组成差异解释。

======================================================================
十一、C0-B低维图像域基线
======================================================================

从SH093 224×224图计算低维特征，仅用于相机域审计：

- mean_R/G/B；
- std_R/G/B；
- mean_Lab_L/a/b；
- skin或有效人脸区域亮度分位数，如已有可靠mask；
- p01、p05、p50、p95、p99；
- dark pixel fraction；
- bright pixel fraction；
- saturation fraction；
- black background fraction；
- Laplacian variance；
- high-frequency energy；
- edge density；
- 有效非黑区域面积比例；
- 水平和垂直边界黑边比例。

若没有与SH093 224严格匹配的可靠人脸mask，只使用全图和非黑区域统计，不得误用原始RGB mask。

使用与C0-B相同outer folds：

- 训练折中位数填补；
- 训练折标准化；
- LogisticRegression C=1.0；
- 不调参；
- outer test cross-fitting。

输出：

c0b_camera/baselines/lowlevel_feature_table.csv
c0b_camera/baselines/lowlevel_oof_predictions.csv
c0b_camera/baselines/lowlevel_metrics.csv
c0b_camera/baselines/lowlevel_coefficients.csv

若低维特征已经能高AUC预测相机，说明SH093管线中存在明显低层域差异。

======================================================================
十二、C0-B 1:1匹配敏感性分析
======================================================================

目的：

减少Xiaomi Patient和HONOR Patient在人群构成上的差异。

匹配必须在patient_group层面完成。

优先变量：

1. sex精确匹配；
2. NYHA原始等级或预定义轻/重症组精确匹配；
3. age最近邻匹配，如age可用；
4. 不使用图像或模型输出。

建议方法：

- 每个Xiaomi Patient group匹配一个HONOR Patient group；
- 无放回；
- 固定seed=2026；
- age使用标准化绝对距离；
- age caliper=0.20 pooled SD；
- 若无法全部匹配，记录匹配成功率；
- 不为达到118对而放宽到不合理距离。

如果age缺失率较高：

- 主匹配仅使用sex + NYHA；
- age平衡作为审计；
- 不静默填补后强行匹配。

形成：

约118 Xiaomi Patient groups
vs
约118 HONOR Patient groups

或实际可匹配数量。

必须报告标准化差异：

- age SMD；
- sex SMD；
- NYHA分布差异；
- 匹配前后对比。

匹配集仍使用patient-group五折，不能在完整数据上训练后只评估匹配子集。

为控制计算量，本轮允许：

方案A，优先：
在冻结匹配队列上重新执行独立nested五折相机分类。

禁止：
使用主C0-B模型预测后再挑选匹配效果最好的子集。

输出：

c0b_camera/matched/matching_contract.json
c0b_camera/matched/matched_group_pairs.csv
c0b_camera/matched/matching_balance.csv
c0b_camera/matched/matched_split.csv
c0b_camera/matched/matched_oof_predictions.csv
c0b_camera/matched/matched_metrics.csv
c0b_camera/matched/matched_report.md

若匹配样本太少，例如每类少于60个patient groups：

- 不运行深度模型匹配五折；
- 只输出匹配可行性和低维/协变量分析；
- 标记matched_deep_model_status=insufficient_sample。

======================================================================
十三、联合解释
======================================================================

生成联合矩阵：

A. C0-A疾病信号none/weak，
C0-B相机域moderate/strong：

结论：
完整500例SH093高性能高度疑似由设备、采集流程或生成域捷径驱动。
SH093不适合作为主要无偏疾病分类输入。

B. C0-A疾病信号moderate/strong，
C0-B相机域moderate/strong：

结论：
SH093在Xiaomi同相机内可能保留分类信号，但同时仍保留明显设备域信息。
完整500例SH093结果不能单独作为主要医学证据。
后续必须以Xiaomi-only结果为主，并继续多SH和生成质量审计。

C. C0-A疾病信号moderate/strong，
C0-B相机域none/weak：

结论：
这是最有利结果。
支持SH093在降低设备域可预测性的同时保留同相机区分信号。
仍不能称为纯医学信号，但具备成为主要方法候选的资格。

D. C0-A疾病信号none/weak，
C0-B相机域none/weak：

结论：
SH093可能削弱设备域信息，但未保留稳定疾病分类信号。
不适合作为主要二分类输入。

还要结合：

- C0-B匹配相机AUC；
- 人群协变量基线；
- 低维图像域基线；
- 匹配Original-RGB-224控制；
- 五折稳定性；
- 置信区间；
- 类别塌缩情况。

======================================================================
十四、主要决策字段
======================================================================

输出：

sh093_main_binary_candidate:
- true
- false
- conditional

c0a_same_camera_signal_level:
- none
- weak
- moderate
- strong
- indeterminate

c0b_camera_domain_predictability_level:
- none
- weak
- moderate
- strong
- indeterminate

full500_sh093_interpretation:
- likely_device_or_collection_shortcut
- mixed_same_camera_signal_and_domain_shortcut
- potentially_relighting_supported_signal
- no_reliable_signal
- indeterminate

next_stage_recommendation:
- stop_sh093_mainline
- proceed_multi_sh_stability_audit
- proceed_fixedcam_vs_origcam_audit
- proceed_both_audits
- collect_balanced_device_data
- indeterminate

SH093成为主要方法候选至少需要：

1. C0-A AUC CI下界高于0.5；
2. 至少4/5折AUC>0.5；
3. BA>0.55；
4. 无明显预测塌缩；
5. 相对B0有明确配对提升；
6. 若匹配Original-RGB-224存在，应明显优于该控制；
7. C0-B相机域可预测性不能为strong，或匹配后需明显下降；
8. 低维域特征不能完全解释C0-B结果；
9. 不能由单一fold驱动。

======================================================================
十五、统计方法
======================================================================

所有OOF主要指标均基于独立outer test预测。

Bootstrap：

- patient_group cluster bootstrap；
- iterations=2000；
- seed=2026；
- 95% percentile CI；
- 记录无效bootstrap次数。

配对比较：

- 相同病例；
- 相同patient_group抽样；
- 每次bootstrap同时计算两个模型指标；
- 输出delta及95% CI。

不需要进行大量p值检验。

若进行多个次要比较：

- 明确标记exploratory；
- 使用Benjamini–Hochberg FDR；
- 不用次要p值推翻主要OOF结论。

======================================================================
十六、过拟合和稳定性审计
======================================================================

C0-A和C0-B均必须输出：

- 每fold train loss；
- inner val loss；
- train AUC；
- inner val AUC；
- selected epoch；
- outer test AUC；
- train-val AUC gap；
- fold AUC range；
- fold AUC SD；
- AUC>0.5折数；
- prediction positive rate；
- 每类recall；
- 是否预测塌缩。

警示包括：

- train AUC接近1但inner/outer接近0.5；
- inner val明显高于outer test；
- selected epoch极不稳定；
- 某fold全部预测为单一类别；
- pooled结果由单一fold驱动。

过拟合警示不能自动删除fold。

======================================================================
十七、图形
======================================================================

至少生成：

shared_asset_audit/sh093_qc_panel.png

c0a_disease/figures/c0a_roc_curve.png
c0a_disease/figures/c0a_fold_auc.png
c0a_disease/figures/c0a_confusion_matrix.png
c0a_disease/figures/c0a_probability_distribution.png
c0a_disease/figures/c0a_calibration_curve.png
c0a_disease/figures/c0a_training_validation_auc.png
c0a_disease/figures/c0a_training_validation_loss.png
c0a_disease/figures/c0a_vs_b0_paired_comparison.png

若控制臂可用：

matched_control/figures/sh093_vs_original224_auc.png
matched_control/figures/sh093_vs_original224_probability.png

c0b_camera/figures/c0b_camera_roc_curve.png
c0b_camera/figures/c0b_fold_auc.png
c0b_camera/figures/c0b_confusion_matrix.png
c0b_camera/figures/c0b_probability_distribution.png
c0b_camera/figures/c0b_covariate_vs_image_baselines.png
c0b_camera/figures/c0b_lowlevel_feature_importance.png
c0b_camera/figures/c0b_matched_vs_full_auc.png

joint/figures/joint_evidence_matrix.png

图中不得使用“pure medical signal”或“camera removed”等未经证明表述。

======================================================================
十八、输出目录
======================================================================

输出根目录：

E:/projects/face2/experiments/lighting_confounding/Stage3_C0_SH093_224_SameCameraSignal_and_DeviceDomain_Audit_v1

建议结构：

shared_asset_audit/
legacy_audit/
matched_control/
c0a_disease/
  cohort/
  splits/
  training_contract/
  fold_0/
  fold_1/
  fold_2/
  fold_3/
  fold_4/
  oof/
  metrics/
  bootstrap/
  comparison/
  stability/
  figures/
  reports/
c0b_camera/
  cohort/
  splits/
  training_contract/
  fold_0/
  fold_1/
  fold_2/
  fold_3/
  fold_4/
  oof/
  metrics/
  bootstrap/
  baselines/
  matched/
  stability/
  figures/
  reports/
joint/
  figures/
  reports/
logs/
preflight/

必须至少生成：

preflight/stage3_c0_preflight_summary.json
preflight/stage3_c0_preflight_report.md
preflight/input_inventory.csv

c0a_disease/oof/c0a_xiaomi_sh093_oof_predictions.csv
c0a_disease/metrics/c0a_oof_metrics.json
c0a_disease/metrics/c0a_fold_metrics.csv
c0a_disease/bootstrap/c0a_cluster_bootstrap_ci.csv
c0a_disease/comparison/c0a_vs_stage3_b0.csv
c0a_disease/comparison/c0a_vs_stage3_b0_bootstrap.csv
c0a_disease/stability/c0a_stability_audit.csv
c0a_disease/reports/c0a_report.md
c0a_disease/reports/c0a_machine_summary.json

c0b_camera/oof/c0b_patient_camera_oof_predictions.csv
c0b_camera/metrics/c0b_oof_metrics.json
c0b_camera/metrics/c0b_fold_metrics.csv
c0b_camera/bootstrap/c0b_cluster_bootstrap_ci.csv
c0b_camera/baselines/covariate_baseline_metrics.csv
c0b_camera/baselines/lowlevel_metrics.csv
c0b_camera/matched/matched_metrics.csv
c0b_camera/stability/c0b_stability_audit.csv
c0b_camera/reports/c0b_report.md
c0b_camera/reports/c0b_machine_summary.json

joint/reports/stage3_c0_joint_report.md
joint/reports/stage3_c0_joint_machine_summary.json
joint/reports/stage3_c0_final_decision.json
joint/reports/stage3_c0_next_stage_decision.md
joint/reports/stage3_c0_output_inventory.json

logs/run.log
logs/warnings.csv
logs/failures.csv

======================================================================
十九、Preflight停止门控
======================================================================

整个任务启动前必须确认：

- face2环境正确；
- SH093 224目录存在；
- C0-A的233例图像全部存在；
- C0-A样本和B0 split严格一致；
- B0 outer和inner split可读取；
- sample_id映射唯一；
- patient_group完整；
- 图像均为224×224 RGB；
- 输出目录不覆盖原实验；
- 旧实验审计已至少完成检索；
- C0-B Patient相机标签可构建。

停止规则：

1. C0-A资产或split失败：
   停止整个任务，不运行任何训练。

2. C0-A通过但C0-B队列失败：
   完成C0-A，停止C0-B，并生成部分完成报告。

3. 匹配Original-RGB-224不可用：
   不停止C0-A/C0-B，只跳过控制臂。

4. Patient 1:1匹配不足：
   不停止主C0-B，只跳过匹配深度模型。

======================================================================
二十、测试要求
======================================================================

新增测试目录：

tests/stage3_c0_sh093_224

至少测试：

1. SH093输入目录存在；
2. C0-A 233例完整；
3. C0-A Control 115；
4. C0-A Patient 118；
5. C0-A仅Xiaomi；
6. C0-A复用B0 outer split；
7. C0-A复用B0 inner split；
8. patient_group不跨outer fold；
9. patient_group不跨inner train/val；
10. 图像严格224×224；
11. 图像为RGB；
12. 图像可解码；
13. 图像hash无异常重复；
14. C0-A未使用HONOR；
15. C0-A未使用EXIF；
16. C0-A未使用camera label；
17. C0-A未使用其他SH；
18. C0-A未使用旧checkpoint；
19. outer test不参与checkpoint选择；
20. OOF严格233行；
21. 每例恰好一个OOF预测；
22. 分类阈值固定0.5；
23. bootstrap按patient_group；
24. Stage3-B0配对样本一致；
25. C0-B只包含Patient；
26. C0-B Xiaomi Patient预计118；
27. C0-B HONOR Patient预计267；
28. C0-B不包含Control；
29. C0-B每fold包含两台相机；
30. C0-B patient_group不泄漏；
31. C0-B class weight仅从inner train计算；
32. C0-B OOF覆盖全部主队列；
33. 协变量基线外层cross-fit；
34. 低维基线外层cross-fit；
35. 匹配不使用图像特征；
36. 匹配不使用模型预测；
37. 匹配按patient_group；
38. 未运行多SH；
39. 未运行fixedcam实验；
40. 未运行Exposure/Gamma训练；
41. 未进入Stage3-B1；
42. 输出报告完整；
43. 机器摘要完整；
44. 决策字段完整。

运行：

E:\resarch\Anaconda3\envs\face2\python.exe -m pytest -q tests/stage3_c0_sh093_224

======================================================================
二十一、执行顺序
======================================================================

严格按以下顺序：

1. 读取本提示词和项目现有代码；
2. 审计fixedSH_origcam_menafg生成契约；
3. 审计500例SH093 224资产；
4. 定位旧0.8281实验；
5. 检索匹配Original-RGB-224控制资产；
6. 构建Preflight；
7. Preflight通过后运行C0-A；
8. 完成C0-A五折训练；
9. 合并C0-A OOF；
10. 与Stage3-B0执行配对比较；
11. 若匹配控制可用，运行C0-A-Control；
12. 生成C0-A报告；
13. 构建C0-B Patient-only队列；
14. 生成C0-B外层和inner split；
15. 运行C0-B五折训练；
16. 合并C0-B OOF；
17. 运行人群协变量基线；
18. 运行低维图像域基线；
19. 执行Patient 1:1匹配；
20. 匹配规模允许时运行匹配C0-B；
21. 生成C0-B报告；
22. 联合解释C0-A和C0-B；
23. 生成最终决策；
24. 运行全部测试；
25. 执行输出完整性审计；
26. 停止。

不得自动进入后续实验。

======================================================================
二十二、正式报告要求
======================================================================

联合报告必须包括：

1. 研究目的；
2. R3DPR SH093实际数据流；
3. fixedSH_origcam与fixedSH_fixedcam区别；
4. fixedSH_origcam_menafg生成契约；
5. SH093统一了什么；
6. SH093不能保证消除什么；
7. 旧0.8281实验审计；
8. C0-A队列和split；
9. C0-A训练协议；
10. C0-A pooled OOF指标；
11. C0-A五折结果；
12. C0-A置信区间；
13. C0-A与Stage3-B0配对比较；
14. 匹配Original-RGB-224控制结果或不可用原因；
15. C0-A同相机信号等级；
16. C0-B为何使用Patient-only；
17. C0-B队列和split；
18. C0-B pooled camera AUC；
19. C0-B五折结果；
20. C0-B相机域证据等级；
21. 人群协变量基线；
22. 低维图像域基线；
23. 1:1匹配敏感性；
24. C0-A和C0-B联合解释；
25. 0.8281结果当前应如何定位；
26. SH093是否可作为主要二分类输入；
27. 是否建议多SH稳定性审计；
28. 是否建议fixedcam vs origcam审计；
29. 是否建议终止SH093主线；
30. 研究限制。

必须明确声明：

- SH093统一的是显式SH光照；
- SH093不是相机无关反射率图；
- C0-B是相机相关域可预测性，不是纯设备指纹；
- 未使用HONOR进行C0-A；
- 未使用Control进行C0-B；
- 未进入Stage3-B1；
- 未进行多SH训练。

======================================================================
二十三、Codex最终回复要求
======================================================================

完成后回复：

1. 新增和修改的代码文件；
2. 实际运行环境；
3. SH093 224预处理契约；
4. 旧0.8281实验定位结果；
5. 旧实验是否存在split或checkpoint选择问题；
6. 匹配Original-RGB-224是否找到；
7. C0-A实际队列数量；
8. C0-A五折分布；
9. C0-A每fold best epoch；
10. C0-A每fold inner val AUC；
11. C0-A每fold outer test AUC；
12. C0-A pooled OOF AUC和95% CI；
13. C0-A Accuracy；
14. C0-A Balanced Accuracy；
15. C0-A Macro-F1；
16. C0-A Sensitivity；
17. C0-A Specificity；
18. C0-A Brier；
19. C0-A有几折AUC>0.5；
20. C0-A与Stage3-B0 Delta AUC和95% CI；
21. 匹配Original-RGB-224结果；
22. c0a_same_camera_signal_level；
23. C0-B实际队列数量；
24. C0-B Xiaomi/HONOR数量；
25. C0-B五折分布；
26. C0-B每fold AUC；
27. C0-B pooled camera AUC和95% CI；
28. C0-B Balanced Accuracy；
29. C0-B Macro-F1；
30. Xiaomi recall；
31. HONOR recall；
32. 人群协变量基线AUC；
33. 低维图像域基线AUC；
34. 匹配成功数量；
35. 匹配后camera AUC；
36. c0b_camera_domain_predictability_level；
37. full500_sh093_interpretation；
38. sh093_main_binary_candidate；
39. next_stage_recommendation；
40. 测试结果；
41. C0-A报告路径；
42. C0-B报告路径；
43. 联合报告路径；
44. 两个OOF路径；
45. split路径；
46. checkpoint路径；
47. 最终决策JSON路径；
48. 明确声明未进入多SH、fixedcam或Stage3-B1。

不要只回复“任务完成”。

必须提供足够具体的数值，使后续可以判断：

- SH093是否在Xiaomi同相机内恢复稳定疾病信号；
- SH093是否仍保留明显设备/采集域信息；
- 完整500例AUC 0.8281是否可信；
- SH093是否有资格成为主要二分类实验；
- 下一步应进入多SH稳定性、fixedcam审计，还是终止该路线。