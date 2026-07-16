# 未解决项与显式适配项

审计结论不是“数据阻塞”，而是 `READY_WITH_EXPLICIT_ADAPTATIONS`。以下事项必须在正式实现或训练前明确处理。

## 1. 未找到已完成的 meanbg ResNet18 G0 产物

- 已找到正式配置：`config/train/preprocess_ablation_resnet18/nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml`。
- 已找到正式图像：`data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images/`（500 PNG）。
- 未找到配置声明的输出根 `experiments/preprocess_ablation_500Data/`，也未找到该配置对应的 checkpoint、run manifest、fold metrics 或 OOF。
- 已搜索目录：`experiments/`、`reports/`、`config/train/`、`scripts/run/`。
- 已搜索关键词：`hybrid_imagenet_meanbg`、`PreprocAblation_ResNet18_NYHA3Class_hybrid_imagenet_meanbg`、`imagenet_meanbg`、`meanbg`、`ResNet18`。
- 处理：正式融合实验必须在同一新 runner 中训练新的 meanbg G0；不能拿已完成的 strict-blackbg 历史结果充当 G0。

## 2. 历史 500Data ResNet18 产物不是自包含可复现实验

- `experiments/500Data/Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold/config.yaml` 没有 `data.image_root`。
- 当前 `data/processed/splits_500/fold_*_{train,val}.csv` 也没有 `image_path` 列；因此用当前 Dataset 读取该保存配置会失败。
- 历史运行时间为 2026-06-24，meanbg 正式数据/配置属于后续预处理链；历史 split provenance 指向 strict-blackbg 图像。
- 处理：历史产物只用于还原模型/训练协议和 checkpoint 形状证据；新配置必须显式写 meanbg `image_root`。

## 3. 未找到现成的 Global `forward_features` / `return_features` 接口

- 已搜索目录：`models/`、`datasets/`、`trainers/`、`scripts/train/`。
- 已搜索关键词：`forward_features`、`return_features`、`feature_extractor`、`fc = nn.Identity`、`avgpool`。
- 仅在 `models/global_roi_fusion_model.py` 和 `models/multi_roi_fusion_nyha_3class.py` 找到把 ResNet `fc` 替换为 `Identity` 的多图分支实现；正式单图基线 `models/resnet_nyha_3class.py` 直接返回 logits。
- 处理：新增轻量 ResNet18 光学融合包装器；不要改变历史模型类和 checkpoint 键。

## 4. 未找到语义完全匹配的“最终六维分类特征 scaler”

- 已搜索目录：`utils/`、`datasets/`、`trainers/`、`scripts/`、`tests/`。
- 已搜索关键词：`StandardScaler`、`standardize`、`scaler`、`ddof`、`fillna`、`imputer`、`std_epsilon`。
- 找到 Stage 2A condition scaler、Stage 2B target scaler，以及分析脚本中的通用 StandardScaler；它们都不同时实现“前三维全训练集、后三维仅 availability=1、标准化后缺失填 0、保存 feature order”的目标语义。
- 处理：新增独立 `optical_feature_preprocessor`；可复用 JSON/哈希/校验风格，不可直接复用现有 scaler 对象。

## 5. schema 的负向 forbidden 列表不够完整

- Stage 1 表含 camera、EXIF、条件、区域中间量和 QC；其 `forbidden_direct_classifier_columns` 未覆盖所有非目标数值字段。
- Stage 2A/2B schema 的 usage note 禁止混用 raw/residual/calibrated，但程序化 forbidden 列表没有完整列出 raw/residual 字段。
- 处理：每个 variant 使用固定、版本化、顺序敏感的正向 `usecols`；任何“选择所有数值列再排除”都不合格。

## 6. 当前 Trainer/Evaluator 不能原样接收辅助向量

- `NYHA3ClassTrainer` 和 `NYHA3ClassEvaluator` 都执行 `model(batch["image"])`。
- 处理：新增专用 trainer/evaluator，或给现有类增加默认保持历史行为的 batch-forward hook；由于当前 trainer/summarizer 已有用户未提交修改，优先新增薄适配层并避免覆盖这些工作。

## 7. 公平比较的 RNG 隔离需要补强

- 当前单图 runner 只在五折循环前调用一次 `set_random_seed`；不同输入头维度会改变全局 RNG 消耗，可能使 RandomHorizontalFlip 和后续 fold 初始化在 variant 间不完全一致。
- 处理：每个 `(variant, fold)` 在模型构建前固定重置同一 fold seed，并把图像增强随机流与模型参数初始化随机流隔离；五个 variant 使用相同 fold、epoch、loss、早停和 loader 预算。

## 8. 当前 PyTorch 环境不能用于正式 GPU 训练

- Python：3.13.9；PyTorch：2.12.0+cpu；`torch.cuda.is_available()` 为 `False`；`torch.version.cuda` 为 `None`。
- NVIDIA 驱动能看到 `NVIDIA GeForce RTX 4060 Laptop GPU`，但当前 Python 安装的是 CPU-only PyTorch。
- 直接先导入 torch 会触发重复 OpenMP 运行库错误；项目训练脚本通过先导入 sklearn 避免该问题。
- 处理：不影响本次编码审计；正式训练前需由用户在既有环境管理流程中恢复可用 CUDA PyTorch。根据任务边界，本次没有安装或升级依赖。

## 9. 现有评估输出缺少的项目

- 已有：逐折概率、ID、fold metrics、OOF、混淆矩阵、五折均值/标准差、per-class AUC、severe-vs-rest、normal-vs-abnormal。
- 未找到：通用 Global 评估器生成的 ROC 坐标表、ROC 图、bootstrap 置信区间。
- 已搜索目录：`evaluators/`、`metrics/`、`scripts/evaluate/`。
- 已搜索关键词：`bootstrap`、`confidence interval`、`roc_curve`、`ROC`、`auc_ci`。
- 处理：这些不是 512+7 融合编码的阻塞项；若列为正式实验验收产物，应在新 evaluator/summary 中统一为五个 variant 增加。

## 10. Stage 2B train/val 分布风险是已知风险，不是缺失文件

- 2B train 是完整 outer-train MLP 的 in-sample 转换，val 是同一 MLP 的严格 outer-val 转换。
- 现有报告显示 10 个网络中有 7 个 outer-val MSE 高于完整 train MSE，最大差为 0.324196；本次直接计算六维 train/val 均值差的最大绝对 train-std 单位为 0.386622。
- 处理：本轮最小实现沿用现有 fold-specific train/val 文件并在报告中披露。若以后改用 cross-fitted train 校准特征，必须预先定义并同时重做 2A/2B 和全部五个分类 variant，不能只为 G-B 事后调整。

## 11. 工作区已有未提交内容

- 审计开始时工作区已有多项已修改/未跟踪文件，且与 Global runner、trainer、summary、Stage 1/2A/2B 相关。
- 本次没有清理、覆盖或改写这些内容；仅新增 `reports/global_resnet18_optical_fusion_preimplementation/` 下的审计文件。
- 正式实现前应先由用户确认这些在途改动的归属，避免实现任务覆盖当前工作。
