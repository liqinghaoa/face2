# Global ResNet18 + 六维光学表型融合实验实现报告

## 1. 实现状态

`GLOBAL_OPTICAL_FUSION_IMPLEMENTATION_STATUS=READY_FOR_SERVER_TRAINING`

本地实现、静态检查、真实元数据 preflight、新增测试、相关 Stage 1/2A/2B 回归测试和五组 CPU smoke-test 均已完成。当前没有执行正式 5 variants × 5 folds 训练，也没有生成或伪造正式性能结果。

## 2. 新增文件

正式新增 14 个实现/配置/测试文件：

- `models/resnet18_optical_fusion.py`
- `datasets/global_optical_fusion_dataset.py`
- `utils/optical_feature_preprocessor.py`
- `trainers/global_optical_fusion_trainer.py`
- `evaluators/global_optical_fusion_evaluator.py`
- `scripts/train/train_global_optical_fusion_5fold.py`
- `scripts/run/run_global_optical_fusion_5fold.py`
- `scripts/evaluate/summarize_global_optical_fusion_5fold.py`
- `config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml`
- `tests/test_global_optical_fusion_model.py`
- `tests/test_global_optical_fusion_dataset.py`
- `tests/test_global_optical_fusion_protocol.py`
- `tests/test_global_optical_fusion_checkpoint.py`
- `tests/test_global_optical_fusion_summary.py`

此外生成本报告、正式 protocol-only 审计文件和独立 smoke 产物。

## 3. 历史文件修改情况

本任务修改历史文件数量：`0`。

没有修改已有 Global 模型、Dataset、Trainer、Evaluator、Stage 1、Stage 2A、Stage 2B、五折 split、标签或图像。开始任务前已经存在的用户未提交修改和未跟踪文件均原样保留。

## 4. 实际读取并复用的现有基线

实现前实际读取了现有 ResNet 构造器和 backbone factory、`NYHA3ClassFaceDataset` 与 `build_transforms`、现有 Trainer/Evaluator、五折训练和汇总入口、分类 metrics、weighted CrossEntropy、YAML/seed/路径工具、Stage 1/2A/2B schema/manifest/测试，以及预实现审计报告。

复用语义包括 `ResNet18_Weights.IMAGENET1K_V1`、Pillow RGB 图像读取、Resize + train-only horizontal flip + ImageNet Normalize、fold-specific `N/(C×count_c)` 类别权重、现有三分类指标和严格 Macro-AUC checkpoint 选择。

## 5. 五个 variant

| variant | 辅助输入 | aux 维度 | fused 维度 | 分类头参数 |
|---|---|---:|---:|---:|
| `global_only` | 无；不读取光学表 | 0 | 512 | 1,539 |
| `global_mask` | `forehead_available` | 1 | 513 | 1,542 |
| `global_raw` | Stage 1 Raw 六维 + mask | 7 | 519 | 1,560 |
| `global_stage2a` | 同分类 fold 的 Stage 2A 六维 + mask | 7 | 519 | 1,560 |
| `global_stage2b` | 同分类 fold 的 Stage 2B 六维 + mask | 7 | 519 | 1,560 |

## 6. 模型结构

`ResNet18OpticalFusion` 将 torchvision ResNet18 原 `fc` 替换为 `Identity`，`forward_features` 输出 `[B,512]`。backbone 全部可训练。模型只执行 Global 特征与 aux 的直接拼接，再经过一个 `Linear(512/513/519,3)`；没有新增 MLP、投影、Dropout、BatchNorm、attention、gate、FiLM 或辅助损失。

forward 对 aux 维度、batch 对齐、浮点 dtype、NaN/Inf 和最后一维二值 availability 做严格检查；G0 只接受 `None` 或 `[B,0]`。

## 7. Dataset 数据流

`GlobalOpticalFusionDataset` 组合现有 `NYHA3ClassFaceDataset`。split CSV 的行序是唯一主顺序，特征通过完整字符串 ID 一对一 join 后重排到该顺序。重复、缺失或意外 ID 会失败。Stage 2A/2B 还校验当前分类 fold 和文件角色。

逐折 train CSV 中每行原 `fold` 表示该样本自身的留出归属，不等于当前训练 fold；当前分类 fold 由明确选择的 `fold_k_train.csv`/`fold_k_val.csv` 路径定义，batch 返回当前运行 fold。最终 OOF 不写绝对 `image_path` 或六维原值。

## 8. 六维字段与正向白名单

Raw、Stage 2A、Stage 2B 各自使用不可变的六字段顺序，并与正式 schema 逐元素交叉验证。CSV 仅通过显式 `usecols` 读取 ID、必要 fold/role、availability 和当前 variant 的六个字段；不通过 dtype、正则、剩余数值列或负向排除推断输入。

camera、EXIF、condition、predicted condition、residual、QC、NYHA、SEX 和 patient group 均不进入 aux tensor。

## 9. 逐折 Stage 2A/2B 文件选择

分类 fold k 的 Stage 2A 仅使用 `fold_k/train_calibrated_features.csv` 与 `fold_k/val_calibrated_features.csv`；Stage 2B 仅使用 `fold_k/train_nn_calibrated_features.csv` 与 `fold_k/val_nn_calibrated_features.csv`。

500 行 Stage 2A/2B OOF 文件仅用于 preflight 集合完整性审计，路径守卫会拒绝它们作为分类器输入。

## 10. Scaler 实现

`FeatureScaler` 提供独立 fit/transform/save/load：

- float64 计算 mean/std，`ddof=0`；
- 前三维 cheek 使用当前 outer train 全部病例；
- 后三维只使用当前 outer train 且 `forehead_available=1` 的病例；
- std `<1e-8`、非有限值、availability 与 NaN 模式不一致均立即报错；
- transform 输出 finite float32；
- G0 和 G-Mask 不拟合六维 scaler；Raw/2A/2B 每 fold 独立拟合，绝不共用。

JSON 保存字段顺序、mean/std/valid_n、available/unavailable 计数、source/schema/upstream manifest/train ID/split/config/code 哈希和时间戳。checkpoint 内嵌相同 payload 与哈希，Evaluator 只 load/transform，不 refit。

## 11. 缺失处理

available 病例后三维必须有限；unavailable 病例后三维必须全部为 NaN。标准化时先使用 train 统计量变换，再把 unavailable 病例的后三维填 0；前三维 cheek 保留，availability 作为最后一维 0。所有 14 个 unavailable 病例均保留。

## 12. Checkpoint 与恢复

best/last checkpoint 保存模型、optimizer、epoch、best epoch/Macro-AUC、patience、variant/fold、模型维度与参数量、字段顺序、scaler payload/hash、source/schema/manifest/train/val/split/config 哈希、类别映射/权重、transform、seed、软件/设备/git相关元数据和 RNG 状态。

resume 恢复模型、optimizer、epoch、best metric、patience、Python/NumPy/torch/DataLoader generator RNG。对于 Raw/2A/2B，resume 只加载既有 `feature_scaler.json`，在任何输出重写前核验 train ID、source、schema、manifest、split、config 和 code 哈希，禁止重新拟合或覆盖 scaler；variant、feature order、scaler/source/config/code/split 身份不一致时拒绝恢复。五组 smoke 的 checkpoint 恢复和一次真实 G-Raw `--resume` 已通过，scaler JSON、checkpoint 和 fold manifest 的 payload hash 完全一致。

## 13. Trainer

独立 Trainer 执行 `model(images, aux_features)`，采用 AdamW、weighted CE、无 scheduler/warmup/gradient clipping/AMP。每 epoch 记录 train/val loss、accuracy、Macro-AUC、val balanced accuracy、val Macro-F1、学习率、耗时、best 标记和 patience。

best checkpoint 仅在 val Macro-AUC 严格增大时更新，相等保留较早 epoch；正式预算 50 epochs、patience 10。outer validation 每 epoch 用于 early stopping，因此结果是 held-out validation，不是独立 test。

## 14. Evaluator

Evaluator 加载并验证 `best_macro_auc.pth` 及 scaler 哈希，对当前 fold val 做一次最终推理，保存三类概率、预测、标签和允许的审计元数据，计算全部既有分类指标并输出 JSON/CSV/PNG confusion matrix。G0 的 availability 只在评估后作为 QC 字段补充，不进入模型输入。

## 15. OOF 与 summary

summary 仅在五个 variant 的 25 个正式 fold manifest 全部为 `COMPLETE` 且 `formal_result=true` 后执行。每套 OOF 必须为 500 行、500 唯一 ID、概率有限且和为 1，并与固定 split 的 ID/patient group/fold/true label 逐行一致。

每个 variant 分别保存 fold metrics、fold mean/sample-std/median/min/max、pooled OOF metrics、OOF confusion matrix和跨折训练曲线。五折均值与 pooled OOF 不混写。

## 16. Paired bootstrap

实现七个预注册比较。逐折和 pooled OOF delta 均为 candidate-reference。bootstrap 默认 2,000 次、seed 2026，以 `patient_group_id` 为整体采样单位、按 true label 分层，并对 candidate/reference 使用同一个样本；重复抽到患者组时重复其全部图像。输出 Macro-AUC、Accuracy、Balanced Accuracy、Macro-F1 和三类 AUC 的 percentile 95% CI、有效重复数与失败原因，不根据 p 值自动宣布胜负。

## 17. RNG 公平性

`fold_seed=2026+fold`，同 fold 五个 variant 相同。model、DataLoader shuffle、augmentation 和 validation 使用分离且记录的 seed。每个 variant 先重置 model seed 并构建模型，再重置数据随机流；DataLoader 使用显式 generator、`num_workers=0`，每 epoch 重置 augmentation seed。测试覆盖相同 fold 的顺序/翻转一致、variant 执行顺序独立、不同 fold 可复现且不同。

## 18. Preflight 结果

`PROTOCOL_STATUS=PASS`，27 项关键检查通过。验证了：500 meanbg 图像、500 标签/split/Stage 1/Stage 2A OOF/Stage 2B OOF ID 集合完全一致；115/237/148；483 patient groups 与 17 个多图组；每 fold 400/100 且 patient group 不交叉；五折 val 覆盖 500 一次；2A/2B 逐折 ID/fold/role、Raw 血缘、availability/NaN 模式、schema 顺序和 COMPLETE manifest；486/14 availability；OOF 禁止作为 train；正向白名单不含禁用字段。配置校验现为整段精确匹配，正式输入/输出路径、ImageNet mean/std、队列统计、训练参数和 bootstrap 参数均不能漂移；替代图像目录、历史实验输出目录和修改后的 mean/std 已验证会被拒绝。

正式 protocol manifest：`experiments/global_resnet18_optical_fusion/protocol/protocol_manifest.json`。

## 19. 新增单元与协议测试

新增五个测试文件共 `59 passed`，覆盖模型结构/错误输入、scaler、Dataset/ID join/allowlist、persisted-scaler resume、checkpoint/RNG、配置漂移拒绝、真实 metadata preflight、正式 fold 缺失产物拒绝、OOF 对齐、delta 方向、患者组 paired bootstrap和服务器完整终端输出。

## 20. 既有 Stage 测试

Stage 1/2A/2B 相关既有测试：`66 passed`。

新增与既有相关测试合并回归：`125 passed`。runner 已真实执行这套固定测试清单并写入 `experiments/global_resnet18_optical_fusion/protocol/test_audit.json`，其中记录命令、文件、实现签名、config SHA256、返回码、输出和耗时；正式训练前会自动复用有效证据或重新执行，失败时停止。

## 21. 协议测试

真实项目元数据协议测试和 runner `--protocol-only` 均为 PASS。协议目录还保存可验证的 `test_audit.json`；正式 summary 会读取 protocol、test 和五组 smoke 的真实 manifest，任一证据缺失、失败或与当前 config/code 签名不一致时拒绝生成 COMPLETE。

## 22. Smoke-test

`SMOKE_TEST_STATUS=PASS`。五个 variant 均使用 3 个类别平衡样本、最多 1 epoch、CPU、64×64 smoke-only resize和随机初始化 backbone，完整通过 forward、backward、weighted CE、best/last checkpoint、恢复、Evaluator、概率/metrics/confusion matrix输出。所有 smoke manifest 均绑定当前实现签名；另行执行的 G-Raw resume 成功证明 scaler 不会被重新拟合。

smoke 产物位于 `experiments/global_resnet18_optical_fusion/smoke/`；所有 fold manifest 均为 `SMOKE_COMPLETE`、`formal_result=false`、`full_training_executed=false`，不会参与正式 summary 或模型选择。

## 23. 当前 CUDA 状态

- Python：3.13.9
- PyTorch：2.12.0+cpu
- `torch.cuda.is_available()`：false
- 当前 Python 环境不能进行 CUDA 正式训练。

## 24. 正式训练执行情况

`FULL_TRAINING_EXECUTED=false`。

已实际验证 CPU 安全门：无 `--allow-cpu-training` 时，`all variants × all folds` 命令退出码为 1，并明确拒绝启动；正式 variant 目录数量保持 0。

## 25. 未完成事项

唯一未完成的是在具有正确 CUDA PyTorch 环境的服务器上执行 25 个正式 fold run，然后运行完整 summary。当前没有正式 OOF、正式 pairwise 结果或正式实验报告；本地不得根据 smoke 结果选择 variant。正式 summary 会在写 COMPLETE 前验收每个 fold 的全部规定文件、非空状态和实际 SHA256，并生成包含逐折 best epoch、五折统计、per-class变化、全部 bootstrap CI、过拟合/分布偏移审计和候选建议的完整报告及提示词规定的终端摘要。

## 26. 服务器正式命令

先执行协议检查：

```text
python scripts/run/run_global_optical_fusion_5fold.py --config config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml --variant all --fold all --protocol-only
```

再执行正式训练（runner 完成 25 folds 后自动 summary）：

```text
python scripts/run/run_global_optical_fusion_5fold.py --config config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml --variant all --fold all
```

中断后可使用同一命令加 `--resume --skip-completed`；全部训练完成后也可单独运行：

```text
python scripts/run/run_global_optical_fusion_5fold.py --config config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml --variant all --fold all --summarize-only
```

## 27. 正式验收条件

1. CUDA 环境通过 protocol-only 和全部测试；
2. 25 个正式 fold manifest 全部为 `COMPLETE` 且 `formal_result=true`；
3. 每 fold 的 config/log/curve/best/last/predictions/metrics/confusion/distribution/scaler（适用时）和哈希完整；
4. 五套 OOF 各 500 行、500 唯一 ID，与固定 split 完全一致；
5. 七组逐折/pooled delta 和患者组 paired bootstrap 完成且有效重复数达阈值；
6. 所有五个 variant、所有正负变化、折间差异、per-class 指标、分布偏移和 outer-val 乐观性进入正式报告；
7. `summary/run_manifest.json` 和正式报告完整后，才允许写 `GLOBAL_OPTICAL_FUSION_EXPERIMENT_STATUS=COMPLETE`。

## 28. 历史输入未修改声明

本任务没有修改或重跑 Stage 1、Stage 2A、Stage 2B；没有修改 fixed split、标签、meanbg 图像、历史 checkpoint、历史 OOF 或既有实验代码。没有安装或升级依赖，没有使用 camera/EXIF/condition/residual/QC/临床/性别作为模型输入，也没有把 500 行 Stage 2A/2B OOF 当作分类训练表。
