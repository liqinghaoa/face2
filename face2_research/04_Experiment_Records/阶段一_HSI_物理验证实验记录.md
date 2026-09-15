# 阶段一 HSI 物理验证实验记录

## 文档用途与维护规则

本文是阶段一真实 HSI 物理验证的持续实验记录，覆盖 `S1-0` 至 `S1-7`。上位方案以[阶段一 HSI 物理验证运行手册](../03_Method_and_Experiment/阶段一_HSI_物理验证运行手册.md)为准；本文记录实际执行内容、产物、结果、异常和门控结论。

后续每实现或运行一个阶段，都必须在本文对应章节追加日期、代码、命令、输入、产物、结果、人工判断和下一步授权。未执行的步骤不得写成完成；失败记录不得被后续成功运行覆盖。Train 用于开发，Validation 用于冻结，Test 仅在规范冻结后使用，必须记录各阶段访问过的数据范围。

论文图片只允许使用数据提供方许可的 `p012`、`p019`、`p027`。其他样本的 QC 图仅限内部审查。数值冲突时，以带哈希的 JSON、CSV、Parquet、冻结规范和最终 decision 文件为准。

## 当前状态

更新日期：2026-09-09

| 阶段 | 状态 | 当前结论 | 下一步 |
|---|---|---|---|
| S1-0 数据合同 | PASS | 306 对 RGB/VIS 完整、可读、无受试者泄漏；证据状态和哈希已记录 | 已授权 S1-1 |
| S1-1 RGB–HSI 配准 | PASS / FROZEN | 自动核验和 QinghaoLi 人工复核均通过；冻结 `transpose` | 已授权 S1-2 |
| S1-2 分层 mask/ROI | PASS_FOR_DEVELOPMENT | r5 主域自动和 QinghaoLi 人工复核通过；压力域失败保留；不构成独立验证 | 已授权 S1-3 |
| S1-3 区域光谱与波段 QC | PASS_FOR_S1_4 | 282 个 Train/Validation 样本完成区域缓存；目标域双侧脸颊 94/94 可用；Test 零访问 | 已授权 S1-4 |
| S1-4 候选模型与测试 | PASS_FOR_S1_5 | v2 六模型注册与数值审计通过；P3-O/P4 仅实现但未激活；Test 和真实区域光谱零访问 | 已授权 S1-5 |
| S1-5 Train 开发 | S1_5_COMPLETE_REVISE_BEFORE_S1_6 | v2 已完成；B0 明显优于物理候选，P2/P3-S 边界坍缩，P3-O/P4 未触发 | 先修订前向模型并只用 Train 重跑 |
| S1-4R 修订模型与合成审计 | PASS_FOR_REVISED_TRAIN_DEVELOPMENT | r7 补齐包装/无包装极限、双空间前向、proxy/PCA/K–M 和十二项合成审计；严格 RTE 保持 deferred | 已完成 S1-5R；不直接授权 S1-6 |
| S1-5R 修订模型 Train 开发 | 历史旧门：REVISE_OR_STOP | D2/D3 表示增益成立；M/tilt 语义门失败；K2 覆盖性与边界失败 | 旧 decision 保留，不作为当前表示层门 |
| S1-5R 表示层重判 | REPRESENTATION_READY_FOR_S1_6 | 只复用 Train 证据；D1/D2/D3 与 B0/B2 对照获准进入 Validation；不评估 M/H 生理身份 | 已授权并完成 S1-6 |
| S1-6 Validation 选择与冻结 | S1_6_FROZEN_FOR_S1_7 | D3 误差最低，D2 位于 10% 近最佳区间并按低维优先冻结；Test 零访问 | 只授权一次 S1-7 |
| S1-7 正式 Test | STAGE1_COMPLETE / PASS | 8/8 主脸颊可用；D2-MH 在 4/4 名 Test 受试者上优于 B0-S；解释层仍未决 | 已授权进入 S2，禁止覆盖重跑 Test |

S1-7 v1 已完成且正式 Test 已使用 1/1 次。阶段一冻结 `D2-MH` 为二维 `theta_repr`，并将 `a_obs` 作为重构辅助量单独保存；下一步进入阶段二。不得再根据 Hyper-Skin Test 改变模型、阈值、mask、波段或全局校准，也不得把两个坐标提前解释成生理 M/H。

## 共同数据边界

```text
原始数据：E:\projects\face2\data\raw\Hyper-Skin(RGB, VIS)
派生数据：E:\projects\face2\data\processed\HyperSkin_Stage1_v1
```

- Train：44 名受试者、264 次采集，用于配准、mask、阈值和模型开发。
- Validation：3 名受试者、18 次采集，用于检查规则泛化并冻结最终规范。
- Test：4 名受试者、24 次采集，冻结前不得用于模型、mask 或阈值选择。

S1-0 为核对数据合同，对所有 split 执行了文件发现、文件哈希、容器结构和数值完整性扫描，没有进行模型选择。S1-1 只读取 Train 图像内容，未打开 Validation/Test HSI 进行配准选择。S1-2 已处理 Train 和 Validation：Train 用于规则开发与冻结，Validation 只使用冻结规则生成 mask，并在其上分别保留广泛压力门与目标部署域重评分；Test 图像内容访问计数为 0。

## S1-0：证据化数据合同

### 目标与实现

目标是把 RGB/VIS 配对、样本语义、受试者划分、HDF5 结构、物理量、波长状态、数值范围和输入哈希从隐含假设转换为可审计合同。

主要代码：

- [`data_contracts.py`](../../src/skin_optics_hsi/data_contracts.py)：发现 RGB/VIS 配对，严格解析 `subject_expression_direction`。
- [`s1_contract.py`](../../src/skin_optics_hsi/s1_contract.py)：生成 SHA-256、检查 RGB/HDF5、执行数值扫描并写出证据化合同。
- [`audit_hyperskin_contract.py`](../../scripts/skin_optics_hsi/audit_hyperskin_contract.py)：S1-0 命令行入口，默认全量扫描并拒绝覆盖既有产物。
- [`test_s1_contract_and_registration.py`](../../tests/skin_optics_hsi/test_s1_contract_and_registration.py)：合同、哈希、配准和冻结门测试。

关键约定：

- 400–700 nm、31 波段、`400:10:700 nm` 标称中心和升序通道映射均记为已确认；中心数组证据状态为 `confirmed_from_official_code`。
- 物理量记为文档支持的 normalized spectral reflectance，同时注明文件缺少独立校准元数据。
- 发布版有效 SRF/带宽记为 `missing`，不把原始 FX10 的 5.5 nm FWHM当作发布波段带宽。
- S1-0 不提前写死空间映射；原始数组不修改，边界误差仅允许在派生数组中裁剪。

### 执行命令

```powershell
python scripts/skin_optics_hsi/audit_hyperskin_contract.py `
  --data-root "E:\projects\face2\data\raw\Hyper-Skin(RGB, VIS)" `
  --output-root "E:\projects\face2\data\processed\HyperSkin_Stage1_v1" `
  --value-scan full
```

### 产物

- [数据合同](../../data/processed/HyperSkin_Stage1_v1/contracts/data_contract.json)
- [合同证据说明](../../data/processed/HyperSkin_Stage1_v1/contracts/contract_evidence.md)
- [完整划分与文件哈希清单](../../data/processed/HyperSkin_Stage1_v1/manifests/split_manifest.csv)

### 实测结果

| 检查项 | 结果 |
|---|---:|
| RGB/VIS 配对总数 | 306 |
| Train / Validation / Test | 264 / 18 / 24 |
| Train / Validation / Test 受试者 | 44 / 3 / 4 |
| 受试者跨 split 泄漏 | 0 |
| HSI 数据键 | `cube` |
| HDF5 存储形状 | `[31, 1024, 1024]` |
| HSI dtype | `float64` |
| RGB 形状 | `1024 × 1024` |
| 数值扫描方式 | 全量 |
| 全局最小值 / 最大值 | `-3.2573552744e-09` / `1.000000009775249` |
| NaN / Inf | 0 / 0 |
| 小于 0 / 大于 1 的值 | 7 / 2443 |

边界越界均为极小浮点误差，最大上越约 `9.8e-9`。处理决定为 warning：保持原始数据只读，后续派生数据按合同裁剪到 `[0,1]`，不因此排除整个样本。

### 门控结论与哈希

`S1-0 = PASS`。文件数、配对、划分、结构和数值完整性不存在关键矛盾，允许进入 S1-1。

S1-0 生成时将空间映射记为 pending，这是当时真实状态，合同保持不改写；该问题后来由 S1-1 decision 关闭。

```text
split_manifest.csv                   71d23c523c632f3efee0235140aed1915415e37e980f7ac4a04396743126be45
data_contract.json                   96f2ea585cbbd2ba282469541297fb91cb7b1966aff9273367f47c84dd86f367
contract_evidence.md                 5c6aad16d5d915b1a6ecdd97c46983368ac1b7581ad9a3842884590e1a9ae01f
wavelength_evidence_amendment_v1.json 5e246ac123f3cc982eb82d86d152ef8cc234a6a992d15411fe32305bd2bdd2b0
```

## S1-1：RGB–HSI 配准核验与冻结

### 目标与实现

目标是确定 HDF5 读出后的 HSI 空间轴如何映射到配对 RGB，保证 RGB mask 能作用于同一位置的 HSI 像素。

主要代码：

- [`s1_registration.py`](../../src/skin_optics_hsi/s1_registration.py)：Train 分层抽样、8 种变换、灰度/边缘指标、审查图和人工冻结门。
- [`audit_hyperskin_registration.py`](../../scripts/skin_optics_hsi/audit_hyperskin_registration.py)：自动核验入口。
- [`finalize_hyperskin_registration.py`](../../scripts/skin_optics_hsi/finalize_hyperskin_registration.py)：人工审查后冻结或拒绝坐标映射。

程序先校验 S1-0 和 manifest SHA-256，只选择 Train。对 neutral/smile × front/left/right 六个组合分别选择 RGB 中央亮度低、中、高样本，共 18 例。每例比较：

```text
identity, rot90_ccw, rot180, rot90_cw,
flip_lr, flip_ud, transpose, anti_transpose
```

HSI 配准代理为 31 波段逐像素中位数，不依赖精确中心波长。工作分辨率为 `256 × 256`，评价公式为：

```text
combined_score
= 0.75 × Pearson(RGB灰度, HSI代理)
 + 0.25 × Pearson(RGB Sobel边缘, HSI Sobel边缘)
```

预声明自动门为：18 例最佳变换一致率 1.0；中位灰度相关不低于 0.95；中位边缘相关不低于 0.70；最佳与次佳变换的中位分数余量不低于 0.05。自动通过后仍需人工看图才能授权 S1-2。

### 执行命令

```powershell
python scripts/skin_optics_hsi/audit_hyperskin_registration.py `
  --contract "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --output-root "E:\projects\face2\data\processed\HyperSkin_Stage1_v1" `
  --samples-per-stratum 3 --working-size 256
```

人工冻结命令：

```powershell
python scripts/skin_optics_hsi/finalize_hyperskin_registration.py `
  --registration-dir "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\registration_qc" `
  --reviewer "QinghaoLi" --decision pass `
  --notes "All 18 overlays align; approve transpose."
```

### 产物

- [最终配准 decision](../../data/processed/HyperSkin_Stage1_v1/registration_qc/registration_decision.json)
- [逐样本逐变换指标](../../data/processed/HyperSkin_Stage1_v1/registration_qc/registration_qc.parquet)
- [变换汇总](../../data/processed/HyperSkin_Stage1_v1/registration_qc/transform_summary.csv)
- [18 例 Train 清单](../../data/processed/HyperSkin_Stage1_v1/registration_qc/selected_train_samples.csv)
- [人工复核记录](../../data/processed/HyperSkin_Stage1_v1/registration_qc/manual_review.csv)
- [RGB–HSI 叠加汇总图](../../data/processed/HyperSkin_Stage1_v1/registration_qc/registration_contact_sheet.png)
- `registration_qc/review_panels/`：18 张单例内部审查图。

### 自动结果

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 最佳变换 | `transpose` | 唯一一致规则 | PASS |
| 18 例一致率 | 1.00000 | ≥ 1.00000 | PASS |
| 灰度相关中位数 | 0.99498 | ≥ 0.95000 | PASS |
| 灰度相关最小值 | 0.98873 | 描述性 | — |
| 边缘相关中位数 | 0.98758 | ≥ 0.70000 | PASS |
| 边缘相关最小值 | 0.96942 | 描述性 | — |
| 最佳分数余量中位数 | 0.47762 | ≥ 0.05000 | PASS |

所有 18 个样本都选择 `transpose`，且与次优变换存在明显余量，自动门 PASS。

### 人工复核与冻结

审核人：QinghaoLi。审核结论：18 个分层 Train 叠加图中的五官和面部轮廓均呈黄色，没有系统性的成对红绿边，确认 RGB 与转置后 HSI 对齐。

冻结映射：

```python
hsi_aligned = np.transpose(hsi_loaded, (1, 0, 2))
```

即交换 HSI 的两个空间轴，把 `[X,Y,band]` 统一为与 RGB 一致的 `[Y,X,band]`；不改变光谱值和波段顺序，也不插值。

```text
status = PASS
coordinate_mapping_frozen = true
frozen_transform = transpose
next_stage_allowed = true
```

`S1-1 = PASS / FROZEN`，允许进入 S1-2。S1-2 必须读取该 decision；映射未冻结、哈希不一致或 transform 不匹配时必须停止。

关键哈希：

```text
registration_decision.json     5e0a273dfb3281f385e8e0b8f8e52bd9f01e27ebde566f0cb2942d1d92b41665
registration_qc.parquet        6a286271cd50605212ba35458103fdc346e130ab0de22a58039ed9bd0e86ac77
transform_summary.csv          a4e6ea7a1b7980700f303d447390584a3f81bdc0c01938cc15d38c8fe5b0eebf
selected_train_samples.csv     7c0dcdb6a4404693b6cc73d98a6ca1bbd58e4778b979df7ab3064ede68280ff8
manual_review.csv               b26e418ab18afe2073a17555a7169a20e1b29e758dbee25e6c4baadce456587f
registration_contact_sheet.png 3334e884643d207770211f32ad62553179e330809eff7bdce337e4e02a49e5fb
```

## 实现验证与环境

```powershell
python -m pytest tests/skin_optics_hsi -q
python -m compileall -q src/skin_optics_hsi scripts/skin_optics_hsi
```

结果：`8 passed in 8.14s`，源码与脚本编译检查通过。

```text
Python 3.13.9；NumPy 2.3.5；pandas 2.3.3；h5py 3.15.1
Pillow 12.0.0；SciPy 1.16.3；Matplotlib 3.10.6；PyArrow 21.0.0
```

本次实现快照哈希：

```text
data_contracts.py                   54bc90b16b6f643ad770f1cc661a4c7630e63d3236d15e661a8c61d818f77b44
s1_contract.py                      b81200ae185add1c37aa30702d8ec2f60c7a5ab6a593fbfe2b9fa2b46c95c8b9
s1_registration.py                  80a3d934f858f5cb3c78d53acab6141f9a7d250a1feb5bcf3ea2c3476aee0256
audit_hyperskin_contract.py          8679cde521b910e79fa712cf9fcff781698bdd7251c4a524953b35e279e93636
audit_hyperskin_registration.py      7aa9e3034793c5c154a6bc37af091421aef31d701a8d73a20e42a2de8abeb859
finalize_hyperskin_registration.py   413b04f075189005c4088d429049c64c93687eb71afd8a89a75683b8bb2e0d80
test_s1_contract_and_registration.py 7210998d1a72a121f67642926b15237dc9aacce33e60f5a6fdfb2bbb9c38db83
```

这些哈希记录 S1-0/S1-1 完成时的实现快照。后续修改时不回写旧哈希，而在对应阶段追加新哈希和变更原因。

## S1-2：分层 mask、ROI、QC 与 manifest

### 当前门控状态

状态：`TRAIN AUTOMATIC PASS / MANUAL_REVIEW_PENDING`。

Train r2 已完成自动构建和定量 QC，但尚未由研究者签署人工结论，因此：

```text
protocol_frozen = false
validation_allowed = false
next_stage_allowed = false
test_access_count = 0
```

本节后续必须继续追加 Train 人工结论、冻结文件、Validation 自动/人工结果以及最终是否授权 S1-3；不能用后续成功结果覆盖下列失败运行。

### 目标与实现

目标是把“可由皮肤前向模型解释的漫反射皮肤像素”从整幅 RGB/HSI 中分离，并生成可追溯的主脸颊 ROI、次要前额 ROI 和 whole-skin 敏感性区域。mask 不使用 M/H 谱形或拟合残差选点，避免用待验证模型反向定义其输入。

主要代码与配置：

- [`s1_2_masks_v1.yaml`](../../configs/skin_optics_hsi/s1_2_masks_v1.yaml)：r2 阈值、运行时、ROI 和数据访问策略。
- [`s1_masks.py`](../../src/skin_optics_hsi/s1_masks.py)：HSI 质量层、分层交集、ROI、QC、manifest、Train 冻结与 Validation 门。
- [`s1_mask_rgb_worker.py`](../../scripts/skin_optics_hsi/s1_mask_rgb_worker.py)：在独立 Python 3.11 环境中运行 MediaPipe、OpenCV 和 BiSeNet/CUDA。
- [`build_hyperskin_masks.py`](../../scripts/skin_optics_hsi/build_hyperskin_masks.py)：只允许 `train` 或 `valid`，代码层禁止 Test；Validation 必须提供冻结 provenance。
- [`finalize_hyperskin_masks.py`](../../scripts/skin_optics_hsi/finalize_hyperskin_masks.py)：记录 Train/Validation 人工结论；不得自动以研究者名义通过。
- [`build_s1_2_review_sheet.py`](../../scripts/skin_optics_hsi/build_s1_2_review_sheet.py)：从既有 manifest 生成不覆盖原产物的分层复核图。
- [`test_s1_masks.py`](../../tests/skin_optics_hsi/test_s1_masks.py)：分层 mask、伪影排除和分层人工抽样测试。

运行时拆分如下：

```text
默认 Python 3.13：pandas / pyarrow / h5py / HSI / manifest
E:\resarch\Anaconda3\envs\face2\python.exe：Python 3.11.15、
PyTorch 2.3.0+cu121（CUDA 可用）、OpenCV 4.10.0、MediaPipe 0.10.14、BiSeNet
```

RGB worker 为每例保存 `parsing_label.npy`、`landmarks.npy`、`rgb_inference.json`。主流程按 S1-1 冻结规则先执行：

```python
hsi_aligned = np.transpose(hsi_loaded, (1, 0, 2))
```

再生成以下布尔层，均为 `1024 × 1024`：

```text
valid_region = anatomical_skin
             ∩ semantic_valid
             ∩ radiometric_valid
             ∩ illumination_valid
             ∩ region_roi
```

- `anatomical_skin`：BiSeNet 面部皮肤/鼻部候选与 FaceMesh 脸廓的交集，并腐蚀 5 px；
- `semantic_valid`：排除背景、头发、眉眼、眼镜相关遮挡、嘴唇、口腔、耳颈和衣物等语义类，并对关键点排除区膨胀 4 px；
- `radiometric_valid`：31 波段有限、范围有效，排除近零、饱和和异常像素；
- `illumination_valid`：排除 HSI 暗区/高亮、RGB 低色度镜面高光和扫描线跳变；
- `left_cheek` / `right_cheek`：图像坐标定义的椭圆脸颊 ROI；front 要求两侧，侧脸只保留主要可见侧；
- `forehead`：次要、可选 ROI，不足时标记 REVIEW 而非补造；
- `whole_skin`：敏感性分析区域，不作为唯一主结果。

侧脸命名按数据集 view，而左右 ROI 按图像坐标，不等于解剖左右。对全部 176 个侧脸 Train 样本检查两个候选脸颊面积后，`left` view 有 82/88 例 image-right 更大，`right` view 有 81/88 例 image-left 更大，且每例至少一个候选脸颊超过 1000 pixels。因此 r2 规则为：

```text
left view  -> 保留 image-right cheek
right view -> 保留 image-left cheek
front      -> 同时保留两个 cheek
```

自动样本门要求 whole-skin 至少 5000 pixels，主脸颊至少 1000 pixels；front 要求双脸颊，侧脸要求可见脸颊。前额不足只进入 REVIEW。样本级 FAIL 比例门为 0.05，REVIEW 比例门为 0.50，各表情×方向分层 usable fraction 门为 0.80。

### 失败运行与规则修订

所有失败/修订运行均保留，没有删除或覆盖：

| 运行 | 范围 | 结果 | 原因与处理 |
|---|---:|---|---|
| `outputs/skin_optics_hsi_v1/s1_2_pilot18_v1` | Train 18 | 0 PASS / 0 REVIEW / 18 FAIL | 16 例因 MediaPipe 接收到非连续 NumPy 输入，2 例 detector 失败；随后保证连续内存，并允许全图 FaceMesh fallback且显式记录 |
| `outputs/skin_optics_hsi_v1/s1_2_pilot18_v2` | Train 18 | 10 PASS / 6 REVIEW / 2 FAIL | 2 例 whole-skin 最大连通域比例略低，6 例前额次要 ROI 小 |
| `outputs/skin_optics_hsi_v1/s1_2_pilot18_v3` | Train 18 | 12 PASS / 6 REVIEW / 0 FAIL | 主层无 FAIL；全量运行后发现侧脸可见脸颊的 view 映射仍写反，未将该版本冻结 |
| `data/processed/HyperSkin_Stage1_v1/masks/`（r1） | Train 264 | 181 PASS / 52 REVIEW / 31 FAIL | 27 例 whole-skin 因眼镜、胡须等被分裂，4 例侧脸主 ROI 选错；decision 为 REVISE，完整保留 |

r2 不降低主脸颊面积要求。修订仅包括：根据 176 例定量证据纠正侧脸可见侧；将 `fragmented_whole_skin` 从 FAIL 调整为 REVIEW，因为 whole-skin 只是敏感性区域，而主脸颊仍达到面积门。眼镜、胡须、头巾和发丝造成的碎片继续被记录，不静默删除。

### Train r2 正式构建

执行命令：

```powershell
python scripts/skin_optics_hsi/build_hyperskin_masks.py `
  --contract "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --registration "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\registration_qc\registration_decision.json" `
  --output-root "E:\projects\face2\data\processed\HyperSkin_Stage1_v1" `
  --config "E:\projects\face2\configs\skin_optics_hsi\s1_2_masks_v1.yaml" `
  --split train
```

主要产物：

- [Train r2 decision](../../data/processed/HyperSkin_Stage1_v1/masks/r2/s1_2_train_decision.json)
- [Train r2 mask manifest](../../data/processed/HyperSkin_Stage1_v1/manifests/mask_manifest_train_r2.parquet)
- [六分层 QC](../../data/processed/HyperSkin_Stage1_v1/masks/r2/train_stratum_qc.csv)
- [分层人工复核图](../../data/processed/HyperSkin_Stage1_v1/masks/r2/train_mask_contact_sheet_stratified.png)
- [分层人工复核清单](../../data/processed/HyperSkin_Stage1_v1/masks/r2/manual_review_train_stratified.csv)
- `masks/r2/train/<sample_id>/`：每例 8 个 mask、RGB 推理结果、`qc.json` 和 `qc_panel.png`。

RGB worker 264/264 PASS，其中 206 例使用 expanded detector crop，58 例使用显式记录的 full-image FaceMesh fallback。BiSeNet checkpoint SHA-256 为：

```text
468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567
```

自动结果：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Train 样本数 | 264 | 固定 Train | PASS |
| PASS / REVIEW / FAIL | 183 / 81 / 0 | 描述性 | — |
| failure fraction | 0.0000 | ≤ 0.05 | PASS |
| review fraction | 0.3068 | ≤ 0.50 | PASS |
| 六分层 usable fraction | 均为 1.0000 | ≥ 0.80 | PASS |
| whole-skin pixels：min / mean / max | 54,597 / 144,285 / 212,125 | ≥ 5,000 | PASS |
| final/anatomical：min / mean / max | 0.7756 / 0.8454 / 0.9021 | ≥ 0.25 | PASS |
| radiometric exclusion：mean / max | 0.00010 / 0.00265 | ≤ 0.10 | PASS |
| illumination exclusion：mean / max | 0.01924 / 0.07920 | ≤ 0.60 | PASS |

REVIEW 原因分布：54 例仅 `forehead_secondary_roi_small`，25 例仅 `fragmented_whole_skin`，2 例同时具有两项。自动 decision 为 `AUTOMATIC_PASS_MANUAL_REVIEW_PENDING`。

额外完整性审计读取了 manifest 指向的全部 2,112 个 mask 数组：缺失/不可读 0，非 `1024 × 1024 bool` 0，侧脸可见侧规则违反 0；manifest 仅含 Train 264 例，Test 行数 0。

### 人工复核范围与当前判断

自动抽样保留 18 个最差/典型 REVIEW，并为六个 `neutral/smile × front/left/right` 分层各选 1 个 PASS 代表，共 24 例。复核图的四列依次为 RGB、橙色 anatomical、黄色 semantic core、最终 ROI；最终图中绿色为 whole-skin、青/洋红为左右图像坐标脸颊、蓝色为前额。

当前技术检查显示：24 例主脸颊 ROI 均落在可见面部皮肤；侧脸只保留主要可见侧；前额不足时没有强行补造。部分 whole-skin 被眼镜、胡须、头巾或发丝切断，少量细发丝仍可穿过 whole-skin，因此相关案例保留 REVIEW，且 whole-skin 只用于敏感性分析。该技术检查不能代替 QinghaoLi 的人工签署。

需要研究者明确查看[分层人工复核图](../../data/processed/HyperSkin_Stage1_v1/masks/r2/train_mask_contact_sheet_stratified.png)后给出通过或拒绝结论。未确认前不运行 `finalize_hyperskin_masks.py`。

### 输入与产物哈希

```text
data_contract.json                       a3f0c0543d27c0ba6a4a0729b7253b323212ffadd754ed17ff4acba0bf4799cf
registration_decision.json               5e0a273dfb3281f385e8e0b8f8e52bd9f01e27ebde566f0cb2942d1d92b41665
s1_2_masks_v1.yaml                        0f0a5aba2fdd22a66706a0a2b5608901501e31fd037b49c041eade761d567406
mask_manifest_train_r2.parquet            c34b2fe496c03192b615efad638d72515858a1e33a7aafbbab1c17248aaf9861
train_mask_contact_sheet_stratified.png   b387d6f22934ca1d655033ec10018fa35d076387963c73091432d3caa2948231
manual_review_train_stratified.csv        3003d257376c0c4ad1fda9bc5af7ade96a59f8531c7eb4358887f86ec421493b
```

当前实现快照（在 r2 运行后追加了 Validation 冻结门、实现哈希冻结、分层复核生成和 Train+Validation 正式 manifest 闭环；r2 的核心 mask 规则未改，产物以 decision/config/manifest 哈希为准）：

```text
s1_masks.py                       06516ca2e91b6470c5bdbdb874e182183192ad69bd723b89e2571d787e39a43a
s1_mask_rgb_worker.py             2794ffd831ff2cc2f71bb6106fbd8513017218d42b4ade38dc7e4e9dde248f4f
build_hyperskin_masks.py          59da490c78ceed3aa05fc15701bff7a4670ecc5d971b58d3a3758283323cd967
finalize_hyperskin_masks.py       b269d44ba60f36be8634879536f70e8cdbcbe623247c5220137f34cc8bf9320b
build_s1_2_review_sheet.py        37785358879d10a27d7614da38cd11e5c3be5cd596a1c1b3eeda85056e9e98dd
test_s1_masks.py                  6f134af924be06e75b389743b8d1d70dd9a88e582885dd088e3dcc6519738e7d
```

r2 运行时没有单独保存源代码快照哈希，这是本次可复现性限制；冻结时必须同时保存/提交当前实现或生成代码 provenance，不能只冻结 YAML。

### 验证结果与下一步

```powershell
python -m pytest tests/skin_optics_hsi -q
python -m compileall -q src/skin_optics_hsi scripts/skin_optics_hsi
E:\resarch\Anaconda3\envs\face2\python.exe -m py_compile scripts/skin_optics_hsi/s1_mask_rgb_worker.py
```

结果：`13 passed`，默认运行时全部源码/脚本编译通过，RGB worker 在专用 Python 3.11 环境编译通过。新增闭环测试验证 S1-1 必须显式授权 S1-2，并验证 Train/Validation 受试者无交叉后才会合并正式 manifest 和生成最终 decision。

Train 已完成人工 PASS 并冻结 `freeze/s1_2_mask_protocol.yaml` 与 `freeze/s1_2_mask_protocol_provenance.json`。原始 Validation 已严格使用冻结配置、数据合同、配准 decision、BiSeNet checkpoint 和实现哈希；结果为 11 PASS、6 REVIEW、1 FAIL。原始广泛 profile 因 `p016_smile_right` 失败而为 `REVISE`，该结果保留，不作为目标部署域的唯一门控。Test 访问数仍为 0。

### Validation 冻结协议检查

执行命令：

```powershell
python scripts/skin_optics_hsi/build_hyperskin_masks.py `
  --contract "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --registration "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\registration_qc\registration_decision.json" `
  --output-root "E:\projects\face2\data\processed\HyperSkin_Stage1_v1" `
  --config "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\freeze\s1_2_mask_protocol.yaml" `
  --split valid `
  --frozen-protocol-provenance "E:\projects\face2\data\processed\HyperSkin_Stage1_v1\freeze\s1_2_mask_protocol_provenance.json"
```

产物：

- [Validation decision](../../data/processed/HyperSkin_Stage1_v1/masks/r2/s1_2_valid_decision.json)
- [Validation manifest](../../data/processed/HyperSkin_Stage1_v1/manifests/mask_manifest_valid_r2.parquet)
- [Validation 分层 QC](../../data/processed/HyperSkin_Stage1_v1/masks/r2/valid_stratum_qc.csv)
- [Validation 复核图](../../data/processed/HyperSkin_Stage1_v1/masks/r2/valid_mask_contact_sheet.png)
- [Validation 复核清单](../../data/processed/HyperSkin_Stage1_v1/masks/r2/manual_review_valid.csv)

自动结果：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Validation 样本数 | 18 | 固定 Validation | PASS |
| PASS / REVIEW / FAIL | 11 / 6 / 1 | 描述性 | — |
| failure fraction | 0.05556 | ≤ 0.05 | FAIL |
| review fraction | 0.3333 | ≤ 0.50 | PASS |
| `smile/right` usable fraction | 0.6667 | ≥ 0.80 | FAIL |
| 其他五个分层 usable fraction | 1.0000 | ≥ 0.80 | PASS |

唯一 FAIL 是 `p016_smile_right`，失败码为 `pipeline_exception: rgb_worker_failed: facemesh_failed`。该样本是戴眼镜的极侧脸；RGB 原图可读，但冻结的 detector/FaceMesh 路径没有产生可靠 landmarks，因此没有生成 mask，也没有被手工补造。这个失败目前只能作为冻结规则的真实泛化失败记录，不能在 Validation 内临时切换算法或放宽阈值。

Validation 的 17 个成功推理样本已生成 QC panel；[原始广泛复核图](../../data/processed/HyperSkin_Stage1_v1/masks/r2/valid_mask_contact_sheet.png)中包括 11 个 PASS 和 6 个 REVIEW。该段是原始广泛门控的历史记录：它不通过，但不覆盖后续目标部署域 profile。Test 继续禁止访问。

### 目标部署域 Validation profile

由于真实推理数据限定为正脸、无表情，阶段一主门控不再要求侧脸和笑脸全部成功。初版 profile `v1` 曾要求主域 0 REVIEW，结果为 REVISE，记录保留在 `masks/r3/`；v2/r4 验证了按比例允许 REVIEW 可以通过，但没有在代码中限制 REVIEW 类型，作为中间结果保留。正式候选为 [`s1_2_validation_target_front_neutral_v3.yaml`](../../configs/skin_optics_hsi/s1_2_validation_target_front_neutral_v3.yaml)，只对既有冻结协议产出的 Validation manifest 重新分层评分，不重写任何 mask，也不覆盖 r2/r3/r4。

主域定义：`expression=neutral` 且 `direction=front`。主域要求 0 FAIL、REVIEW 比例不超过 0.50、usable fraction 1.0、双侧脸颊均可见且每侧至少 1000 pixels；REVIEW 码只允许 `fragmented_whole_skin` 和 `forehead_secondary_roi_small`。其余 `smile/left/right` 作为非阻断压力测试，仍须报告失败和分层统计。该 profile 的独立执行入口为 [`evaluate_s1_2_validation_profile.py`](../../scripts/skin_optics_hsi/evaluate_s1_2_validation_profile.py)。

眼镜限制必须单独记录：Hyper-Skin 合同没有 eyewear 字段；当前 Validation 的 3 个 `neutral/front` RGB 样本均可见眼镜。因此新 profile 能验证“正脸+无表情”的 mask 可用性，但不能验证“无眼镜”这一部署条件。无眼镜是目标数据的部署假设，不得写成已由 Hyper-Skin 验证的结论。

### 目标域重评分结果

初版 `target_front_neutral_v1` 使用 0 REVIEW 主门，结果为 REVISE，保留在 `masks/r3/`；v2 自动 PASS，保留在 `masks/r4/`。由于 v2 只限制 REVIEW 比例，v3 进一步把允许的 REVIEW 码和双脸颊面积写成可执行门控，结果保留在 `masks/r5/`。

执行命令：

```powershell
python scripts/skin_optics_hsi/evaluate_s1_2_validation_profile.py `
  --valid-decision "data\processed\HyperSkin_Stage1_v1\masks\r2\s1_2_valid_decision.json" `
  --profile "configs\skin_optics_hsi\s1_2_validation_target_front_neutral_v3.yaml" `
  --output-root "data\processed\HyperSkin_Stage1_v1" --output-revision r5
```

结果：`AUTOMATIC_PASS_PRIMARY_DOMAIN_STRESS_NONBLOCKING_REVIEW`。

| 范围 | 样本数 | PASS | REVIEW | FAIL | 作用 |
|---|---:|---:|---:|---:|---|
| `neutral/front` 主域 | 3 | 2 | 1 | 0 | 阻断性主门 |
| 其余压力域 | 15 | 9 | 5 | 1 | 非阻断，但必须报告 |

主域自动检查全部通过：非空、failure fraction `0.0`、review fraction `0.3333 ≤ 0.50`、usable fraction `1.0`、REVIEW 码均在白名单内、3/3 样本的双侧主脸颊均达到可用标准。压力域的唯一 FAIL 仍为 `p016_smile_right` 的 FaceMesh 失败；它不再决定正脸无表情主域是否可进入 S1-3，但必须作为超出部署域的压力测试失败报告。

产物：

- [r5 target-domain decision](../../data/processed/HyperSkin_Stage1_v1/masks/r5/s1_2_valid_target_front_neutral_decision.json)
- [r5 主域复核图](../../data/processed/HyperSkin_Stage1_v1/masks/r5/valid_target_front_neutral_contact_sheet.png)
- [r5 主域复核清单](../../data/processed/HyperSkin_Stage1_v1/masks/r5/manual_review_valid_target_front_neutral.csv)
- [r5 压力域分层 QC](../../data/processed/HyperSkin_Stage1_v1/masks/r5/valid_stress_stratum_qc.csv)

人工签署前，r5 的 `next_stage_allowed=false`，且程序没有提前生成最终合并 manifest；该状态保留为签署前记录。Test 访问数为 0。

证据解释限制：目标部署域是在观察原始 Validation 广泛门控失败后才由研究者明确收窄，因此 r5 是事后、部署域对齐的重分类结果，不是预先注册的独立 Validation PASS。该限制已写入 r5 decision 的 `evidence_policy`。人工复核通过后，它最多授权 S1-3 继续开发；不得据此声称模型已经在“正脸、无表情、无眼镜”人群独立验证。最终需要新的目标域留出集或外部数据，且当前 Hyper-Skin Test 仍不得提前访问。

关键哈希：

```text
s1_2_validation_target_front_neutral_v3.yaml   2bea58b3af6c29002b68a98da17a9349d42e9e22913e1e1b1b0dfa3e8d5e778b
evaluate_s1_2_validation_profile.py            41e1727e4507ca5c59bcfb7330bb6aa6a73c10c859c0beb7d9558d6a0373d2ba
r5 automatic decision（签署前）                 ecd6fef6da4c72a38d48cc764a4613488792e2f0ec4a764220a8ecac6884160e
r5 target-domain contact sheet                 57aadfc0cc50a29f3f72b8e69b8fb2e669306c698b69d4d411c407cb1c41c83b
```

### r5 人工签署与 S1-3 授权

审核人：QinghaoLi。审核原话：

> 确认 S1-2 Validation r5 的正脸无表情主脸颊 ROI 可以接受，同意将其作为目标域开发门并进入 S1-3；理解其不构成无眼镜目标域的独立验证。

签署结果：`PASS_FOR_DEVELOPMENT`。程序生成了带分析角色的 Train+Validation 正式 manifest：

| 分析角色 | 样本数 | 含义 |
|---|---:|---|
| `primary_development` | 44 | Train 的 `neutral/front` 主域开发样本 |
| `stress_development` | 220 | Train 的笑脸/侧脸压力样本 |
| `primary_validation` | 3 | Validation 的 `neutral/front` 主域样本 |
| `stress_validation` | 15 | Validation 的笑脸/侧脸压力样本 |

正式 manifest 共 282 行、47 名受试者、重复 sample ID 为 0、Test 行为 0。281 行可用；`p016_smile_right` 作为 `stress_validation` FAIL 保留。新增字段为 `s1_2_target_domain_flag`、`s1_2_usable_flag`、`s1_2_analysis_role` 和 `s1_2_gate_profile`，供 S1-3 明确分流。

最终产物：

- [S1-2 目标域最终 decision](../../data/processed/HyperSkin_Stage1_v1/freeze/s1_2_target_domain_final_decision.json)
- [正式 mask manifest](../../data/processed/HyperSkin_Stage1_v1/manifests/mask_manifest.parquet)
- [正式 mask manifest CSV](../../data/processed/HyperSkin_Stage1_v1/manifests/mask_manifest.csv)
- [r5 人工复核记录](../../data/processed/HyperSkin_Stage1_v1/masks/r5/manual_review_valid_target_front_neutral.csv)

```text
status = PASS_FOR_DEVELOPMENT
next_stage_allowed = true
authorized_next_stage = S1-3
independent_validation_claim_allowed = false
test_access_count = 0
```

关键哈希：

```text
s1_2_target_domain_final_decision.json   9a004ca2a0844fe671bf377c0af3d08b7735d02afb7eb65f1fbf8395faa66a5b
r5 signed target-profile decision        a087bab6a778b8a95b6cf531b8a53e89660b6305a62f8701267a53a4b75fc2a2
mask_manifest.parquet                    6cbc25225ab85da120553ad64724d12f93766f10c890fcfa75ef849afcb5ae63
mask_manifest.csv                        d022bf1f1799df1e09c1b5d5009ada92eec9f6722bb68561ebacf010cef03b3c
manual_review_valid_target_front_neutral 4178ab6440cc6d2a20937fd52dc3cf226417d983d22e581df6c08ad54032253c
s1_target_profile.py                     f8114e38eb984f3355735ad44aae0e148a8c642dee277d97e909e5d8d15e8b17
finalize_s1_2_target_profile.py           b1b4f8c06de225675019b98495d957d2598da8fa7e4e668e40d6c624b98a8fd2
test_s1_target_profile.py                 00029093ce36e7d37c99aaa79b077cbf29b353b861a76bfd20b40319864571d4
```

最终实现验证：`14 passed`，源码与脚本编译通过。S1-2 允许进入 S1-3，但只能声称目标域开发门通过，不能声称无眼镜目标域已获独立验证。

## S1-3：区域光谱与波段 QC

状态：`PASS_FOR_S1_4`。执行日期：2026-09-08。

### 实现与数据约束

新增实现：

- [`s1_region_spectra.py`](../../src/skin_optics_hsi/s1_region_spectra.py)：核验 S1-0/S1-1/S1-2 前置 decision 与哈希，提取区域中位谱、MAD、IQR、均值和 10% 双尾截尾均值；Test split 在任何 HSI 打开前硬拒绝。
- [`extract_hyperskin_region_spectra.py`](../../scripts/skin_optics_hsi/extract_hyperskin_region_spectra.py)：S1-3 正式命令入口，拒绝覆盖既有产物。
- [`s1_3_region_spectra_v1.yaml`](../../configs/skin_optics_hsi/s1_3_region_spectra_v1.yaml)：固定 Train/Validation 范围、31 波段合同、四类区域、统计量和主域脸颊门。
- [`summarize_s1_3_band_reliability.py`](../../scripts/skin_optics_hsi/summarize_s1_3_band_reliability.py)：仅使用 Train `primary_development` 双侧脸颊形成波段可靠性和侧别效应诊断，不用 Validation 选择波段。
- [`test_s1_region_spectra.py`](../../tests/skin_optics_hsi/test_s1_region_spectra.py)：覆盖稳健统计、官方代码波长状态、缺失 SRF 保留、Train/Validation 提取和 Test 文件不打开。

波长合同升级同时固化到 `s1_contract.py`：中心状态改为 `confirmed_from_official_code`，通道顺序改为 `ascending`，移除 `WAVELENGTH_CENTERS_INFERRED`，继续保留 `EFFECTIVE_SRF_MISSING`。由于 S1-2 冻结 provenance 应保持当时历史哈希，不回写该冻结文件；新增 [`wavelength_evidence_amendment_v1.json`](../../data/processed/HyperSkin_Stage1_v1/contracts/wavelength_evidence_amendment_v1.json)记录旧/新合同哈希和纯证据变更边界。

### 执行恢复记录

首次正式调用误写了 mask manifest 路径，前置门立即以 `Passed mask manifest is not the S1-2 finalized manifest` 拒绝，未创建输出。修正路径后的运行在 100/282 时因交互任务中断导致进程退出，检查确认没有输出目录或半成品。随后使用完全相同的冻结输入从头恢复；最终运行未覆盖任何既有 S1-3 产物。

正式命令：

```powershell
python scripts/skin_optics_hsi/extract_hyperskin_region_spectra.py `
  --contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --mask-manifest "data\processed\HyperSkin_Stage1_v1\manifests\mask_manifest.parquet" `
  --output-root "data\processed\HyperSkin_Stage1_v1\region_spectra"

python scripts/skin_optics_hsi/summarize_s1_3_band_reliability.py `
  --s1-3-root "data\processed\HyperSkin_Stage1_v1\region_spectra"
```

### 正式结果

| 指标 | 结果 |
|---|---:|
| 输入样本 | 282（Train 264，Validation 18） |
| 样本×区域记录 | 1128 |
| 可用 / 不可用 | 930 / 198 |
| 目标域双侧脸颊 | 94/94 可用 |
| 可用中位光谱范围 | 0.048991–0.918025 |
| 31 波段有限率 | 全部 1.0 |
| Test 访问计数 | 0 |

不可用记录均可追溯：Train 侧脸冻结遮挡侧脸颊 176 条、空前额 7 条；Validation 侧脸冻结遮挡侧脸颊 11 条；已知压力失败 `p016_smile_right` 对四个区域产生 4 条不可用。前额不足和遮挡侧脸颊未插值、未补造、未静默删除。

Train 主域诊断使用 44 名受试者的 88 条正脸无表情脸颊光谱。区域内相对 MAD 的逐波段中位数为 0.081–0.123；局部曲率最高的内部波段为 680、690、670、420 和 590 nm。400/700 nm 作为端点，400 nm 还表现出不同于主体波段的左右方向。诊断候选波段为 400、420、590、670、680、690、700 nm，但本阶段不自动删除或降权，留待 S1-4/S1-5 做全波段、端点移除、高曲率波段和 FWHM/波长偏移敏感性比较。

固定侧别效应不可忽略：正脸无表情 Train 中图像左侧脸颊宽带反射率在 44/44 名受试者上高于右侧，中位有符号差为 0.076692。这一现象更符合采集照明/几何 nuisance，S1-4 必须保留散射/观察 nuisance 接口；S1-5 应分别报告左右脸颊和对称聚合结果，不能让 M/H 参数无约束代偿。

### 产物与哈希

```text
region_spectra.parquet                    e5c29ddc37608aba119c5e07d6351361ce347fba4558bbeb980228b390820076
region_spectra.csv                        e6cd48733c7a4af6d543a5a8ee9c65ee741cf01e66c34ac71a48a93764a6439b
band_qc.parquet                           acb10ded3b391ec7e5dcb851435833fcc9e176982483686760d1adadd43b2bec
band_qc.csv                               a08368b6aec4b0f4c03ba1a835064e8a8e6ce0508021ea3f5f76bc6858439947
subject_condition_variability.parquet     d2ecfd6f367d93df81798ef355231dd0a16cb7a96f575ec652a72e8ef80750a8
subject_condition_variability.csv         02e1926f970430a7e6d222d38938cd4c72e225b4b6a6505e754f711599e6a939
band_reliability_train_primary.csv        21634d84c5f1a68910cbcf1354e7a007439009860ca18b7e9428a94ca15c9cb4
band_reliability_report.md                919ef5fe15bb462cfe1c591b85d2f6c164359316306f1d60a2a6942012363a30
s1_3_decision.json                        ffecaa41b86f58900df3835cf55815767862f511a7251eed343ff4fb508ba749
s1_3_final_decision.json                  8e090c9a22e791b6f0c6214efe58442488ba5952dc50933a44d920793ac95a97
```

实现文件哈希：

```text
s1_region_spectra.py                      8724b0e41b7b7341dc2d336b5ab5c0a3a3601480d9f7a862d9c652ff620b4ca7
extract_hyperskin_region_spectra.py       6ed56eac91584acdd25d17021c63178a1bf7c1adca34a37e0d8dfe42063cd0d7
summarize_s1_3_band_reliability.py        9527a5abc2b8e9aa4b0b9eece5edea5147b000a9fe6120a2e87f7d9d7ae09275
s1_3_region_spectra_v1.yaml               7150e8ab21ed3868b102884bfbead085950536a1df7caad520122c68cfa3ae6e
test_s1_region_spectra.py                 4a6e1416215fa12db650a046f5d36f965a59dabda4b89f8ef8d695dbaf85b96b
```

最终门控为 `PASS_FOR_S1_4`，`next_stage_allowed=true`，`authorized_next_stage=S1-4`。发布版有效 SRF 继续为 `missing`，允许代理参数开发，但正式 Test 前必须完成敏感性分析。全套 `tests/skin_optics_hsi` 为 `17 passed`，源码和脚本编译通过。

## S1-4：候选物理模型与测试

状态：`PASS_FOR_S1_5`。执行日期：2026-09-08。

### 实现范围

新增或重构以下文件：

- `configs/skin_optics_hsi/s1_4_model_registry_v1.yaml`：参数分类、单位、暂定范围、证据、模型阶梯、激活条件、全局参数、侧别观察 nuisance 和敏感性合同；
- `src/skin_optics_hsi/model_registry.py`：严格加载和验证注册表，冻结每个模型的参数顺序；
- `src/skin_optics_hsi/skin_forward.py`：统一 `theta_bio / theta_nuisance / global_params / model_id` 接口，以及 NumPy、Torch 和分量审计路径；旧 `SkinForwardModel` 保留为 P2 兼容层；
- `src/skin_optics_hsi/s1_spectral_sensitivity.py`：预注册波段集合、中心偏移与假定 Gaussian SRF/FWHM；
- `src/skin_optics_hsi/s1_model_audit.py` 和 `scripts/skin_optics_hsi/audit_s1_4_models.py`：S1-3 前置授权、Test 隔离、不可覆盖写入、数值审计和哈希决策；
- `tests/skin_optics_hsi/test_s1_models.py`：六模型、变维参数、梯度、边界、分量、侧别、曝光拒绝、敏感性和不可覆盖产物测试。

注册模型如下：

| 模型 | `theta_bio` | `theta_nuisance` | S1-5 入口状态 |
|---|---|---|---|
| B0 | 无 | 无 | 激活；实际 Train 平均谱只能由 S1-5 计算并哈希 |
| B1 | M | 无 | 激活 |
| P2 | M + H | 无 | 激活，主候选 |
| P3-S | M + H | `S_amp` | 激活，必须比较的散射混杂模型 |
| P3-O | M + H | `sO2` | 已实现但未激活；等待稳定 Hb 谱形残差 |
| P4 | M + H | `S_amp + sO2` | 已实现但未激活；仅为双重残差证据下的诊断上限 |

共同骨架使用 `m(λ)=(λ/570)^(-4.3)`、表皮双程传输、在 570 nm 归一化的 OMLC HbO2/Hb 混合谱、文献基线吸收、Jonasson 形式的约化散射以及半无限 Kubelka–Munk 真皮反射。它是降阶筛查模型，不是 Jonasson 三层 inverse Monte Carlo 的等价实现。M/H 是光学代理，S_amp/sO2 是 nuisance；四者范围均为开发期范围而非 Hyper-Skin 个体真值。

S1-3 检出的左右侧效应没有被放入 `theta_bio`。实现只允许 `image_left / image_right / other` 三个固定全局 log-gain，默认均为 0；若 S1-5 使用，只能在 Train 上一次估计后冻结。任何 `per_image_scale`、`per_spectrum_scale` 或自由曝光参数均被硬拒绝。主要报告仍必须保留左右脸颊分开结果和对称聚合，防止 M/H 代偿侧别照明。

有效 SRF 状态继续为 `missing`。预注册敏感性为：全 31 波段；去 400/700 nm 端点；去 400/420/590/670/680/690/700 nm 高曲率候选；中心偏移 `-5/-2.5/0/2.5/5 nm`；假定 Gaussian FWHM `0/5.5/10/20 nm`。中心负向偏移在全波段下会越过当前审计资产 400 nm 下界，因此必须与去端点设置配合，不能外推 Hb 谱形。高曲率候选只是敏感性集合，不是坏波段删除结论。

### 正式命令与不可覆盖修订

第一次正式运行写入 `models/s1_4_v1`。数值审计通过，但模型卡把带冒号的 Jacques 参考文献显示成 YAML 映射。没有覆盖 v1；补充“所有 references 必须是字符串”的配置校验后，新运行写入 `models/s1_4_v2`，其 decision 通过 `supersedes` 字段引用 v1 决策及哈希。v1 作为失败/恢复证据保留，不得作为后续输入。

```powershell
python scripts/skin_optics_hsi/audit_s1_4_models.py `
  --s1-3-decision "data\processed\HyperSkin_Stage1_v1\region_spectra\s1_3_final_decision.json" `
  --data-contract "data\processed\HyperSkin_Stage1_v1\contracts\data_contract.json" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\models\s1_4_v2" `
  --registry "configs\skin_optics_hsi\s1_4_model_registry_v1.yaml" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\models\s1_4_v1\s1_4_decision.json"
```

正式 v2 产物：

```text
data/processed/HyperSkin_Stage1_v1/models/s1_4_v2/
├── parameter_contract.yaml
├── model_registry.json
├── model_unit_audit.json
├── MODEL_CARDS.md
└── s1_4_decision.json
```

### 审计结果

- B0、B1、P2、P3-S、P3-O、P4 全部通过参考点、上下界和非有限检查；
- 五个物理模型 NumPy/Torch 最大绝对误差均为 0（本机 float64 审计）；
- 所有自由参数的 Torch 自动梯度均为有限值，并通过中心有限差分相对误差阈值 `2e-5`；
- 分量满足 `total_dermal_absorption = baseline_absorption + hemoglobin_absorption`；
- 三个波段集合分别选择 31、29、24 个波段，四个 FWHM 假设输出均有限；
- B0 缺少显式 Train 平均谱时硬失败；逐图自由尺度硬失败；
- S1-4 未读取 `region_spectra.parquet`，真实区域光谱访问计数为 0；Test 访问计数为 0；
- 全套 `tests/skin_optics_hsi`：`33 passed`；源码与脚本 compileall 通过。

关键哈希：

```text
s1_4_model_registry_v1.yaml cb812d097f0211206aee0ae1a414be9a2f8014c5c0c8e30650ee5ba8e06f8917
model_registry.py           94f8a744df45c1c4566dc8e3774759f62235f67a4e46f42ce6dcd23a3137b050
skin_forward.py             672dd2113c6aa5a9a08589900f21d96c6e969cb2116865e2c78294b8ff20c62d
s1_spectral_sensitivity.py  228c89a023a96f870204c1166c81d9947e832e39f8eed9dcf5a6cc9d71f86973
s1_model_audit.py           647d128f5562d92d5ec7682e6bd861ec71be6371346f0ef2df0eb367c5d2bee9
audit_s1_4_models.py        fffc0c0d6e6ad48575bcffe50ba34091b8993262e98f6720be4388f7d4a2f442
test_s1_models.py           b970c27f3a181a8dcf60d8684dc4205be2a6ead9c54d5bcff0618ad3e9891c32
s1_4_v1/s1_4_decision.json  47b22d5e7bb2c596371fe19a648ee3c4871458a38c3932ad46dd96608e834c72
s1_4_v2/s1_4_decision.json  c11262330216e1bee4628b4111953355f2ab5d6fbba6779c29bab862ec383c21
```

最终状态为 `PASS_FOR_S1_5`，`next_stage_allowed=true`，`authorized_next_stage=S1-5`。S1-5 的主入口只包含 B0/B1/P2/P3-S；P3-O/P4 未获自动激活授权。

## S1-4R：修订候选实现与纯合成审计

状态：`PASS_FOR_REVISED_TRAIN_DEVELOPMENT`；`authorized_next_stage=S1-5R`，`s1_6_allowed=false`。

### 实现范围（2026-09-08）

新增并冻结以下独立 v3 文件，不替换旧 S1-4/S1-5 模型：

```text
configs/skin_optics_hsi/s1_4_revised_registry_v3.yaml
src/skin_optics_hsi/s1_revised_forward.py
src/skin_optics_hsi/s1_proxy_inverse.py
scripts/skin_optics_hsi/audit_s1_4_revised_models.py
tests/skin_optics_hsi/test_s1_revised_models.py
```

注册表包含 B0-R/B0-S、B2-PCA/B2-RPCA、D1-M/D2-MH/D3-MHG/D3-MHO、K2-MH-KM/K3-MHS-KM 和 deferred 的 T2-MH-RTE。实现完成了共同中心化 log-ratio、解析 `a_obs`、包装 Hb 固定基底、D 系列加权线性反演、受试者隔离 PCA、修正 K–M 系数映射及 K2/K3 有界数值反演。公开接口同时返回原始反射率、中心化谱形、固定组件、证据层级和 provenance。D 系列主解析解不使用硬边界；K 系列保留注册的物理/数值边界。严格 RTE 因缺少采集观测算子而 fail-closed。

### 运行与版本记录

首次目录 `s1_4_revised_v3_synthetic` 在 K2 PyTorch 路径中因固定 `S_amp` 仍是 Python 标量、无法执行张量广播而失败，目录保留且没有正式 JSON 产物。修复张量化后 r2 完成初步六项检查；r3/r4 增加完整角点、梯度、PCA 隔离、源码哈希和阶段授权；r5 补齐注册波段子集、负吸收拒绝及 PyTorch K–M 边界。S1-5R 敏感性复核又发现需显式验证 `D_v=0` 无包装极限，因此 r6/r7 继续补齐包装函数、K2 无包装前向和散射覆盖参数的真实生效检查。最终 r7 显式继承 r6：

```powershell
python scripts/skin_optics_hsi/audit_s1_4_revised_models.py `
  --output-dir "data\processed\HyperSkin_Stage1_v1\models\s1_4_revised_v3_synthetic_r7" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\models\s1_4_revised_v3_synthetic_r6\s1_4r_decision.json"
```

### r7 合成结果

| 检查 | 结果 |
|---|---:|
| 血管包装 `D -> 0` 与强吸收极限 | PASS；强吸收 `C_vp*mu_a` 极限误差 0 |
| D2 固定基底 | rank 2；condition 4.241619 |
| D2 合成恢复 | 最大参数误差 `7.11e-9`；weighted RMSE `6.57e-10` |
| D2 解析/数值反演 | 最大参数差 `7.11e-9` |
| 注册波段子集 | 去 400/700 nm 后 29 波段可直接前向与反演；最大参数误差 `7.11e-9` |
| NumPy/PyTorch 前向 | D2/K2 最大差均为 0 |
| 自动梯度/有限差分 | D2 `4.42e-9`；K2 `1.76e-7` |
| K–M 边界角点 | K2 4 个、K3 8 个全部有效；最小 `S_KM=0.543273 mm^-1` |
| K2 合成有界恢复 | theta `[0.21,0.018]` 回收到 `[0.210000000006,0.017999999998]`；log-RMSE `1.39e-11` |
| K2 无包装 Hb 极限 | `D_v=0` 前向反射率全为正且有限 |
| PCA 受试者隔离 | held-out subject 未进入训练 ID 或基底拟合 |
| T2-MH-RTE | 按合同拒绝运行 |

数据访问计数为 Validation 0、Test 0、原始 HSI 内容 0；本次也没有读取 `region_spectra.parquet`。r7 decision 显式继承 r6 decision，r6 SHA-256 为 `cd33bfbc1bac1608fb5d178f27d5cbf291354fc88b1ad24f041685f3633f51df`。

```text
registry                              46f0be41dd8c2e728da4ddc62660fb6b231d71e2695d6e508e3dd6a9b8e30027
production_10nm optical asset         30de54c4bd5af2453a32490559c96e36854b86b31f65336ab700cf32de0488f6
reference_1nm SRF sensitivity asset   6dc1350517a45f5e13a06a025be59921e0c2f398d022b93bf2fd6bf2293feef9
synthetic_audit_checks.json           46093e24c8545a3ab58606b9d6614a9efa953f2edf6c7b82879db96844ecba66
provenance.json                       67bd1c59b484f32b23ed779c36a14d10ed70f9a83a80c46d86926e78e869d198
s1_4r_decision.json                    fd214b5dac764c5359aa92d6cce04b9a6aa27760b0dfad27f574c1f7a1cfcf26
```

全套回归测试最终为 `56 passed`，源码和脚本编译通过。合成 PASS 只说明公式按注册实现、数值路径一致且在自生成数据上可恢复；真实覆盖性与稳定性由后续 S1-5R v7 决定。

## S1-5：Train 开发与诊断

状态：`S1_5_COMPLETE_REVISE_BEFORE_S1_6`；`next_stage_allowed=false`，`P3-O_activated=false`，`P4_activated=false`。

### 实现与数据边界（2026-09-08）

新增配置与入口：

```text
configs/skin_optics_hsi/s1_5_train_development_v1.yaml
src/skin_optics_hsi/s1_inverse.py
src/skin_optics_hsi/s1_train_development.py
scripts/skin_optics_hsi/run_s1_5_train_development.py
tests/skin_optics_hsi/test_s1_inverse.py
```

实现只允许从 `region_spectra.parquet` 读取过滤后的 Train 行，拒绝 Validation/Test/原始 HSI 内容。主分析固定为 44 名 Train 受试者的 88 条 `neutral/front` 左右脸颊光谱，比较 B0/B1/P2/P3-S；P3-O/P4 只允许由预定义 Train 残差证据触发。每条主光谱使用 16 个确定性 Halton 初值，目标函数为 pseudo-Huber log-reflectance 与 SAM；诊断包括 Jacobian/SVD、局部协方差、边界、近最优解聚类、profile objective、经验扰动、表情/方向/区域稳定性，以及波段、中心偏移、假定 FWHM、统计量和零侧别增益敏感性。固定侧别 log-gain 只从 Train 主脸颊估计一次，图像左/右侧分别为 `+0.128799/-0.128799`；没有逐图自由曝光尺度。

正式运行命令：

```powershell
python scripts/skin_optics_hsi/run_s1_5_train_development.py `
  --config "configs\skin_optics_hsi\s1_5_train_development_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\train_development\s1_5_v2" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\train_development\s1_5_v1\s1_5_decision.json"
```

v1 已完成核心拟合，但审计合同不完整：缺少零侧别增益敏感性、完整 Jacobian 奇异值/局部协方差持久化、显式 `fit_failures.json`、物理模型未达到 B0 的阻断 flag，以及源码和依赖环境哈希。v1 目录完整保留，没有覆盖。v2 不改变模型公式、参数范围、数据范围、主分析人群或模型选择逻辑；其 decision 通过 `supersedes` 记录 v1 decision 路径和 SHA-256 `a7b215abab3de7a828f4ecff67191be4754d59eb6ab72913696a28267856edc2`，实测匹配。v1/v2 的 log-RMSE、SAM 和边界主结果完全一致；多解比例在 v2 改为只对近最优解聚类，因此数值按修正后的口径报告。

### 主结果

| 模型 | median log-RMSE | median SAM | 任一参数触边界比例 | 多个近最优解簇比例 |
|---|---:|---:|---:|---:|
| B0 | 0.1660 | 0.0876 | 0.000 | 0.000 |
| B1 | 0.6385 | 0.2967 | 0.023 | 0.000 |
| P2 | 0.6532 | 0.2685 | 0.898 | 0.193 |
| P3-S | 0.6141 | 0.2672 | 1.000 | 0.136 |

非物理受试者留一均值基线 B0 明显优于全部物理模型。最佳物理候选 P3-S 的 median log-RMSE 是 B0 的 3.699 倍，触发 `physical_models_fail_to_match_B0=true`。P2 的 `Hb_absorbance_proxy`、`M_absorbance` 边界比例分别为 73.9% 和 26.1%；P3-S 的 `S_amp` 100% 位于 `1.0 mm^-1` 下界。P3-S 相对 P2 的受试者中位 log-RMSE 改善仅 5.51%，加入散射 nuisance 后 H 的中位归一化绝对漂移为 0.140，说明当前 H 语义不稳。

profile 进一步显示 P2 的 H 目标面很平，90.9% 受试者的 profile 最小值位于外侧网格；P3-S 的 `S_amp` 为 100%。虽然 880/880 次经验扰动复拟合成功且局部漂移较小，但边界坍缩会机械地产生“小漂移”，不能把它解释为参数可辨识。P2/P3-S 仍存在强逐波长结构残差；去趋势 P2 残差与 sO2 对比的中位相关为 -0.0362，一致受试者比例为 0，未达到 `|median correlation|>=0.50` 且一致比例 `>=0.70` 的触发条件，所以 P3-O/P4 均未激活。

全 31 波段、去 400/700 nm 端点、去高曲率候选、中心偏移 `-5` 至 `+5 nm`、假定 Gaussian FWHM `0/5.5/10/20 nm`、截尾均值和固定侧别增益置零敏感性共 1232 次拟合全部成功，但没有任何结果足以将物理候选提升到 B0 水平。置零侧别增益时，P2/P3-S 的 median log-RMSE 分别为 0.6577/0.6144；相对主拟合的 M/H 中位归一化位移分别约为 0.0072/0 和 0.0051/0.0086，说明侧别校正不是本次主要失败原因。有效 SRF 仍缺失，因此绝对生理量和跨设备解释仍受限。

### 审计、测试与门控结论

v2 生成 22 个带 SHA-256 的正式输出，逐一复算后 0 个不匹配；`fit_failures.json` 中主拟合、条件诊断、扰动和敏感性失败清单均为空，profile 非收敛数为 0。`input_audit.json` 固化了配置、区域光谱、S1-3/S1-4 decision、模型注册、波段诊断、五个实现源码及 Python/NumPy/Pandas/SciPy/Torch 版本。访问计数为：Validation 0、Test 0、原始 HSI 内容 0。

```text
python -m pytest tests/skin_optics_hsi -q
40 passed in 9.93s

python -m compileall -q src/skin_optics_hsi scripts/skin_optics_hsi
exit code 0
```

本次 Train 证据不支持把 P2 或 P3-S 提交 Validation，也不能用“数值优化成功”替代模型覆盖性和参数可辨识性。正式 decision 为 `REVISE_BEFORE_S1_6`。其要求的前向式、吸收/散射组合、基线/路径长度参数化及单位复核随后已在 S1-4R 完成，并在 S1-5R v7 重新接受 Train 检验；后续结论见下一节。

## S1-5R：修订候选 Train-only 开发

历史 v7 解释层合并门控状态：`REVISE_OR_STOP`；`next_stage_allowed=false`，`selected_proxy_model=null`，`selected_semi_mechanistic_model=null`。该状态保留，不覆盖；后续表示层重判见本节末尾。

### 实现与数据隔离（2026-09-09）

新增：

```text
configs/skin_optics_hsi/s1_5_revised_train_v3.yaml
src/skin_optics_hsi/s1_revised_train.py
scripts/skin_optics_hsi/run_s1_5_revised_train.py
scripts/skin_optics_hsi/build_s1_3_train_only_cache.py
tests/skin_optics_hsi/test_s1_revised_train.py
```

实现按受试者留一构建几何均值参考谱、侧别 log-gain、B2-PCA/B2-RPCA 基底和 K2 全局 `g_system`；被留出受试者不参与对应折的任何校准。主分析固定为 44 名 Train 受试者、88 条 `neutral/front` 左右脸颊光谱。比较 B0-S/B2-PCA/D1-M/D2-MH，D3-MHG 为必做混杂压力测试；B0-R/B2-RPCA/K2-MH-KM 为原始尺度组；D3-MHO 与 K3-MHS-KM 只按预定义残差/覆盖性规则触发。另运行 440 次 D2 经验扰动、873 条可用条件/区域压力拟合，以及波段、黑色素幂次、包装直径、sO2、假定 FWHM、中心偏移、表皮厚度和散射幅度敏感性。

原 `region_spectra.parquet` 只有一个同时含 Train/Validation 的 row group。虽然谓词过滤只向模型返回 Train 行，为使存储层也可审计，新增一次性 Train-only 派生缓存：

```powershell
python scripts/skin_optics_hsi/build_s1_3_train_only_cache.py `
  --source "data\processed\HyperSkin_Stage1_v1\region_spectra\region_spectra.parquet" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\region_spectra\train_only_v1"
```

缓存包含 1056 行、44 名受试者、5 个 row group；每个 row group 的 split 最小值和最大值均为 `train`。最终 runner 在加载前逐 row group 执行硬检查，混合输入会直接拒绝。缓存 SHA-256 为 `7aca06c8607727190cfff548ea3cf40bd2ddae94d3e3e8761fec707c0847f2fa`，provenance SHA-256 为 `975f6700af38fdbc2d56a03dddeb92044f1a6199e8436c17ed4af4ae5df28e5f`。

### 版本记录与正式运行

- `s1_5r_v3`：主拟合、扰动和压力拟合完成后，中心偏移 `-5 nm` 将 400 nm 端点移到资产覆盖外的 395 nm，安全停止；目录保留。
- `s1_5r_v4`：改为在去端点波段执行中心偏移后完成；随后审计发现 K2 散射敏感性覆盖值未传入前向模型，结果保留但不作最终敏感性证据。
- `s1_5r_v5`：加入无包装 Hb 极限后，被仍要求 `D_v>0` 的全局校验安全停止；目录保留。
- `s1_5r_v6`：修复后完整运行，主结论与 v4 一致；但输入仍来自混合 row-group 文件的谓词过滤。
- `s1_5r_v7`：使用物理隔离 Train-only 缓存完成最终运行，显式继承 v6 decision；这是本阶段正式结果。

```powershell
python scripts/skin_optics_hsi/run_s1_5_revised_train.py `
  --config "configs\skin_optics_hsi\s1_5_revised_train_v3.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\train_development\s1_5r_v7" `
  --supersedes-decision "data\processed\HyperSkin_Stage1_v1\train_development\s1_5r_v6\s1_5r_decision.json"
```

### 主结果

| 模型 | median shape log-RMSE | median raw log-RMSE | median SAM | 任一边界比例 |
|---|---:|---:|---:|---:|
| B0-S | 0.08552 | 0.08552 | 0.08791 | 0 |
| B2-PCA | 0.03671 | 0.03671 | 0.03763 | 0 |
| D1-M | 0.04178 | 0.04178 | 0.04231 | 0 |
| D2-MH | 0.04015 | 0.04015 | 0.04081 | 0 |
| D3-MHG | 0.03133 | 0.03133 | 0.03154 | 0 |
| B0-R | 0.08552 | 0.16088 | 0.08791 | 0 |
| B2-RPCA | 0.04605 | 0.04637 | 0.04660 | 0 |
| K2-MH-KM | 0.26254 | 0.27180 | 0.21505 | 1.000 |

D2 在 88/88 条主光谱上优于 B0-S，median shape log-RMSE 比值为 0.4695；相对 B2-PCA 的二维可解释变化保留率中位数为 0.8831。固定 M/H 基底条件数为 4.2416，最大逐波段绝对中位残差为 0.0422，经验扰动后的 M/H 最大中位位移为 0.1265 IQR。这说明修订的中心化 proxy 明显改善了旧 P2 的覆盖性和边界问题。

但 D3-MHG 揭示 M 的语义不稳定：D2 与 D3 的 M 受试者排序 Spearman 仅 0.1944，M 位移为 1.1958 个 D2 IQR；Hb 的对应值为 0.9143 和 0.2140。D3 误差继续下降并不能挽救这一点，因为 `q_tilt` 与 M 固定基底高度相关，结果说明 D2 中的 M 主要吸收了平滑宽带倾斜。proxy 门的 8 项检查中，拟合比例、误差比、物理增益保留、条件数、扰动和残差 6 项通过，M/H 排序与位移 2 项失败。

光谱敏感性进一步限定 Hb 解释：去 400--440 nm 后 Hb 排序相关降至 0.5526、位移 0.7632 IQR；`D_v=44 um` 时相关为 0.3226、位移 1.2195 IQR；无包装极限时相关为 0.7004、位移 0.4127 IQR。sO2 0.41--0.60、假定 FWHM 0--20 nm 和中心偏移 -5--+5 nm 内的排序较稳定。因此主要风险不是小幅波长中心或带宽假设，而是 Soret 波段与血管包装假设决定了 Hb proxy 的分配。

K2 在原始尺度上仅 20.45% 光谱优于 B0-R，median raw log-RMSE 是 B0-R 的 1.6895 倍，shape 误差是 D2 的 6.538 倍；`M_epi_OD` 100% 位于下边界邻域。所有预注册 K2 物理敏感性中，最佳 median raw log-RMSE 仍为 0.18583，高于 B0-R 的 0.16088；最小边界比例仍为 77.27%。K2 的四项半机制门全部失败，K3-MHS-KM 因此未触发。D2 残差与氧合对比的受试者中位相关为 -0.0473，也未触发 D3-MHO。

### 审计与决策

v7 的 16 个输出哈希逐一复算，全部匹配。`fit_failures.json` 为 0；v7 decision 的 `supersedes` 指向 v6 decision，SHA-256 `fe24bcf61505453c0acc3e7cd33a57a5dbd1233aab914b8ec01b9f1d1c6e431a` 已复核。主要产物：

```text
train_evidence.json                    9cd5af87478d412ca47eb16c418a0be080857ff4aed00e7459c8252be9838f46
input_audit.json                       8eb5b20bc5a8a9ea1647029c1132a0bac35d5c3c03fbf6d07382462179f8ccd7
s1_5r_decision.json                    4a9ba930ce0c6a35007f7536e6b3a3c66f1480037f1d7f58dfa8ec24a54b1897
```

Validation 行、Test 行和原始 HSI 内容访问计数均为 0；正式模型输入仅为物理隔离的 Train-only 派生缓存。v7 的历史结论为 `REVISE_OR_STOP`。它不是说 D2 没有光谱拟合价值，而是说旧门控不能把其中两个系数同时解释为稳定的 M/H-sensitive proxy。K2 在广泛物理敏感性下仍失败，不建议继续修补当前半机制公式。按当时的解释层合并门，S1-6 未授权；随后研究目标被纠正为表示层可行性，旧 decision 原样保留并另行重判。

### S1-5R 表示层重判（2026-09-09）

重判只复用 v7 已有 Train-only 证据，不读取 Validation、Test 或原始 HSI，不覆盖 v7。新增：

```text
configs/skin_optics_hsi/s1_5r_representation_v1.yaml
scripts/skin_optics_hsi/reclassify_s1_5r_representation.py
data/processed/HyperSkin_Stage1_v1/train_development/s1_5r_representation_v3/
```

正式 decision 为 `REPRESENTATION_READY_FOR_S1_6`、`next_stage_allowed=true`。D2-MH 的 Train median shape log-RMSE 为 0.04015，D3-MHG 为 0.03133，B2-PCA 为 0.03671；这些证据证明 Train 上存在可提交的低维表示候选，但不选择最终维度，也不验证 M/H 生理身份。提交 S1-6 的主候选为 D1-M、D2-MH、D3-MHG，对照为 B0-R、B0-S、B2-PCA、B2-RPCA；K2 维持淘汰。

## S1-6：Validation 选择与冻结

状态：`S1_6_FROZEN_FOR_S1_7`；`next_stage_allowed=true`，`selected_representation_model=D2-MH`，`selected_representation_dimension=2`。

### 实现与冻结规则（2026-09-09）

新增：

```text
configs/skin_optics_hsi/s1_6_validation_freeze_v1.yaml
src/skin_optics_hsi/s1_validation_freeze.py
scripts/skin_optics_hsi/run_s1_6_validation_freeze.py
tests/skin_optics_hsi/test_s1_validation_freeze.py
```

选择规则在计算 Validation 光谱指标前写入配置：主选择候选为 D1-M、D2-MH、D3-MHG 和 B2-PCA；B0-S/B0-R 是零维基线，B2-RPCA 是原始尺度诊断，K2/K3 与未触发的 D3-MHO 不允许在 Validation 恢复。候选必须完成 6/6 主域拟合、参数有限、设计满秩、至少 2/3 受试者优于 B0-S、受试者中位误差比不高于 0.90，且最大逐波段中位 shape 残差不高于 0.15。通过后，在最佳受试者中位 shape log-RMSE 的 10% 容差内优先最低维；同维按冻结顺序选择。

正式运行前曾执行一次仅用于核对 Parquet schema、split/subject/ROI 数量的元数据检查。该命令在内存中加载了包含 Train/Validation 的派生 Parquet，但没有打印、汇总或据此修改任何反射率数值或选择阈值；此访问发生在用户明确授权 S1-6 之后，作为透明边界记录。正式 runner 随后记录 1056 条 Train 派生行、72 条 Validation 派生行，模型选择只使用其中 6 条 `primary_validation + neutral/front + bilateral cheeks`。

首次 CLI 启动因入口未将仓库根加入 `sys.path`，在模块导入阶段以 `ModuleNotFoundError: No module named 'src'` 停止；未创建输出目录，也未读取 Validation。按现有脚本模式补齐路径初始化后正式运行：

```powershell
python scripts/skin_optics_hsi/run_s1_6_validation_freeze.py `
  --config "configs\skin_optics_hsi\s1_6_validation_freeze_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\freeze\s1_6_v1"
```

### Validation 主域结果

| 候选 | 维度 | 受试者中位 shape log-RMSE | 相对 B0-S 比值 | 优于 B0-S 的受试者 | 结果 |
|---|---:|---:|---:|---:|---|
| D1-M | 1 | 0.03427 | 0.4333 | 3/3 | eligible |
| D2-MH | 2 | 0.03342 | 0.4226 | 3/3 | eligible，最终冻结 |
| D3-MHG | 3 | 0.03064 | 0.3874 | 3/3 | eligible，绝对误差最低 |
| B2-PCA | 2 | 0.03594 | 0.4544 | 3/3 | eligible，对照 |

D3-MHG 给出最低绝对误差 0.03064，10% 近最佳上限为 0.03370；D2-MH 的 0.03342 位于该区间，而 D1-M 和 B2-PCA 超出。因此按预注册低维优先规则冻结 D2-MH。选择依据是重构与复杂度，不是因为其参数名包含 M/H。

Validation 只有 3 名受试者，因此这里只作候选选择和冻结，不计算或宣称稳定的显著性结论、总体效应大小或跨人群泛化；独立留出判断仍需 S1-7。

冻结的主表示为：

```text
theta_repr = [delta_M_OD, delta_Hb_OD]
theta_aux  = [a_obs]
```

两个 `theta_repr` 元素是固定谱形坐标；`a_obs` 是完整光谱重构所需的解析 log 幅度。阶段一不主张它们分别是真实黑色素和血红蛋白浓度。完整 Train 一次性冻结的侧别 log-gain 为 image-left `+0.1287991441`、image-right `-0.1287991441`，Validation/Test 均禁止重估。

### 压力域、审计与冻结产物

45 条可用 Validation 压力光谱仅用于非阻断报告，D2-MH 总体中位 shape log-RMSE 为 0.04253，16 个表达/方向/区域分层中位数范围为 0.02546–0.05755。`p016_smile_right` 按既有 mask 决策保持不可用。主域和压力域拟合失败均为 0。

正式目录：

```text
data/processed/HyperSkin_Stage1_v1/freeze/s1_6_v1/
```

`frozen_stage1_spec.yaml` 不含 `null`，并以 SHA-256 固化 `frozen_calibration.npz`、模型注册、上游 decisions、选择门和目标域。decision 登记的 14 个前置输出、output manifest 登记的 15 个文件（含 decision）均复算匹配。访问计数为 Validation 72 条派生行、Test 0、原始 HSI 内容 0。

```text
candidate_summary.json       9954c23ac067e743691f47379bf3da1febb9b49ca8c7a9de59e44750aefe381c
frozen_calibration.npz       6ec87a3448cab24acf5def91aca3868af728dd022bfc582a30567dcd48b0a320
frozen_stage1_spec.yaml      72db6a7347b12f2e9c520e81187a409d4e9ac5de31a32c8b7e93ec7813ba16d4
```

验证结果：

```text
python -m pytest tests/skin_optics_hsi -q
60 passed in 12.05s

python -m compileall -q src/skin_optics_hsi scripts/skin_optics_hsi
exit code 0
```

S1-6 只完成表示模型选择和冻结，不是阶段一最终 Test PASS，也不是 M/H 解释层 PASS。它只授权使用当前冻结规范执行一次 S1-7。

## S1-7：正式 Test 与阶段决策

状态：`STAGE1_COMPLETE / PASS`；正式 Test 已使用 1/1 次，不得覆盖或作为首次独立 Test 重跑。

### 实现与锁前冻结（2026-09-09）

新增：

```text
configs/skin_optics_hsi/s1_7_formal_test_v1.yaml
src/skin_optics_hsi/s1_formal_test.py
scripts/skin_optics_hsi/s1_7_mask_rgb_worker.py
scripts/skin_optics_hsi/run_s1_7_formal_test.py
tests/skin_optics_hsi/test_s1_formal_test.py
```

S1-7 使用独立 inference-only 入口，没有放宽 S1-2/S1-3 开发入口的 Test 硬隔离，也没有修改被 S1-2 provenance 锁定的源码。正式配置在首次读取 Test 内容前固化以下门：主脸颊至少 6/8 可用且每名受试者至少 1 条；D2-MH 至少在 2/3 受试者上优于 B0-S；受试者中位 shape log-RMSE 比值不高于 0.90；最大逐波段绝对中位 shape 残差不高于 0.15；D2 拟合失败率为 0；参数必须有限且设计矩阵满秩。最多两条脸颊缺失只是预定义缺失处理，不允许人工补 mask 或修改冻结像素阈值。

runner 在输出目录建立前逐一核验：S1-6 decision 和全部冻结输出哈希、`frozen_stage1_spec.yaml`、`frozen_calibration.npz`、模型注册、S1-2 mask 配置/实现/检查点、S1-1 `transpose`、S1-0 manifest 和波长证据 amendment。只有全部通过后才创建 `FORMAL_TEST_LOCK.json`；一旦建锁即登记 Test 使用 1/1 次。

锁前出现过两次不涉及 Test 内容的安全停止：首次 CLI 因入口未加入 `src` 搜索路径，在导入阶段停止；修复后 preflight 发现当前数据合同哈希不同于历史 S1-2 provenance。核对既有 `wavelength_evidence_amendment_v1.json` 后确认，这是 S1-2 之后“官方代码确认 400:10:700 nm”造成的纯证据升级；amendment 明确把旧哈希 `a3f0c054...` 连到当前哈希 `96f2ea58...`，且声明 RGB/HSI、split、mask 和 ROI 均未改变。runner 因而改为核验这条 amendment 哈希链，而不是错误要求历史 provenance 等于更新后的合同。两次停止均未创建正式输出目录，Test RGB/HSI 内容访问为 0。

最终 preflight：

```text
status=PREFLIGHT_PASS_NO_TEST_CONTENT_ACCESSED
config_sha256=aba7245117de50ebed5c48fc3d4beef03ced3b13b0230c87db3e1707bfd622ab
test_manifest_rows_seen_as_metadata=24
test_rgb_files_opened=0
test_hsi_files_opened=0
selected_model=D2-MH
```

测试与编译：

```text
python -m pytest tests/skin_optics_hsi -q
63 passed in 10.75s

python -m compileall -q src/skin_optics_hsi scripts/skin_optics_hsi
exit code 0
```

### 唯一正式运行

```powershell
python scripts/skin_optics_hsi/run_s1_7_formal_test.py `
  --config "configs\skin_optics_hsi\s1_7_formal_test_v1.yaml" `
  --output-dir "data\processed\HyperSkin_Stage1_v1\formal_test\s1_7_v1"
```

正式锁建立后共读取 24 个 Test RGB 和 24 个 Test HSI；Train/Validation 文件读取为 0。没有重估 Train 参考、侧别 gain、PCA 或任何全局量，没有重选候选、修改门或人工修补 Test mask。24 个 RGB worker 全部成功；自动 mask 为 21 PASS、3 REVIEW、0 FAIL。三个 REVIEW 分别为 `p035_smile_right` 前额不足、`p039_neutral_right` 前额不足和 `p039_smile_right` 整体皮肤连通性提示，均不影响 `neutral/front` 主脸颊。内部 `test_mask_contact_sheet.png` 已复核，五官排除、主脸颊位置和侧脸可见侧与冻结协议一致；由于这些 Test 受试者不在发布许可名单，该图只能内部 QC，不能出版。

### 正式主域结果

| 受试者 | B0-S shape log-RMSE | B2-PCA | D2-MH | D2 是否优于 B0-S |
|---|---:|---:|---:|---|
| p002 | 0.13473 | 0.04953 | 0.05492 | 是 |
| p035 | 0.09260 | 0.04966 | 0.05896 | 是 |
| p036 | 0.06861 | 0.04001 | 0.03852 | 是 |
| p039 | 0.06618 | 0.04309 | 0.05191 | 是 |

正脸无表情双侧主脸颊 8/8 可用，4/4 名受试者的 D2-MH 均优于 B0-S。受试者中位 D2-MH shape log-RMSE 为 0.053413，B0-S 为 0.080605，比值为 0.662650；最大逐波段绝对中位 shape 残差为 0.084475；拟合失败率为 0，参数全部有限，设计矩阵全部满秩。9 项正式 gate 全部通过。

B2-PCA 在 p002、p035、p039 上略优于 D2，在 p036 上略差；它只用于描述冻结数据驱动对照，不能在 Test 后取代 S1-6 已冻结的 D2。16 个笑脸/侧脸/其他区域压力分层为非阻断报告，D2-MH 分层中位 shape log-RMSE 范围为 0.03646--0.06464。该结果不授权把主域 PASS 外推为表情或姿态稳定。

### 最终判定与审计

正式 `STAGE1_DECISION.json`：

```text
status=STAGE1_COMPLETE
decision=PASS
selected_representation_model=D2-MH
selected_representation_dimension=2
test_access_count=1
next_stage_allowed=true
authorized_next_stage=S2
physiological_M_H_claim=NOT_ASSESSED_IN_STAGE1
```

`output_manifest.json` 登记 332 个文件，逐一复算 SHA-256 为 0 个不匹配；`test_fit_failures.json` 为 0。关键产物：

```text
STAGE1_DECISION.json             5fe3830ac0c5bd79517c5d3cd0ee5732450f97da6f9f0186ade01b1b7d5d4cef
STAGE1_DECISION.md               08d153ddcc035447c2b4224419bd74623f4f91c346285b463e5a6fd96385dcad
test_mask_manifest.parquet       d3b3eb65ebf668f52415b89f526e7a4c67e45965c04dc14cce0bcb600c2b121c
test_region_spectra.parquet      5fa8f5403550fac97f1badaff87c8db61115014f3471dfa5d8210b9422cfe954
test_per_spectrum_fits.parquet   059a0ab72fca6b9f069176015bc68f3907c1e15abac3015ac4e986ae0e2efa15
output_manifest.json             fc285480f99f255578fa1185aa43a72b9aeb0f5b1731e5fcdddf4388cbe1c0e9
```

阶段一 PASS 的严格含义是：冻结的二维 D2-MH 谱形表示在未见 Hyper-Skin 受试者的正脸无表情脸颊上，相对零维 B0-S 保留了可复现的低维重构增益。它不证明两个坐标是真实黑色素和血红蛋白浓度；不解决 D2/D3 坐标混合、31 波段有效 SRF 缺失、无眼镜目标域未验证或跨设备可比性。下一步可以进入阶段二，但只能把 D2-MH 当作冻结 `theta_repr`，解释层结论必须由阶段二另行建立。
