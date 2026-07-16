# EXIF拍摄参数完整性与合理性审计

> 本报告检查元数据字段是否存在、是否可解析、是否满足基础EXIF合法范围，并单独标注设备内统计离群值。统计离群或跨字段不一致是复核线索，不等同于源数据错误。

## 数据概况

- 图片记录：522张；唯一图像ID：522。
- 患者组：505；分组来源：project_patient_group_id; fallback_for_0_unmapped_images。
- 清单字段：42项，其中常规EXIF 19项、拍摄参数 23项。
- 严格42项全部非空的图片：0张。
- 严格23项拍摄参数全部非空的图片：0张。
- 满足研究核心参数规则的图片：515张（98.7%）；至少有一张合格图片的患者组：498组（98.6%）。

## 关键结论

- 两个主要设备组合为 HONOR/BVL-AN00 与 Xiaomi/M2006J10C；多项字段的缺失具有明显设备系统性，不能把缺失简单解释为随机漏采。
- `ISOSpeedRatings` 在全部图片中存在，可作为统一ISO主字段；`ISOSpeed` 是新版补充字段，不宜要求每张图片同时存在两者。
- `FNumber`、`ExposureTime`、`FocalLength` 等核心直接拍摄参数覆盖完整；`ApertureValue`、`SensingMethod`、`DigitalZoomRatio` 等设备依赖字段不适合作为全队列必需条件。
- `ImageDescription` 基本为空，虽然标签行存在，但没有可用于研究的值。
- 正常范围判断同时输出明确非法值与待复核警告；后者包括设备编码、非信息性枚举码、跨字段不一致和设备内统计离群。

- 完全没有非空值的字段：ImageDescription。

## 全部存在且基础取值合法的拍摄参数

以下14个拍摄参数在全部522张图片中均有非空记录，且单字段格式和基础EXIF取值范围检查全部通过。这里的“基础取值合法”不等于字段一定适合直接建模；设备依赖、变量缺乏变异或跨字段不一致仍需按说明处理。

| 字段 | 中文说明 | 本数据取值概况 | 后续研究建议 |
|---|---|---|---|
| `ExposureTime` | 曝光时间，单位为秒，表示快门开启时长 | 0.002222–0.050009 s，中位数0.020 s | 推荐作为快门相关的主变量；比APEX形式更直观 |
| `FNumber` | 光圈F值，反映镜头进光量和景深 | 1.89–2.00，中位数1.90 | 可直接使用，但其取值高度依赖设备型号 |
| `ISOSpeedRatings` | ISO感光度，反映传感器增益 | 50–1600，中位数241.5 | 推荐作为统一ISO主字段；14张设备内高ISO离群图片应保留并复核 |
| `FocalLength` | 实际焦距，通常以毫米为单位 | 1.82–6.67 mm，中位数6.67 mm | 可用于表征视角，但必须结合设备型号分析 |
| `ExposureMode` | 曝光控制方式的EXIF枚举值 | 全部记录且枚举合法，本数据均为0（自动曝光） | 字段没有组内变异，不能单独提供预测信息 |
| `MeteringMode` | 相机测量场景亮度的方式 | 合法值2或3 | 可作为分类变量，使用前应按EXIF标准解码 |
| `Flash` | 闪光灯是否触发及工作模式的位掩码 | 合法值16或24 | 不能按连续数值处理，应解码或作为分类变量 |
| `WhiteBalance` | 白平衡控制方式 | 全部为0（自动白平衡） | 无组内变异，不适合直接作为预测变量 |
| `BrightnessValue` | 设备估计的场景亮度APEX值 | −2.67至10.90，中位数3.055 | 可用于环境亮度分析，但应做设备分层或标准化 |
| `ShutterSpeedValue` | 快门速度的APEX表达 | 0–8.815，中位数5.058 | 字段本身合法，但239张Xiaomi图片与`ExposureTime`换算不一致；建模优先使用`ExposureTime` |
| `SceneCaptureType` | 场景拍摄类型的EXIF枚举值 | 全部为0（标准场景） | 无组内变异，不适合直接作为预测变量 |
| `SubsecTime` | `DateTime`对应的亚秒部分 | 全部为1–9位数字文本 | 主要用于时间精确匹配，不应作为连续拍摄参数直接建模 |
| `SubsecTimeOriginal` | 原始拍摄时间的亚秒部分 | 522张均有合法数字文本 | 与`DateTimeOriginal`组合构成更精确时间戳 |
| `SubsecTimeDigitized` | 数字化时间的亚秒部分 | 522张均有合法数字文本 | 主要用于时间一致性检查，通常不作为影像表型变量 |

其中更适合进入后续跨设备拍摄参数研究的连续或有序核心变量是 `ExposureTime`、`FNumber`、`ISOSpeedRatings`、`FocalLength` 和 `BrightnessValue`。`MeteringMode`、`Flash` 可在正确解码后作为分类变量；`ExposureMode`、`WhiteBalance`、`SceneCaptureType` 在当前数据中没有变异，不能用于解释个体差异。

## 字段覆盖率最低项

| Parameter | Nonblank n | Nonblank rate |
|---|---:|---:|
| ImageDescription | 0 | 0.0% |
| Software | 239 | 45.8% |
| DigitalZoomRatio | 239 | 45.8% |
| ISOSpeed | 279 | 53.4% |
| OffsetTime | 283 | 54.2% |
| OffsetTimeOriginal | 283 | 54.2% |
| ApertureValue | 283 | 54.2% |
| SensingMethod | 283 | 54.2% |

## 存在明确非法值或警告的字段

| Parameter | Invalid n | Warning n |
|---|---:|---:|
| Orientation | 1 | 0 |
| FocalLengthIn35mmFilm | 239 | 0 |
| ExposureBiasValue | 6 | 0 |
| ExposureProgram | 0 | 239 |
| LightSource | 0 | 522 |
| MaxApertureValue | 239 | 0 |
| DigitalZoomRatio | 0 | 239 |
| SensingMethod | 0 | 283 |

## 后续研究建议

1. 建议使用统一核心变量集：设备厂商/型号、原始拍摄时间、曝光时间、FNumber、ISOSpeedRatings、实际焦距、曝光补偿、曝光/测光/闪光/白平衡模式、亮度值和场景类型。
2. `ISOSpeed` 与 `ISOSpeedRatings` 合并为一个ISO变量，以Ratings优先或在一致时互补；不要将新版ISO字段缺失视为病例不合格。
3. 35mm等效焦距为0、最大光圈APEX为0等值应按设备特异的无信息哨兵处理，而不是当作真实0值建模。
4. 对设备系统性缺失字段，若进入模型必须增加缺失指示变量，并在设备分层或敏感性分析中验证；不能直接均值填补后忽略设备来源。
5. 正式建模前优先复核 `parameter_value_issues.csv` 中 severity=invalid 的记录，再评估 review级离群值是否为真实拍摄差异。

## 输出说明

- `parameter_coverage.csv`：42项字段的覆盖、合法和警告统计。
- `device_parameter_coverage.csv`：按相机型号分层的系统性缺失和合法性统计。
- `image_parameter_audit.csv`：522张图片逐例完整性及研究可用性。
- `patient_level_audit.csv`：按项目patient_group_id汇总的患者级可用性。
- `parameter_value_issues.csv`：明确非法、设备编码、跨字段冲突和统计离群明细。
- `parameter_values_long.csv`：逐图片逐参数的原始值、解析值和判断结果。
