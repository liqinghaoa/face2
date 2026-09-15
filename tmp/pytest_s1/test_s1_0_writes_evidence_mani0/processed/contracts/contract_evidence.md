# Hyper-Skin 阶段一数据合同证据

生成时间：`2026-09-07T18:31:04.199986+00:00`

## 可以确认的事实

- 下载内容是 Hyper-Skin RGB/VIS 发布集；HSI 文件是 MATLAB v7.3/HDF5，数据键为 `cube`。
- 论文与采集文档将发布光谱描述为暗场校正、白参考归一化后的光谱反射率。这里的 `confirmed` 表示文档证据充分，不表示当前文件携带独立校准元数据。
- 发布 VIS 范围为 400–700 nm，共 31 波段；原始 FX10 数据为 400–1000 nm、448 波段，发布数据经过插值。

## 推定和缺失

- 31 个中心波长暂用 `400, 410, ..., 700 nm`。这是由范围、波段数和等间隔描述得到的强推定，正式 Test 前应取得作者或官方代码确认。
- 下载目录未提供 31 波段 SRF/有效带宽、原始 wavelength vector、精确插值函数与端点策略。原始系统约 5.5 nm FWHM 不能直接当作发布 31 波段的有效带宽。
- HDF5 读出后的两个空间轴与 RGB 的唯一映射留给 S1-1 冻结；S1-0 不提前把转置规则写成事实。
- 原始数据只读；极小的 `[0,1]` 越界只在派生数据中按已记录规则裁剪，绝不改写原文件。

## 证据来源

- **confirmed** — 暗场校正和白参考归一化后的光谱反射率  来源：`face2_research\01_Background_and_Literature\Key_Papers\11_Ng_2023_Hyper_Skin\Hyper-Skin_A_Hyperspectral_Dataset_for_Reconstructing_Facial_Skin-Spectra_from_RGB_Images.pdf`（Methods / hyperspectral acquisition and preprocessing）
- **confirmed** — Specim FX10；400–1000 nm；448 波段；约 1.34 nm sampling；约 5.5 nm raw FWHM  来源：`face2_research\01_Background_and_Literature\Key_Papers\11_Ng_2023_Hyper_Skin\SupportingInformation\Appendix A1 - Hyperspectral_Data_Acquisition_Documentation.pdf`（camera and acquisition specification）
- **confirmed** — 发布 VIS 为 400–700 nm、31 波段；由原始波段插值得到  来源：`face2_research\01_Background_and_Literature\Key_Papers\11_Ng_2023_Hyper_Skin\SupportingInformation\Appendix A1 - Hyper_Skin_Datasheets.pdf`（VIS dataset description）
- **inferred** — 31 个中心暂按 400:10:700 nm  来源：`range + band count + regular-grid interpretation`（not explicitly enumerated in supplied documents）
- **missing** — 31 波段有效 SRF/带宽未提供  来源：`downloaded release and supplied documents`（not found）

## 使用限制

数据提供方邮件说明：除 p012、p019、p027 外，不将其他受试者图像用于出版图。所有样本仍可在许可范围内用于内部分析；任何公开材料必须再次核对原许可文本。
