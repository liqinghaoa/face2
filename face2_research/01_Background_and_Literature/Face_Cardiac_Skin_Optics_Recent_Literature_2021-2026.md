# Face–Cardiac Skin Optics 近五年相关文献（2021-09-05—2026-09-05）

## 一、检索口径

- **项目主线**：智能手机正面人脸照片 → 标准化与皮肤 mask → 光谱皮肤光学前向模型 → 合成 RGB–M/H 代理图 → 冻结真实人脸推理与全脸融合 → 心功能状态二分类。
- **检索日期**：2026-09-05。
- **时间范围**：2021-09-05 至 2026-09-05；2021 年仅纳入截止日之后发表的文献。
- **主要来源**：PubMed、Crossref/OpenAlex、出版社论文页；优先纳入与“面部影像—心血管表型”“皮肤色素光学分解”“合成生理影像与跨域泛化”直接相关的原始研究，辅以少量路线图/综述。
- **分区与 IF 口径**：统一报告 **2024 JCR JIF 与最佳 JCR 分区**（同一期刊属于多个学科时取最高分区）。会议论文没有 JIF；无法从公开来源核验 2024 JCR 的期刊标为“暂无可核验数据”。IF 是期刊层级指标，不代表单篇论文质量。
- **摘要说明**：以下为基于论文原始摘要的中文转述，并非逐字翻译。

## 二、优先级总览

| 优先级 | 文献                                                          | 与当前项目的直接关系                           |
| --- | ----------------------------------------------------------- | ------------------------------------ |
| A   | Jung et al., 2023, *Journal of Biomedical Optics*           | 与当前 M/H map、前向重建、patch 处理高度同构        |
| A   | Jung et al., 2024, *Computers in Biology and Medicine*      | 以光谱框架生成 M/H 图及可控模拟图                  |
| A   | Jung et al., 2024, *Biomedical Engineering Letters*         | 标准图→交叉偏振图→M/H map，全脸 patch 融合        |
| A   | Lin et al., 2024, *Chinese Journal of Cardiology*           | 大样本人脸照片冠心病分类，直接临床邻域证据                |
| A   | Jimenez-Blanco Bravo et al., 2021, *European Heart Journal* | 心衰、NYHA、智能手机面部视频和色度变化直接相关            |
| A   | Ng et al., 2023, NeurIPS Datasets and Benchmarks            | 人脸 RGB–高光谱配对、28 种相机响应，可支持真实光谱监督/外部验证 |
| A   | Wang et al., 2022, CVPR                                     | 生物物理约束合成人脸生理数据及肤色/光照公平性              |
| B   | Jung et al., 2023, *Journal of Biophotonics*                | 说明少量真实 ground truth 可加强物理自监督 M/H 分解  |
| B   | Jonasson et al., 2023, *Journal of Biomedical Optics*       | 3,809 人在体皮肤吸收/散射参数，可校准前向模型先验         |
| B   | Casado & López, 2023, *IEEE JBHI*                           | 人脸对齐、皮肤 ROI、RGB→生理信号与跨数据集评估          |
| B   | Knorr et al., 2022, *EHJ Digital Health*                    | 面部/全身照片可学习高血压、吸烟、糖尿病等心血管混杂因素         |
| B   | Shi et al., 2025, *Scientific Reports*                      | 眼白血管/色素影像与冠心病风险，提供可解释邻域证据            |
| C   | Jung et al., 2023, *Skin Research and Technology*           | 可控改变 M/H 图并重建图像，适合反事实与敏感性分析          |
| C   | Li et al., 2022, *Physiological Measurement*                | 短时人脸视频 rPPG，主要适用于未来动态扩展              |
| C   | Charlton et al., 2023, *Physiological Measurement*          | PPG 生物物理、肤色偏差和验证规范的背景路线图             |

---

## 三、面部/眼部影像与心血管、心衰表型

### 1. Artificial intelligence model for diagnosis of coronary artery disease based on facial photos

- **文献名**：Artificial intelligence model for diagnosis of coronary artery disease based on facial photos（基于面部照片的冠心病人工智能诊断模型）
- **年份**：2024
- **期刊**：*Zhonghua Xin Xue Guan Bing Za Zhi / Chinese Journal of Cardiology*，52(11):1272–1276；DOI：[10.3760/cma.j.cn112148-20240705-00370](https://doi.org/10.3760/cma.j.cn112148-20240705-00370)，PMID：[39557525](https://pubmed.ncbi.nlm.nih.gov/39557525/)
- **分区**：2024 JCR 暂无可核验分区
- **IF**：2024 JIF 暂无可核验数据
- **文献摘要**：连续纳入 2022–2023 年拟行冠状动脉造影的患者，共 5,974 人、84,964 张面部照片；每人采集正面、左右 60° 侧面及头顶部照片。研究以 VGGFace 约 200 万张图像预训练，再比较 MAE、ViT 与 ResNet。测试集中，预训练 MAE 和 ViT 的 AUC 分别为 0.841 和 0.824，优于随机初始化 ResNet（0.810）及预训练 ResNet（0.816）。结果支持面部照片用于冠心病早筛，但研究为横断面设计，目标是冠心病而非心功能分级，仍需外部验证与偏倚审计。

### 2. Predicting cardiovascular risk factors from facial & full body photography using deep learning

- **文献名**：Predicting cardiovascular risk factors from facial & full body photography using deep learning
- **年份**：2022
- **期刊**：*European Heart Journal – Digital Health*，3(4):ztac076.2780；DOI：[10.1093/ehjdh/ztac076.2780](https://doi.org/10.1093/ehjdh/ztac076.2780)
- **分区**：2024 JCR 暂无可核验分区（会议摘要）
- **IF**：2024 JIF 暂无可核验数据
- **文献摘要**：基于 Hamburg City Health Study 的约 6,500 名参与者，以标准化面部和全身照片、年龄与性别训练 ResNet-18，并使用患者级 5 折验证预测高血压、吸烟和糖尿病。AUC 分别约为 0.711、0.733 和 0.744，均优于只用年龄与性别的逻辑回归。该研究说明照片模型容易学习年龄、体型、吸烟和代谢状态等心血管风险代理，因此当前项目必须把这些变量视为潜在混杂因素，而不能直接把分类性能归因于 M/H-sensitive 表征。

### 3. Prediction of major adverse cardiovascular events in heart failure patients using face recognition: rationale and study design of the CARDIOMIRROR trial

- **文献名**：Prediction of major adverse cardiovascular events in heart failure patients using face recognition: rationale and study design of the CARDIOMIRROR trial
- **年份**：2021（2021-10，位于本次五年窗口内）
- **期刊**：*European Heart Journal*，42(Suppl 1):ehab724.0794；DOI：[10.1093/eurheartj/ehab724.0794](https://doi.org/10.1093/eurheartj/ehab724.0794)
- **分区**：2024 JCR Q1（Cardiac & Cardiovascular Systems；会议摘要/研究设计）
- **IF**：35.7（2024 JIF）
- **文献摘要**：前瞻性、多中心试点研究拟纳入 100 名 NYHA≥II 的 HFrEF 门诊患者；参与者用智能手机每日拍摄 7 秒面部视频，连续 12 个月，通过深度学习分析形态和色度变化。主要终点是这些面部变化与 1 年 MACE（急性失代偿心衰、心肌梗死、卒中或死亡）的关联；次要终点包括 BNP、LVEF、NYHA、生活质量和活动量。该文是研究设计而不是结果论文，但它是“面部色度/形态—心衰状态”最直接的近五年先例之一。

### 4. A deep-learning system for the assessment of coronary heart disease risk via scleral photographs

- **文献名**：A deep-learning system for the assessment of coronary heart disease risk via scleral photographs
- **年份**：2025
- **期刊**：*Scientific Reports*；DOI：[10.1038/s41598-025-33510-9](https://doi.org/10.1038/s41598-025-33510-9)，PMID：[41469469](https://pubmed.ncbi.nlm.nih.gov/41469469/)
- **分区**：2024 JCR Q1（Multidisciplinary Sciences）
- **IF**：3.9（2024 JIF）
- **文献摘要**：研究构建基于巩膜照片的冠心病风险系统，最终分析 500 名参与者的 4,000 张高质量图像，其中 240 例经心血管专科确诊为冠心病。多实例学习模型报告总体准确率 0.891、AUC 0.942；解释分析显示模型主要依赖巩膜血管异常，部分病例也使用色素斑。该研究不是面部皮肤 M/H 映射，但支持“浅表血管与色素空间结构可形成心血管影像表型”的邻域假设。

---

## 四、皮肤光学前向建模、M/H 分解与全脸重建

### 5. Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution

- **文献名**：Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution
- **年份**：2023
- **期刊**：*Journal of Biomedical Optics*，28(3):035001；DOI：[10.1117/1.JBO.28.3.035001](https://doi.org/10.1117/1.JBO.28.3.035001)，PMID：[36992693](https://pubmed.ncbi.nlm.nih.gov/36992693/)
- **分区**：2024 JCR Q2（最佳分区）
- **IF**：2.9（2024 JIF）
- **文献摘要**：作者提出一种以光—组织前向问题约束的深度学习皮肤分析方法。全脸图像被切成 patch，网络输出 melanin、hemoglobin、shading 和 specular map，再通过前向模型重建输入图像，以重建误差间接约束色素分解。30 名受试者上，与 VISIA 系统相比，melanin 和 hemoglobin map 的相关系数分别为 0.932 和 0.857。该工作与当前项目的 patch、linear RGB、M/H 代理图和前向建模结构高度一致，是首要精读文献。

### 6. Spectrum-based deep learning framework for dermatological pigment analysis and simulation

- **文献名**：Spectrum-based deep learning framework for dermatological pigment analysis and simulation
- **年份**：2024
- **期刊**：*Computers in Biology and Medicine*，178:108741；DOI：[10.1016/j.compbiomed.2024.108741](https://doi.org/10.1016/j.compbiomed.2024.108741)，PMID：[38879933](https://pubmed.ncbi.nlm.nih.gov/38879933/)
- **分区**：2024 JCR Q1（最佳分区）
- **IF**：6.3（2024 JIF）
- **文献摘要**：研究引入光谱驱动框架，从皮肤图像生成 melanin/hemoglobin 分布图，并允许通过调整色素参数合成不同皮肤状态。该框架无需人工制作逐像素色素 ground truth，而是通过光谱与前向重建形成训练监督。与 VISIA 比较时，melanin 和 hemoglobin 的相关系数分别为 0.913 和 0.941；重建光谱与色素吸收特性一致，并可按设定幅度生成色素变化图。对当前项目而言，该文直接提示应以光谱误差、RGB 重建误差和参数可辨识性共同验收 M/H map。

### 7. Integrated deep learning approach for generating cross-polarized images and analyzing skin melanin and hemoglobin distributions

- **文献名**：Integrated deep learning approach for generating cross-polarized images and analyzing skin melanin and hemoglobin distributions
- **年份**：2024
- **期刊**：*Biomedical Engineering Letters*，14(6):1355–1364；DOI：[10.1007/s13534-024-00409-9](https://doi.org/10.1007/s13534-024-00409-9)，PMID：[39465115](https://pubmed.ncbi.nlm.nih.gov/39465115/)
- **分区**：2024 JCR Q3（Engineering, Biomedical）
- **IF**：2.8（2024 JIF）
- **文献摘要**：作者先用 CycleGAN、pix2pix 或 pix2pixHD 将普通皮肤图像转换为交叉偏振图像，再用轻量 ResUNet++ 分解 melanin、hemoglobin 和 shading map。测试阶段使用 256×256 patch、15 像素重叠并融合为全脸图；同时进行 sRGB 线性化、皮肤分割及相机/光源前向建模。pix2pixHD 综合表现最佳，与 VISIA 的相关系数约为 hemoglobin 0.923、melanin 0.908。该文的空间处理和全脸融合细节可直接对照当前实现。

### 8. Deep learning-based pigment analysis model trained with optical approach and ground truth assistance

- **文献名**：Deep learning-based pigment analysis model trained with optical approach and ground truth assistance
- **年份**：2023
- **期刊**：*Journal of Biophotonics*，16(12):e202300231；DOI：[10.1002/jbio.202300231](https://doi.org/10.1002/jbio.202300231)，PMID：[37602740](https://pubmed.ncbi.nlm.nih.gov/37602740/)
- **分区**：2024 JCR Q3（最佳分区）
- **IF**：2.3（2024 JIF）
- **文献摘要**：研究把光学前向重建与少量色素 ground truth 结合，以兼顾高分辨率 patch 训练和分解准确度。模型输出 melanin、hemoglobin 和 shading map，并用这些分量重建皮肤图像；加入真实参考后，与 VISIA 的相关性提升至 melanin 0.978、hemoglobin 0.975。该结果说明：纯合成监督之外，引入少量独立光谱/临床仪器标定样本，可能显著增强当前代理图的可解释性和外部锚定。

### 9. Generation of skin tone and pigmented region-modified images using a pigment discrimination model trained with an optical approach

- **文献名**：Generation of skin tone and pigmented region-modified images using a pigment discrimination model trained with an optical approach
- **年份**：2023
- **期刊**：*Skin Research and Technology*，29(10):e13486；DOI：[10.1111/srt.13486](https://doi.org/10.1111/srt.13486)，PMID：[37881042](https://pubmed.ncbi.nlm.nih.gov/37881042/)
- **分区**：2024 JCR Q1（Dermatology）
- **IF**：3.2（2024 JIF）
- **文献摘要**：研究从交叉偏振皮肤图像估计 melanin、hemoglobin 和 shading map，再独立修改 M/H 图并通过前向模型重建可控肤色/色素图像。74 名受试者被扩展为 1,480 个 patch；与 VISIA 的相关系数为 melanin 0.915、hemoglobin 0.931，图像变化与 ITA、melanin index 和 erythema index 呈比例关系。该方法特别适合当前项目开展反事实实验：在保持身份与几何不变时，单独扰动 M 或 H，检验分类输出是否按生理合理方向变化。

### 10. Absorption and reduced scattering coefficients in epidermis and dermis from a Swedish cohort study

- **文献名**：Absorption and reduced scattering coefficients in epidermis and dermis from a Swedish cohort study
- **年份**：2023
- **期刊**：*Journal of Biomedical Optics*，28(11):115001；DOI：[10.1117/1.JBO.28.11.115001](https://doi.org/10.1117/1.JBO.28.11.115001)，PMID：[38078153](https://pubmed.ncbi.nlm.nih.gov/38078153/)
- **分区**：2024 JCR Q2（最佳分区）
- **IF**：2.9（2024 JIF）
- **文献摘要**：研究对 3,809 名受试者开展 475–850 nm 空间分辨漫反射光谱测量，并用含表皮与两层真皮的模型估计在体吸收和约化散射系数。结果给出大样本人群参考范围，同时显示皮肤光学参数会随性别、年龄、BMI、日晒季节、皮肤部位以及硬件和反演算法改变。该文可用于审计当前前向模型的参数范围，避免把单一文献或单一肤色人群的光学常数固定为“普适真值”。

### 11. Hyper-Skin: A Hyperspectral Dataset for Reconstructing Facial Skin-Spectra from RGB Images

- **文献名**：Hyper-Skin: A Hyperspectral Dataset for Reconstructing Facial Skin-Spectra from RGB Images
- **年份**：2023
- **期刊/会议**：NeurIPS 2023 Datasets and Benchmarks Track；论文页：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4c0986bd04d747745beba3752bdf4d9d-Abstract-Datasets_and_Benchmarks.html)，arXiv：[2310.17911](https://arxiv.org/abs/2310.17911)
- **分区**：不适用（会议论文；非期刊 JCR 分区）
- **IF**：不适用
- **文献摘要**：该数据集包含 51 名受试者、330 个 1024×1024×448 的面部高光谱立方体，覆盖 400–1000 nm，并提供由 28 种真实相机响应函数合成的 RGB 图像。作者评估了 RGB→光谱重建模型，表明去除背景后面部皮肤区域的重建更稳定，并在真实智能手机图像上进行了验证。它是当前项目从“仅合成 M/H 监督”升级到“真实 RGB–光谱配对验证”、检验相机响应泛化与建立外部物理锚点的重要公开资源。

---

## 五、合成生理影像、rPPG 与跨相机/肤色泛化

### 12. Synthetic Generation of Face Videos with Plethysmograph Physiology

- **文献名**：Synthetic Generation of Face Videos with Plethysmograph Physiology
- **年份**：2022
- **期刊/会议**：CVPR 2022，pp. 20587–20596；DOI：[10.1109/CVPR52688.2022.01993](https://doi.org/10.1109/CVPR52688.2022.01993)
- **分区**：不适用（会议论文；非期刊 JCR 分区）
- **IF**：不适用
- **文献摘要**：针对 rPPG 数据量小、需同步医疗级真值且肤色分布不均的问题，作者提出受生物物理约束的人脸视频生成方法，可给定参考人脸和目标 PPG 信号合成具有脉搏变化的视频，并控制肤色、姿态和光照。加入合成数据后，rPPG 性能和不同肤色组之间的公平性均有改善。该文与当前项目“由光学前向模型生成受控 RGB–代理参数对”的思想一致，并提示合成数据验收必须同时检查生理保真、跨域收益和肤色分层误差。

### 13. Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces

- **文献名**：Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces
- **年份**：2023
- **期刊**：*IEEE Journal of Biomedical and Health Informatics*，27(11):5530–5541；DOI：[10.1109/JBHI.2023.3307942](https://doi.org/10.1109/JBHI.2023.3307942)
- **分区**：2024 JCR Q1（多个相关学科均为 Q1）
- **IF**：6.8（2024 JIF）
- **文献摘要**：论文系统拆解无监督 rPPG 流程，并在 6 个数据集上比较关键选择。作者提出刚性网格归一化以稳定人脸、动态选择高质量面部区域，以及基于 QR 分解的 OMIT RGB→rPPG 变换以增强压缩伪影鲁棒性；三项改进均提升了人脸 BVP 提取表现。虽然当前研究使用静态照片，该文对人脸对齐、皮肤区域筛选、空间质量加权和跨数据集复现有直接方法学价值。

### 14. Deep learning-based remote-photoplethysmography measurement from short-time facial video

- **文献名**：Deep learning-based remote-photoplethysmography measurement from short-time facial video
- **年份**：2022
- **期刊**：*Physiological Measurement*，43(10):105002；DOI：[10.1088/1361-6579/ac98f1](https://doi.org/10.1088/1361-6579/ac98f1)
- **分区**：2024 JCR Q2
- **IF**：2.7（2024 JIF）
- **文献摘要**：作者提出短时端到端心率估计框架，用带跨层残差结构的 3D 多尺度网络构建自编码器，联合学习面部空间特征和帧间时间关系，并通过融合注意机制提取稳健 rPPG 特征。其目标是减少传统方法对手工 ROI 和简化皮肤反射假设的依赖。该文主要支持未来从静态照片扩展到短视频的研究，不应被当作当前单帧 M/H map 的直接证据。

### 15. The 2023 wearable photoplethysmography roadmap

- **文献名**：The 2023 wearable photoplethysmography roadmap
- **年份**：2023
- **期刊**：*Physiological Measurement*，44(11):111001；DOI：[10.1088/1361-6579/acead2](https://doi.org/10.1088/1361-6579/acead2)，PMID：[37494945](https://pubmed.ncbi.nlm.nih.gov/37494945/)
- **分区**：2024 JCR Q2
- **IF**：2.7（2024 JIF）
- **文献摘要**：多学科专家从传感器设计、信号处理、临床应用和研究规范总结 PPG 发展方向。文中强调光源波长、解剖部位、运动伪影、肤色、年龄、肥胖、温度和静脉搏动都会改变信号；真实世界验证应覆盖异质人群、开放数据与算法、信号质量控制、不确定性和公平性。对当前项目最重要的启示是：H-sensitive 图只能作为受光学条件影响的代理表征，必须通过相机/光照/肤色分层和独立生理参照验证，不能直接等同于血红蛋白浓度或心功能。

---

## 六、本轮检索形成的初步判断（供后续创新分析使用）

1. **当前路线并非孤立思路**：Jung 团队 2023–2024 年系列论文已经证明“皮肤光学前向模型 + patch 网络 + M/H/shading 分解 + 全脸拼接”具有可行性，因此创新点不能仅表述为“首次从 RGB 生成 M/H map”。
2. **项目的潜在独特性更可能位于跨任务链**：把受控、合成监督得到的 M/H-sensitive 代理图冻结后用于心功能状态分类，并与 RGB、M-only、H-only 和融合结构进行患者级 OOF 比较，这一完整链条在本轮检索中尚未发现完全同构的成熟临床研究。
3. **最需要补强的是代理图的独立锚定**：现有最强邻域证据通常与 VISIA、交叉偏振图像、高光谱数据或漫反射光谱比较；当前项目若只有合成域验收与分类 AUC，仍不足以证明真实人脸上的 M/H-sensitive map 具有稳定的光学生理含义。
4. **分类增益不能直接解释为心功能机制**：面部模型能够学习年龄、性别、肤色、BMI、糖尿病、高血压、吸烟、皱纹、眼袋、毛发和成像条件。需要病例级混杂审计、遮挡/区域消融、相机与光照分层、反事实 M/H 扰动及外部验证。
5. **下一轮优先精读顺序**：第 5、6、7、8、10、11 篇用于优化 SO-0/SO-1 和真实光谱锚定；第 1、2、3、4 篇用于临床任务定义与混杂控制；第 12、13、15 篇用于合成数据、域泛化、公平性与验证规范。

## 七、期刊指标核验来源

- 2024 JCR 指标的公开汇总核验：武汉大学高影响力期刊投稿指南系统（检索具体期刊）；例如 [European Heart Journal](https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=7530)、[IEEE Journal of Biomedical and Health Informatics](https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=7866)、[Journal of Biophotonics](https://topj.lib.whu.edu.cn/show.asp?cat=zk&id=745)、[Scientific Reports](https://topj.lib.whu.edu.cn/show.asp?cat=sci&id=11300)。
- 交叉核验示例：[Computers in Biology and Medicine 2024 JIF/Q1](https://www.iit.comillas.edu/publicacion/info_revista/en/730/Computers_in_Biology_and_Medicine)、[Physiological Measurement 2024 JIF/Q2](https://www.iit.comillas.edu/publicacion/info_revista/en/140/Physiological_Measurement)。
