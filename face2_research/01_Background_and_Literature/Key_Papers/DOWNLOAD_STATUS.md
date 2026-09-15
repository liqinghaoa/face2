# Key Papers 下载状态（2026-09-05）

| 编号 | 文献 | 状态 | 补充材料 |
|---|---|---|---|
| 5 | Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution | 目录中原已存在，按要求未重复下载 | 未发现需补充下载的公开附件 |
| 6 | Spectrum-based deep learning framework for dermatological pigment analysis and simulation | 尚未下载。文献为 CC BY-NC-ND 开放获取，但 Elsevier API 缺少密钥，ScienceDirect 与作者公开稿页面均触发反自动化验证 | 出版社页面未列出补充材料 |
| 7 | Integrated deep learning approach for generating cross-polarized images and analyzing skin melanin and hemoglobin distributions | 已下载 PMC 完整全文，并生成 26 页 PDF；同时保留可追溯的 PMC HTML 源文件 | PMC 未列出补充材料；代码和数据声明为不公开 |
| 8 | Deep learning-based pigment analysis model trained with optical approach and ground truth assistance | 尚未下载。公开检索仅发现摘要/元数据，Wiley 正文需订阅或作者授权 | 未发现公开补充材料 |
| 10 | Absorption and reduced scattering coefficients in epidermis and dermis from a Swedish cohort study | 已从 DiVA 机构仓储下载期刊 PDF | Europe PMC 全文记录未包含补充材料 |
| 11 | Hyper-Skin: A Hyperspectral Dataset for Reconstructing Facial Skin-Spectra from RGB Images | 已从 NeurIPS 官方页面下载 PDF | 已下载官方 ZIP 并解压，共 6 个 PDF 附件 |

## 文件组织说明

- 新下载文献均使用“文献目录 / PDFs / 正文”结构。
- 补充材料放在 `SupportingInformation` 下；压缩包保留原件，并解压到 `Extracted`。
- `manifest.json` 记录来源、文件大小、SHA-256 和补充材料状态。
- 7 号文献的出版社 PDF 入口触发反自动化页面，因此保存的是 PubMed Central 完整 HTML 全文生成的 PDF，不是出版社原始版式 PDF。
