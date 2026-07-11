import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function colName(indexZeroBased) {
  let n = indexZeroBased + 1;
  let result = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function xmlSafe(value) {
  if (typeof value !== "string") return value;
  let result = "";
  for (const char of value) {
    const code = char.codePointAt(0);
    const valid = code === 0x09 || code === 0x0A || code === 0x0D
      || (code >= 0x20 && code <= 0xD7FF)
      || (code >= 0xE000 && code <= 0xFFFD)
      || (code >= 0x10000 && code <= 0x10FFFF);
    if (valid) result += char;
  }
  return /^[=+\-@]/.test(result) ? `'${result}` : result;
}

function sanitize(value) {
  if (Array.isArray(value)) return value.map(sanitize);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, sanitize(item)]));
  }
  return xmlSafe(value);
}

function applyHeader(range) {
  range.format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    horizontalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
  };
  range.format.rowHeight = 32;
}

function addTable(sheet, range, name) {
  const table = sheet.tables.add(range, true, name);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  return table;
}

const [jsonPath, outputPath, previewDir] = process.argv.slice(2);
if (!jsonPath || !outputPath || !previewDir) {
  throw new Error("Usage: node build_image_metadata_workbook.mjs <input.json> <output.xlsx> <preview-dir>");
}

const payload = sanitize(JSON.parse(await fs.readFile(jsonPath, "utf8")));
const workbook = Workbook.create();

const summary = workbook.worksheets.add("提取汇总");
const metadata = workbook.worksheets.add("图片元数据");
const raw = workbook.worksheets.add("EXIF原始明细");
const definitions = workbook.worksheets.add("字段说明");

for (const sheet of [summary, metadata, raw, definitions]) {
  sheet.showGridLines = false;
}

// Main metadata table.
const mainRows = [payload.main_headers, ...payload.main_rows];
const mainLastCol = colName(payload.main_headers.length - 1);
const mainLastRow = mainRows.length;
metadata.getRange(`A1:${mainLastCol}${mainLastRow}`).values = mainRows;
applyHeader(metadata.getRange(`A1:${mainLastCol}1`));
metadata.freezePanes.freezeRows(1);
metadata.freezePanes.freezeColumns(3);
addTable(metadata, `A1:${mainLastCol}${mainLastRow}`, "ImageMetadataTable");
metadata.getRange(`A2:A${mainLastRow}`).format.numberFormat = "0";
for (const header of ["文件大小(字节)", "宽度(px)", "高度(px)", "通道数", "位深(每通道)", "帧数", "EXIF字节数", "ICC字节数", "XMP字节数"]) {
  const idx = payload.main_headers.indexOf(header);
  if (idx >= 0) metadata.getRange(`${colName(idx)}2:${colName(idx)}${mainLastRow}`).format.numberFormat = "#,##0";
}
for (const header of ["文件大小(MB)", "总像素(MP)"]) {
  const idx = payload.main_headers.indexOf(header);
  if (idx >= 0) metadata.getRange(`${colName(idx)}2:${colName(idx)}${mainLastRow}`).format.numberFormat = "0.0000";
}
for (const header of ["长宽比", "GPS纬度", "GPS经度"]) {
  const idx = payload.main_headers.indexOf(header);
  if (idx >= 0) metadata.getRange(`${colName(idx)}2:${colName(idx)}${mainLastRow}`).format.numberFormat = "0.00000000";
}
metadata.getRange(`A1:${mainLastCol}${Math.min(mainLastRow, 40)}`).format.verticalAlignment = "center";
metadata.getRange(`B2:E${mainLastRow}`).format.numberFormat = "@";
metadata.getRange(`A1:${mainLastCol}${mainLastRow}`).format.font = { name: "Microsoft YaHei", size: 9 };
metadata.getRange(`A1:${mainLastCol}1`).format.font = { name: "Microsoft YaHei", size: 9, bold: true, color: "#FFFFFF" };

const mainWidths = {
  "序号": 7, "ID": 15, "文件名": 21, "相对路径": 28, "绝对路径": 42,
  "扩展名": 9, "SHA256": 66, "图像格式": 11, "MIME类型": 14,
  "文件创建时间": 20, "文件修改时间": 20, "方向解释": 24,
  "图像描述": 28, "厂商": 16, "相机型号": 20, "镜头型号": 24,
  "文件内修改时间": 20, "原始拍摄时间": 20, "数字化时间": 20,
  "用户注释": 30, "错误信息": 34,
};
payload.main_headers.forEach((header, idx) => {
  const width = mainWidths[header] ?? (header.length > 12 ? 16 : 12);
  metadata.getRange(`${colName(idx)}1:${colName(idx)}${mainLastRow}`).format.columnWidth = width;
});

// Raw metadata long table.
const rawRows = [payload.raw_headers, ...payload.raw_rows];
const rawLastCol = colName(payload.raw_headers.length - 1);
const rawLastRow = rawRows.length;
raw.getRange(`A1:${rawLastCol}${rawLastRow}`).values = rawRows;
applyHeader(raw.getRange(`A1:${rawLastCol}1`));
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(2);
addTable(raw, `A1:${rawLastCol}${rawLastRow}`, "RawExifDetailsTable");
raw.getRange(`A1:${rawLastCol}${rawLastRow}`).format.font = { name: "Microsoft YaHei", size: 9 };
raw.getRange(`A1:${rawLastCol}1`).format.font = { name: "Microsoft YaHei", size: 9, bold: true, color: "#FFFFFF" };
raw.getRange(`A1:A${rawLastRow}`).format.columnWidth = 15;
raw.getRange(`B1:B${rawLastRow}`).format.columnWidth = 22;
raw.getRange(`C1:C${rawLastRow}`).format.columnWidth = 15;
raw.getRange(`D1:D${rawLastRow}`).format.columnWidth = 11;
raw.getRange(`E1:E${rawLastRow}`).format.columnWidth = 28;
raw.getRange(`F1:F${rawLastRow}`).format.columnWidth = 55;
raw.getRange(`F2:F${rawLastRow}`).format.wrapText = false;

// Field definitions.
const defRows = [payload.definition_headers, ...payload.definition_rows];
const defLastRow = defRows.length;
definitions.getRange(`A1:D${defLastRow}`).values = defRows;
applyHeader(definitions.getRange("A1:D1"));
definitions.freezePanes.freezeRows(1);
addTable(definitions, `A1:D${defLastRow}`, "MetadataDefinitionsTable");
definitions.getRange(`A1:D${defLastRow}`).format.font = { name: "Microsoft YaHei", size: 10 };
definitions.getRange(`A1:A${defLastRow}`).format.columnWidth = 29;
definitions.getRange(`B1:B${defLastRow}`).format.columnWidth = 17;
definitions.getRange(`C1:C${defLastRow}`).format.columnWidth = 72;
definitions.getRange(`D1:D${defLastRow}`).format.columnWidth = 18;
definitions.getRange(`C2:C${defLastRow}`).format.wrapText = true;
definitions.getRange(`A2:D${defLastRow}`).format.verticalAlignment = "top";

// Formula-driven summary.
summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["图像元数据提取汇总"]];
summary.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { name: "Microsoft YaHei", size: 16, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A1:F1").format.rowHeight = 34;
summary.getRange("A3:B5").values = [
  ["图片目录", payload.summary.image_dir],
  ["提取时间", payload.summary.generated_at],
  ["工作簿说明", "每张图片一行；EXIF原始明细保存所有可解析标签；空白表示源文件未写入该字段。"],
];
summary.getRange("A3:A5").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("A3:A5").format.columnWidth = 16;
summary.getRange("B3:F5").merge(true);
summary.getRange("B3:F5").format.wrapText = true;
summary.getRange("B:B").format.columnWidth = 24;
summary.getRange("C1:F19").format.columnWidth = 16;

summary.getRange("A7:B7").values = [["指标", "数量"]];
applyHeader(summary.getRange("A7:B7"));
const statusCol = colName(payload.main_headers.indexOf("提取状态"));
const exifCol = colName(payload.main_headers.indexOf("EXIF存在"));
const iccCol = colName(payload.main_headers.indexOf("ICC存在"));
const xmpCol = colName(payload.main_headers.indexOf("XMP存在"));
const gpsLatCol = colName(payload.main_headers.indexOf("GPS纬度"));
const gpsLonCol = colName(payload.main_headers.indexOf("GPS经度"));
summary.getRange("A8:A15").values = [
  ["图片总数"], ["成功提取"], ["提取失败"], ["含EXIF"], ["含ICC"], ["含XMP"], ["含完整GPS坐标"], ["原始元数据标签行数"],
];
summary.getRange("B8:B15").formulas = [
  [`=COUNTA('图片元数据'!$B$2:$B$${mainLastRow})`],
  [`=COUNTIF('图片元数据'!$${statusCol}$2:$${statusCol}$${mainLastRow},"成功")`],
  [`=COUNTIF('图片元数据'!$${statusCol}$2:$${statusCol}$${mainLastRow},"失败")`],
  [`=COUNTIF('图片元数据'!$${exifCol}$2:$${exifCol}$${mainLastRow},"是")`],
  [`=COUNTIF('图片元数据'!$${iccCol}$2:$${iccCol}$${mainLastRow},"是")`],
  [`=COUNTIF('图片元数据'!$${xmpCol}$2:$${xmpCol}$${mainLastRow},"是")`],
  [`=COUNTIFS('图片元数据'!$${gpsLatCol}$2:$${gpsLatCol}$${mainLastRow},"<>",'图片元数据'!$${gpsLonCol}$2:$${gpsLonCol}$${mainLastRow},"<>")`],
  [`=COUNTA('EXIF原始明细'!$A$2:$A$${rawLastRow})`],
];
summary.getRange("A8:B15").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
summary.getRange("B8:B15").format.numberFormat = "#,##0";
summary.getRange("B8:B15").format.font = { bold: true, color: "#1F4E78" };
summary.getRange("D7:E7").values = [["文件扩展名", "数量"]];
applyHeader(summary.getRange("D7:E7"));
const extensions = Object.entries(payload.summary.extensions);
if (extensions.length) {
  summary.getRange(`D8:D${7 + extensions.length}`).values = extensions.map(([ext]) => [ext]);
  summary.getRange(`E8:E${7 + extensions.length}`).formulas = extensions.map(([ext]) => [
    `=COUNTIF('图片元数据'!$F$2:$F$${mainLastRow},"${ext}")`,
  ]);
  summary.getRange(`E8:E${7 + extensions.length}`).format.numberFormat = "#,##0";
}
summary.getRange("A17:F19").merge();
summary.getRange("A17").values = [["注意：EXIF 时间通常由拍摄设备写入，可能缺少时区或被后期软件修改；文件创建/修改时间来自当前文件系统，不等同于拍摄时间。GPS 信息属于敏感位置数据，分享工作簿前请确认是否需要脱敏。"]];
summary.getRange("A17:F19").format = { fill: "#FFF2CC", font: { color: "#7F6000" }, wrapText: true, verticalAlignment: "center" };
summary.getRange("A17:F19").format.rowHeight = 24;
summary.freezePanes.freezeRows(1);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const checks = {
  summary: (await workbook.inspect({ kind: "table", range: "提取汇总!A1:F19", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 8, maxChars: 7000 })).ndjson,
  metadata: (await workbook.inspect({ kind: "table", range: `图片元数据!A1:P${Math.min(mainLastRow, 8)}`, include: "values", tableMaxRows: 8, tableMaxCols: 16, maxChars: 6000 })).ndjson,
  errors: (await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" })).ndjson,
};
await fs.writeFile(path.join(previewDir, "checks.json"), JSON.stringify(checks, null, 2), "utf8");

const renderSpecs = [
  ["提取汇总", "A1:F19", "summary.png"],
  ["图片元数据", "A1:P12", "metadata_left.png"],
  ["图片元数据", `${colName(Math.max(0, payload.main_headers.length - 16))}1:${mainLastCol}12`, "metadata_right.png"],
  ["EXIF原始明细", "A1:F18", "raw_details.png"],
  ["字段说明", "A1:D18", "definitions.png"],
];
for (const [sheetName, range, fileName] of renderSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: payload.main_rows.length, rawRows: payload.raw_rows.length, checks }, null, 2));
