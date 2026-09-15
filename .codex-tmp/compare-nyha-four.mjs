import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const csvPath = "E:/projects/face2/data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold-new.csv";
const workbookPath = "E:/resarch/face/face data/data-shift/summary/用户心脏指标数据表.xlsx";

const csvText = await fs.readFile(csvPath, "utf8");
const csvIds = new Set(csvText.trim().split(/\r?\n/).slice(1)
  .filter(Boolean)
  .map((line) => line.split(",")[0].replace(/^"|"$/g, "").trim()));

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Sheet1");
const values = sheet.getUsedRange(true).values;
const nyhaFour = values.slice(1)
  .filter((row) => Number(row[2]) === 4)
  .map((row) => ({ id: String(row[0]).trim(), sex: row[1], nyha: row[2] }))
  .filter(({ id }) => id);
const missing = nyhaFour.filter(({ id }) => !csvIds.has(id));

console.log(JSON.stringify({
  table2NyhaFourRows: nyhaFour.length,
  presentInCsv: nyhaFour.length - missing.length,
  missingCount: missing.length,
  missing,
}, null, 2));
