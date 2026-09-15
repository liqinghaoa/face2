import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "E:/resarch/face/face data/收集照片(补充）/2026.4.20.xlsx";
const csvPath = "E:/projects/face2/data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold-new.csv";

const csvText = await fs.readFile(csvPath, "utf8");
const csvIds = new Set(csvText.trim().split(/\r?\n/).slice(1)
  .filter(Boolean)
  .map((line) => line.split(",")[0].replace(/^"|"$/g, "").trim()));

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("2026.4.20");
const values = sheet.getUsedRange(true).values;
const nyhaTwo = values.slice(1)
  .filter((row) => Number(row[3]) === 2)
  .map((row) => ({ id: String(row[0]).trim(), sex: row[1], nyha: row[3] }))
  .filter(({ id }) => id);
const missing = nyhaTwo.filter(({ id }) => !csvIds.has(id));

console.log(JSON.stringify({
  table1NyhaTwoRows: nyhaTwo.length,
  presentInCsv: nyhaTwo.length - missing.length,
  missingCount: missing.length,
  missing,
}, null, 2));
