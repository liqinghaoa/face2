import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const csvPath = "E:/projects/face2/data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold-new.csv";
const workbookPath = "E:/resarch/face/face data/data-shift/summary/用户心脏指标数据表.xlsx";

const csvText = await fs.readFile(csvPath, "utf8");
const csvRows = csvText.trim().split(/\r?\n/).slice(1)
  .filter(Boolean)
  .map((line) => {
    const [id, sex, nyha] = line.split(",").map((value) => value.replace(/^"|"$/g, "").trim());
    return { id, sex, nyha: Number(nyha) };
  });
const csvById = new Map(csvRows.map((row) => [row.id, row]));

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("Sheet1");
const values = sheet.getUsedRange(true).values;
const nyhaZero = values.slice(1)
  .filter((row) => Number(row[2]) === 0)
  .map((row) => ({ id: String(row[0]).trim(), sex: row[1], nyha: row[2] }))
  .filter(({ id }) => id);
const missing = nyhaZero.filter(({ id }) => !csvById.has(id));
const presentWithDifferentNyha = nyhaZero
  .filter(({ id }) => csvById.has(id) && csvById.get(id).nyha !== 0)
  .map((record) => ({ ...record, csvNyha: csvById.get(record.id).nyha, csvSex: csvById.get(record.id).sex }));

console.log(JSON.stringify({
  csvRows: csvRows.length,
  csvNyhaZeroRows: csvRows.filter((row) => row.nyha === 0).length,
  table2NyhaZeroRows: nyhaZero.length,
  missingNyhaZeroRows: missing.length,
  missing,
  presentWithDifferentNyha,
}, null, 2));
