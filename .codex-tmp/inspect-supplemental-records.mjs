import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "E:/resarch/face/face data/收集照片(补充）/2026.4.20.xlsx";
const imageDirectory = "E:/projects/face2/data/processed/P0_Physics_Audit_v1/images/replaceData";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const values = sheet.getUsedRange(true).values;
const recordIds = new Set(
  values.slice(1).map((row) => String(row[0]).trim()).filter(Boolean),
);

async function listFiles(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const fullPath = path.join(directory, entry.name);
    return entry.isDirectory() ? listFiles(fullPath) : [fullPath];
  }));
  return nested.flat();
}

const imageFiles = await listFiles(imageDirectory);
const imageRecords = imageFiles.map((file) => {
  const stem = path.parse(file).name;
  return {
    file: path.relative(imageDirectory, file),
    id: stem,
    normalizedId: stem.replace(/-\d+$/, ""),
  };
});
const missingExact = imageRecords.filter(({ id }) => !recordIds.has(id));
const missingNormalized = imageRecords.filter(({ normalizedId }) => !recordIds.has(normalizedId));

console.log(JSON.stringify({
  sheet: sheet.name,
  tableRecordCount: recordIds.size,
  imageFileCount: imageRecords.length,
  exactMatches: imageRecords.length - missingExact.length,
  normalizedMatches: imageRecords.length - missingNormalized.length,
  missingFromTable: missingNormalized,
  suffixOnlyMatches: missingExact.filter(({ normalizedId }) => recordIds.has(normalizedId)),
}, null, 2));
