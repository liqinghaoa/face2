import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "E:/resarch/face/face data/data-shift/summary/用户心脏指标数据表.xlsx";
const imageDirectory = "E:/projects/face2/data/processed/P0_Physics_Audit_v1/images/replaceData";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const values = sheet.getUsedRange(true).values;
const recordsById = new Map(
  values.slice(1).map((row) => [String(row[0]).trim(), { sex: row[1], nyha: row[2] }]),
);

async function listFiles(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const results = await Promise.all(entries.map(async (entry) => {
    const fullPath = path.join(directory, entry.name);
    return entry.isDirectory() ? listFiles(fullPath) : [fullPath];
  }));
  return results.flat();
}

const imageFiles = await listFiles(imageDirectory);
const imageRecords = imageFiles.map((file) => {
  const id = path.parse(file).name;
  return { file: path.relative(imageDirectory, file), id, normalizedId: id.replace(/-\d+$/, "") };
});
const matched = imageRecords
  .filter(({ normalizedId }) => recordsById.has(normalizedId))
  .map((image) => ({ ...image, ...recordsById.get(image.normalizedId) }));
const unmatched = imageRecords.filter(({ normalizedId }) => !recordsById.has(normalizedId));

console.log(JSON.stringify({
  sheet: sheet.name,
  tableRecords: recordsById.size,
  imageFiles: imageRecords.length,
  matchedImages: matched.length,
  matched,
  unmatchedImages: unmatched,
}, null, 2));
