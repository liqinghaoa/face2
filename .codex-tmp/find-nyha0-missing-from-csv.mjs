import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "E:/resarch/face/face data/data-shift/summary/用户心脏指标数据表.xlsx";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItemAt(0);
const values = sheet.getUsedRange(true).values;
const headers = values[0];
const idIndex = headers.indexOf("ID");
const nyhaIndex = headers.indexOf("NYHA分级");
const sexIndex = headers.indexOf("性别");

if (idIndex < 0 || nyhaIndex < 0 || sexIndex < 0) {
  throw new Error("Required columns ID, 性别, or NYHA分级 were not found.");
}

const nyha0Records = values.slice(1)
  .filter((row) => String(row[nyhaIndex]).trim() === "0")
  .map((row) => ({ id: String(row[idIndex]).trim(), sex: row[sexIndex], nyha: row[nyhaIndex] }));

console.log(JSON.stringify({ sheet: sheet.name, nyha0Records }, null, 2));
