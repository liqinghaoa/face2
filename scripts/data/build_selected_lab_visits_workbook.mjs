import fs from "node:fs/promises";
import readline from "node:readline";
import { createReadStream } from "node:fs";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [detailPath, auditPath, outputPath] = process.argv.slice(2);
if (!detailPath || !auditPath || !outputPath) {
  throw new Error("Usage: node build_selected_lab_visits_workbook.mjs <detail.jsonl> <audit.json> <output.xlsx>");
}

function columnName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function parseDate(value) {
  if (typeof value !== "string") return value;
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  if (!match) return value;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]), Number(match[4]), Number(match[5]), Number(match[6]));
}

const audit = JSON.parse(await fs.readFile(auditPath, "utf8"));

const workbook = Workbook.create();
const detail = workbook.worksheets.add("选中检验数据");
const auditSheet = workbook.worksheets.add("匹配审计");
detail.showGridLines = false;
auditSheet.showGridLines = false;

const lines = readline.createInterface({ input: createReadStream(detailPath, { encoding: "utf8" }), crlfDelay: Infinity });
let detailHeaders = null;
let detailRowIndex = 1;
let chunk = [];
const chunkSize = 2000;
const dateColumns = new Set([2, 7, 15, 16]);

async function writeChunk() {
  if (!chunk.length) return;
  detail.getRangeByIndexes(detailRowIndex, 0, chunk.length, detailHeaders.length).values = chunk;
  detailRowIndex += chunk.length;
  chunk = [];
}

for await (const line of lines) {
  const value = JSON.parse(line);
  if (!detailHeaders) {
    detailHeaders = value.headers;
    detail.getRangeByIndexes(0, 0, 1, detailHeaders.length).values = [detailHeaders];
    continue;
  }
  for (const index of dateColumns) value[index] = parseDate(value[index]);
  chunk.push(value);
  if (chunk.length >= chunkSize) await writeChunk();
}
await writeChunk();

const detailLastCol = columnName(detailHeaders.length - 1);
const detailLastRow = detailRowIndex;
detail.getRange(`A1:${detailLastCol}1`).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
detail.getRange(`A2:${detailLastCol}${detailLastRow}`).format.font = { name: "Arial", size: 9 };
detail.getRange(`C2:C${detailLastRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
detail.getRange(`H2:H${detailLastRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
detail.getRange(`P2:Q${detailLastRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
detail.getRange(`I2:I${detailLastRow}`).format.numberFormat = "0.0000";
detail.freezePanes.freezeRows(1);
detail.freezePanes.freezeColumns(2);
detail.getRange("A:A").format.columnWidth = 17;
detail.getRange("B:B").format.columnWidth = 15;
detail.getRange("C:C").format.columnWidth = 20;
detail.getRange("D:D").format.columnWidth = 25;
detail.getRange("E:G").format.columnWidth = 14;
detail.getRange("H:H").format.columnWidth = 20;
detail.getRange("I:J").format.columnWidth = 12;
detail.getRange(`K:${detailLastCol}`).format.columnWidth = 15;

const auditRows = audit.rows.map((row) => row.map(parseDate));
// Columns 16-19 are 住院2锚点申请时间/住院2最晚申请时间 and stay strings when
// only one admission exists, so only format the pairs that actually parsed.
const auditDateColumns = [2, 10, 11, 16, 17];
for (const row of auditRows) {
  for (const index of auditDateColumns) {
    if (!(row[index] instanceof Date)) row[index] = row[index] === null ? null : String(row[index]);
  }
}
auditSheet.getRangeByIndexes(0, 0, 1, audit.headers.length).values = [audit.headers];
auditSheet.getRangeByIndexes(1, 0, auditRows.length, audit.headers.length).values = auditRows;
const auditLastCol = columnName(audit.headers.length - 1);
const auditLastRow = auditRows.length + 1;
auditSheet.getRange(`A1:${auditLastCol}1`).format = {
  fill: "#4472C4",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
auditSheet.getRange(`A2:${auditLastCol}${auditLastRow}`).format.font = { name: "Arial", size: 9 };
for (const column of ["C", "K", "L", "P", "Q", "R", "S"]) {
  auditSheet.getRange(`${column}2:${column}${auditLastRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
}
for (const column of ["M", "T"]) {
  auditSheet.getRange(`${column}2:${column}${auditLastRow}`).format.numberFormat = "0.0000";
}
auditSheet.freezePanes.freezeRows(1);
auditSheet.freezePanes.freezeColumns(2);
auditSheet.getRange("A:B").format.columnWidth = 17;
auditSheet.getRange("C:C").format.columnWidth = 20;
auditSheet.getRange("D:F").format.columnWidth = 16;
auditSheet.getRange(`G:${auditLastCol}`).format.columnWidth = 18;

detail.tables.add(`A1:${detailLastCol}${detailLastRow}`, true, "SelectedLabData").style = "TableStyleMedium2";
auditSheet.tables.add(`A1:${auditLastCol}${auditLastRow}`, true, "MatchingAudit").style = "TableStyleMedium2";
workbook.recalculate();

const inspection = await workbook.inspect({
  kind: "table",
  range: "匹配审计!A1:T8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 20,
  maxChars: 8000,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(new URL(".", `file:///${outputPath.replaceAll("\\", "/")}`).pathname, { recursive: true }).catch(() => {});
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, detailRows: detailLastRow - 1, auditRows: auditLastRow - 1, summary: audit.summary }));
