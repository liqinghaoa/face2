import fitz, sys
sys.stdout.reconfigure(encoding='utf-8')
p = r"E:\projects\face2\face2_research\01_Background_and_Literature\Key_Papers\Jung 等 - 2023 - Deep learning-based optical approach for skin analysis of melanin and hemoglobin distribution.pdf"
doc = fitz.open(p)
print("PAGES:", doc.page_count)
out = []
for i, page in enumerate(doc):
    t = page.get_text()
    out.append("\n===== PAGE %d =====\n" % (i + 1) + t)
txt = "".join(out)
open(r"E:\projects\face2\.workbuddy\tmp\jung2023.txt", "w", encoding="utf-8").write(txt)
print("CHARS:", len(txt))
