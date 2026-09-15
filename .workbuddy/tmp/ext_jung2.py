import fitz, sys
sys.stdout.reconfigure(encoding='utf-8')
p = r"E:\projects\face2\face2_research\01_Background_and_Literature\Key_Papers\08_Jung_2023_Optical_GT_assisted_pigment_analysis\Journal of Biophotonics - 2023 - Jung - Deep learning‐based pigment analysis model trained with optical approach and ground.pdf"
doc = fitz.open(p)
print("PAGES:", doc.page_count)
out = []
for i, page in enumerate(doc):
    out.append("\n===== PAGE %d =====\n" % (i + 1) + page.get_text())
txt = "".join(out)
open(r"E:\projects\face2\.workbuddy\tmp\jung2023b.txt", "w", encoding="utf-8").write(txt)
print("CHARS:", len(txt))
