"""The OCR checks' pages, written to ocr/pages/ (git-ignored; deterministic from this script and the release):
two synthetic report pages (A4 at 150 dpi, macOS's Arial) and three pages of the paper that ships in the release
(Unlimited-OCR-2.pdf: the title page, a page of prose, a page of tables), rendered at 150 dpi with PyMuPDF 1.27.2.2.

python3 ocr/pages.py MODEL_DIR"""

import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "pages"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
PAPER_PAGES = (1, 3, 10)


def report(lines, path):
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    y = 110
    for kind, text in lines:
        if kind == "h":
            d.text((110, y), text, font=ImageFont.truetype(BOLD, 44), fill="black")
            y += 80
        elif kind == "p":
            for row in text:
                d.text((110, y), row, font=ImageFont.truetype(FONT, 28), fill="black")
                y += 40
            y += 24
        elif kind == "t":
            f = ImageFont.truetype(FONT, 26)
            for row in text:
                for c, cell in enumerate(row):
                    d.text((120 + 330 * c, y + 8), cell, font=f, fill="black")
                d.line([(110, y), (1100, y)], fill="black", width=2)
                y += 48
            d.line([(110, y), (1100, y)], fill="black", width=2)
            y += 40
    im.save(path)


def main(model_dir: str) -> None:
    OUT.mkdir(exist_ok=True)
    report([("h", "1. Quarterly Report"),
            ("p", ["The third quarter closed with revenue of 4.2 million dollars,", "an increase of 12 percent over the same period last year.",
                   "Operating costs rose by 3 percent, mostly from new hiring."]),
            ("t", [["Region", "Revenue", "Growth"], ["North", "1.6 M", "+14%"], ["South", "1.1 M", "+9%"], ["West", "1.5 M", "+12%"]]),
            ("p", ["The board approved a dividend of 0.25 per share, payable", "on the fifteenth of December to holders of record."])],
           OUT / "report-1.png")
    report([("h", "2. Outlook"),
            ("p", ["For the fourth quarter we expect revenue between 4.3 and", "4.6 million dollars. The new warehouse in Leeds opens",
                   "in January and will shorten delivery times by two days."]),
            ("h", "3. Risks"),
            ("p", ["Currency movements and shipping costs remain the main risks.", "A 10 percent rise in freight rates would reduce the",
                   "operating margin by about one point."])],
           OUT / "report-2.png")
    doc = fitz.open(Path(model_dir).expanduser() / "Unlimited-OCR-2.pdf")
    for n in PAPER_PAGES:
        doc[n - 1].get_pixmap(dpi=150).save(OUT / f"paper-{n}.png")
    print("\n".join(sorted(str(p) for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/models/Unlimited-OCR-2")
