// docs/progress_report_draft.md -> docs/progress_report.docx (A4, 맑은 고딕)
// 사용: node scripts/build_report_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, Footer, PageNumber, ImageRun,
} = require("docx");

const SRC = path.join(__dirname, "..", "docs", "progress_report_draft.md");
const OUT = path.join(__dirname, "..", "docs", "progress_report.docx");
const FONT = { ascii: "Malgun Gothic", eastAsia: "Malgun Gothic", hAnsi: "Malgun Gothic", cs: "Malgun Gothic" };
const CONTENT_W = 9638; // A4 11906 - 2*1134
const BODY = 21; // 10.5pt

function runs(text, base = {}) {
  const out = [];
  for (const tok of text.split(/(\*\*[^*]+\*\*|`[^`]+`|【[^】]*】)/g)) {
    if (!tok) continue;
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: { ascii: "Consolas", eastAsia: "Malgun Gothic", hAnsi: "Consolas" }, ...base, size: (base.size || BODY) - 2 }));
    else if (tok.startsWith("【")) out.push(new TextRun({ text: tok, highlight: "yellow", font: FONT, ...base }));
    else out.push(new TextRun({ text: tok, font: FONT, ...base }));
  }
  return out;
}

// ![캡션](docs/figures/x.png) 형식. PNG 헤더에서 크기를 읽어 본문 폭(약 16.5cm)에 맞춘다.
function imageBlock(alt, rel) {
  const data = fs.readFileSync(path.join(__dirname, "..", rel));
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const maxW = 620; // px(96dpi) 기준 약 16.4cm
  const width = Math.min(maxW, w), height = Math.round((h * width) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width, height }, altText: { title: alt, description: alt, name: alt } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs(alt, { size: 18, color: "555555" }) }),
  ];
}

function vlen(s) { let n = 0; for (const ch of s) n += ch.charCodeAt(0) > 0x2e80 ? 2 : 1; return n; }

function buildTable(lines) {
  const rows = lines.map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const header = rows[0];
  const body = rows.slice(2);
  const n = header.length;
  const weights = header.map((_, i) => Math.min(40, Math.max(6, ...[header, ...body].map((r) => vlen((r[i] || "").replace(/\*\*|`/g, ""))))));
  const wsum = weights.reduce((a, b) => a + b, 0);
  let widths = weights.map((w) => Math.max(900, Math.round((w / wsum) * CONTENT_W)));
  const diff = CONTENT_W - widths.reduce((a, b) => a + b, 0);
  widths[widths.indexOf(Math.max(...widths))] += diff;
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders,
    shading: head ? { fill: "E7ECF4", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { after: 0 }, children: runs(txt, { size: 18, bold: head || undefined }) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...body.map((r) => new TableRow({ cantSplit: true, children: header.map((_, i) => cell(r[i] || "", widths[i], false)) })),
    ],
  });
}

function build() {
  const lines = fs.readFileSync(SRC, "utf-8").split("\n");
  const kids = [];
  let listInstance = 0;
  let i = 0;
  let cover = true; // 제목 아래 첫 블록(작성일 등)은 가운데 정렬
  const para = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 320 }, ...opts, children: runs(text, opts.run || {}) });

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith(">")) { while (i < lines.length && lines[i].startsWith(">")) i++; continue; } // 작성 메모는 제외
    if (/^---+$/.test(line.trim())) {
      kids.push(new Paragraph({ spacing: { before: 60, after: 120 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } }, children: [] }));
      cover = false; i++; continue;
    }
    if (line.startsWith("# ")) {
      kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 200 }, children: [new TextRun({ text: line.slice(2), bold: true, size: 44, font: FONT, color: "1F3864" })] }));
      i++; continue;
    }
    if (line.startsWith("## ")) { kids.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: line.slice(3), font: FONT })] })); cover = false; i++; continue; }
    if (line.startsWith("### ")) { kids.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: line.slice(4), font: FONT })] })); i++; continue; }
    const im = line.match(/^!\[(.*)\]\((.+)\)\s*$/);
    if (im) { kids.push(...imageBlock(im[1], im[2])); i++; continue; }
    if (line.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      kids.push(buildTable(block));
      kids.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    if (/^\s*- /.test(line) || /^\d+\.\s/.test(line)) {
      const ordered = /^\d+\.\s/.test(line);
      const inst = ordered ? ++listInstance : 0;
      while (i < lines.length && (/^\s*- /.test(lines[i]) || /^\d+\.\s/.test(lines[i]) || /^\s{2,}- /.test(lines[i]))) {
        const l = lines[i];
        const nested = /^\s{2,}- /.test(l);
        const text = l.replace(/^\s*(- |\d+\.\s)/, "");
        if (ordered && !nested) kids.push(new Paragraph({ numbering: { reference: "num", level: 0, instance: inst }, spacing: { after: 80, line: 300 }, children: runs(text) }));
        else kids.push(new Paragraph({ numbering: { reference: "bul", level: nested ? 1 : 0 }, spacing: { after: 80, line: 300 }, children: runs(text) }));
        i++;
      }
      continue;
    }
    kids.push(para(line, cover ? { alignment: AlignmentType.CENTER, spacing: { after: 60 } } : {}));
    i++;
  }

  return new Document({
    creator: "Progress Report",
    title: "Progress Report",
    styles: {
      default: { document: { run: { font: FONT, size: BODY } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, color: "1F3864", font: FONT },
          paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0, keepNext: true } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, color: "2F5496", font: FONT },
          paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1, keepNext: true } },
      ],
    },
    numbering: {
      config: [
        { reference: "bul", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } },
        ] },
        { reference: "num", levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } } } },
        ] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Progress Report · ", size: 16, color: "888888", font: FONT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888", font: FONT }),
      ] })] }) },
      children: kids,
    }],
  });
}

Packer.toBuffer(build()).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
