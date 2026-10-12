#!/usr/bin/env node
// Retypeset a scientific comparison grid's footer without changing its data panels.
// Requires sharp (set NODE_PATH when using the bundled workspace runtime).
const fs = require('node:fs/promises');
const path = require('node:path');
const sharp = require('sharp');

async function main() {
  const [source, output] = process.argv.slice(2);
  if (!source || !output || process.argv.length !== 4) {
    throw new Error('Usage: node tools/relabel_qualitative_grid.cjs SOURCE.png OUTPUT.png');
  }
  const names = ['Labels', 'YOLOv8n', 'YOLO11n', 'YOLO26n',
    'RT-DETR-R18', 'Faster R-CNN-R50', 'DMFR-YOLO'];
  const tile = 256, pad = 3, rows = 7, fontSize = 24;
  const width = names.length * (tile + pad) + pad;
  const footerTop = rows * (tile + pad) + pad;
  const input = await fs.readFile(source);
  const metadata = await sharp(input).metadata();
  if (metadata.width !== width || metadata.height !== footerTop + fontSize + 14) {
    throw new Error(`Unexpected grid dimensions: ${metadata.width} x ${metadata.height}`);
  }
  const footerHeight = metadata.height - footerTop;
  const labels = names.map((name, col) => {
    const center = pad + col * (tile + pad) + tile / 2;
    return `<text x="${center}" y="25" text-anchor="middle">${name}</text>`;
  }).join('');
  const footer = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${footerHeight}">
    <rect width="100%" height="100%" fill="white"/>
    <g font-family="Arial, sans-serif" font-size="${fontSize}" fill="black">${labels}</g>
  </svg>`);
  const result = await sharp(input).composite([{input: footer, left: 0, top: footerTop}])
    .removeAlpha().withMetadata({density: 300}).png().toBuffer();
  const region = {left: 0, top: 0, width, height: footerTop};
  const before = await sharp(input).extract(region).removeAlpha().raw().toBuffer();
  const after = await sharp(result).extract(region).removeAlpha().raw().toBuffer();
  if (!before.equals(after)) {
    throw new Error('Image or detection pixels changed; refusing to save.');
  }
  await fs.mkdir(path.dirname(path.resolve(output)), {recursive: true});
  await fs.writeFile(output, result);
  console.log(`Saved ${output}; all seven rows of detection panels are pixel-identical.`);
  console.log(`Columns: ${names.join(', ')}`);
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
