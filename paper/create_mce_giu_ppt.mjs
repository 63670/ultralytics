/**
 * Editable paper diagrams for MCE and GIU.
 * Run a copy from a private build directory with @oai/artifact-tool installed.
 * Environment: PRESENTATIONS_SKILL_DIR, CODEX_RUNTIME_PYTHON.
 * Usage: node create_mce_giu_ppt.mjs <output-directory>
 * All diagrams are native PowerPoint shapes, paths and editable text.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const out = path.resolve(process.argv[2] ?? 'output');
const build = path.dirname(fileURLToPath(import.meta.url));
const p = Presentation.create({ slideSize: { width: 1600, height: 900 } });
// Uniformly fit the scientific figure canvas within a standard 16:9 slide.
const place = (x, y, w, h) => ({ left: 80 + x * 0.9, top: y * 0.9, width: w * 0.9, height: h * 0.9 });
const C = {
  ink: '#243445', muted: '#637386', line: '#435466', faint: '#CBD3DC',
  blue: '#E7EFFA', blueStroke: '#6186B5', teal: '#E2F2EC', tealStroke: '#4C907C',
  peach: '#FBE9D9', peachStroke: '#C28E57', purple: '#EEE9F7', purpleStroke: '#8B78AE',
  neutral: '#F4F6F8', white: '#FFFFFF', gate: '#B87C3D',
};
let seq = 0;
const name = (s) => `${String(++seq).padStart(3, '0')}-${s}`;

function text(sl, value, x, y, w, h, size = 25, options = {}) {
  const sh = sl.shapes.add({
    geometry: 'textbox', name: name(value.replace(/\n/g, ' ')),
    position: place(x, y, w, h),
    fill: 'none', line: { fill: 'none', width: 0 },
  });
  sh.text = value;
  sh.text.style = {
    typeface: options.math ? 'Cambria Math' : 'Arial', fontSize: size * 0.9,
    color: options.color ?? C.ink, bold: options.bold ?? false,
    alignment: options.align ?? 'center', verticalAlignment: 'middle',
    autoFit: 'none', wrap: 'none', insets: 0,
  };
  return sh;
}

function box(sl, value, x, y, w, h, fill = C.neutral, stroke = C.faint, size = 25) {
  const sh = sl.shapes.add({
    geometry: 'roundRect', name: name(value.replace(/\n/g, ' ')),
    position: place(x, y, w, h),
    fill, line: { fill: stroke, width: 1.7, style: 'solid' }, borderRadius: 7,
  });
  sh.text = value;
  sh.text.style = {
    typeface: 'Arial', fontSize: size * 0.9, color: C.ink,
    alignment: 'center', verticalAlignment: 'middle',
    autoFit: 'none', wrap: 'none', insets: 3,
  };
  return sh;
}

function poly(sl, points, { color = C.line, width = 2.2, arrow = true, dashed = false } = {}) {
  const xs = points.map(pt => pt[0]), ys = points.map(pt => pt[1]);
  const left = Math.min(...xs), top = Math.min(...ys);
  const w = Math.max(1, Math.max(...xs) - left), h = Math.max(1, Math.max(...ys) - top);
  sl.shapes.add({
    geometry: 'custom', name: name('flow'), position: place(left, top, w, h),
    fill: 'none', line: { fill: color, width, style: dashed ? 'dashed' : 'solid' },
    customPaths: [{ width: w, height: h, commands: points.map(([x, y], i) => (
      i ? { lineTo: { x: x - left, y: y - top } } : { moveTo: { x: x - left, y: y - top } }
    )) }],
  });
  if (arrow) {
    const a = points.at(-2), b = points.at(-1);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
    const tri = [b, [b[0] - 9 * ux + 4.2 * uy, b[1] - 9 * uy - 4.2 * ux],
      [b[0] - 9 * ux - 4.2 * uy, b[1] - 9 * uy + 4.2 * ux]];
    const l = Math.min(...tri.map(pt => pt[0])), t = Math.min(...tri.map(pt => pt[1]));
    const tw = Math.max(1, Math.max(...tri.map(pt => pt[0])) - l);
    const th = Math.max(1, Math.max(...tri.map(pt => pt[1])) - t);
    sl.shapes.add({
      geometry: 'custom', name: name('arrowhead'), position: place(l, t, tw, th),
      fill: color, line: { fill: 'none', width: 0 },
      customPaths: [{ width: tw, height: th, commands: [
        ...tri.map(([x, y], i) => i ? { lineTo: { x: x - l, y: y - t } } : { moveTo: { x: x - l, y: y - t } }),
        { close: {} },
      ] }],
    });
  }
}

function dot(sl, x, y, color = C.line) {
  sl.shapes.add({ geometry: 'ellipse', name: name('junction'),
    position: place(x - 3.5, y - 3.5, 7, 7),
    fill: color, line: { fill: 'none', width: 0 } });
}
function op(sl, value, x, y, r = 23, fill = C.white, stroke = C.line) {
  const sh = sl.shapes.add({ geometry: 'ellipse', name: name(`operator ${value}`),
    position: place(x - r, y - r, 2 * r, 2 * r),
    fill, line: { fill: stroke, width: 2 } });
  sh.text = value;
  sh.text.style = { typeface: 'Cambria Math', fontSize: 27.9, color: C.ink,
    alignment: 'center', verticalAlignment: 'middle', insets: 0, autoFit: 'none' };
  return sh;
}
function rule(sl, y) { poly(sl, [[60, y], [1540, y]], { color: C.faint, width: 1, arrow: false }); }
function heading(sl, main, expanded) {
  text(sl, main, 60, 27, 200, 55, 43, { bold: true, align: 'left' });
  text(sl, expanded, 250, 35, 1290, 42, 29, { align: 'left' });
  rule(sl, 108);
}

// Slide 1: placement in the original C2f and the exact partial-channel mechanism.
{
  const s = p.slides.add();
  s.background.fill = C.white;
  s.speakerNotes.text = 'Source: ultralytics/nn/modules/block.py, C2fAxisResidual (MCE in the manuscript), and ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml. The illustrated nano configuration has two retained bottlenecks at each modified stage. Only the final hidden branch is modulated. q = max(1, floor(c/4)). Descriptor addition uses spatial broadcasting. The depthwise convolutions include biases. gamma is a learned per-channel scale initialized to zero. Conv denotes the original Conv-BN-SiLU block in the C2f overview.';
  heading(s, 'MCE', 'Morphology-aware context encoding');
  text(s, '(a) Integration in C2f', 60, 128, 550, 35, 26, { bold: true, align: 'left' });

  // The four original aggregation branches: Z0, Z1, Z2 and enhanced Z3.
  poly(s, [[103, 320], [150, 320]]);
  poly(s, [[270, 320], [320, 320]]);
  poly(s, [[420, 320], [500, 320]]);
  poly(s, [[660, 320], [750, 320]]);
  poly(s, [[910, 320], [1000, 320]]);
  poly(s, [[1120, 320], [1200, 320]]);
  poly(s, [[1300, 320], [1350, 320]]);
  poly(s, [[1470, 320], [1514, 320]]);
  poly(s, [[370, 285], [370, 200], [1200, 200]]);
  poly(s, [[460, 320], [460, 240], [1200, 240]]);
  poly(s, [[705, 320], [705, 280], [1200, 280]]);
  dot(s, 460, 320); dot(s, 705, 320);
  text(s, 'X', 62, 298, 36, 42, 31, { math: true });
  box(s, '1 × 1 Conv', 150, 285, 120, 70, C.blue, C.blueStroke, 22);
  box(s, 'Split', 320, 285, 100, 70);
  box(s, 'Bottleneck 1', 500, 285, 160, 70, C.blue, C.blueStroke, 23);
  box(s, 'Bottleneck 2', 750, 285, 160, 70, C.blue, C.blueStroke, 23);
  box(s, 'MCE', 1000, 285, 120, 70, C.teal, C.tealStroke, 27);
  box(s, 'Concat', 1200, 179, 100, 176, C.purple, C.purpleStroke, 23);
  box(s, '1 × 1 Conv', 1350, 285, 120, 70, C.blue, C.blueStroke, 22);
  text(s, 'Y', 1518, 298, 28, 42, 31, { math: true });
  text(s, 'Z₀', 780, 169, 60, 30, 24, { math: true });
  text(s, 'Z₁', 780, 209, 60, 30, 24, { math: true });
  text(s, 'Z₂', 960, 248, 60, 30, 24, { math: true });
  text(s, 'F', 930, 284, 50, 31, 24, { math: true });
  text(s, 'F′', 1132, 284, 55, 31, 24, { math: true });
  text(s, 'Backbone P3/8 and P4/16 stages', 60, 382, 730, 35, 23, { align: 'left', color: C.muted });
  text(s, 'Original local branches retained', 850, 382, 690, 35, 23, { align: 'right', color: C.muted });
  rule(s, 437);

  text(s, '(b) MCE on the final hidden branch', 60, 453, 800, 35, 26, { bold: true, align: 'left' });
  // Active channels provide the context descriptors, multiplicand and identity residual.
  poly(s, [[103, 710], [180, 710]]);
  poly(s, [[290, 710], [340, 710]], { arrow: false });
  poly(s, [[340, 710], [340, 575], [400, 575]]);
  poly(s, [[340, 710], [340, 695], [400, 695]]);
  poly(s, [[540, 575], [605, 575]]);
  poly(s, [[540, 695], [605, 695]]);
  poly(s, [[755, 575], [855, 575], [855, 602]]);
  poly(s, [[755, 695], [855, 695], [855, 648]]);
  poly(s, [[878, 625], [910, 625]]);
  poly(s, [[1050, 625], [1090, 625]]);
  poly(s, [[1200, 625], [1237, 625]]);
  poly(s, [[340, 575], [340, 510], [1260, 510], [1260, 602]]);
  poly(s, [[1260, 687], [1260, 648]], { color: C.gate });
  poly(s, [[1283, 625], [1340, 625], [1340, 767]]);
  poly(s, [[340, 710], [340, 790], [1317, 790]]);
  poly(s, [[1363, 790], [1410, 790]]);
  poly(s, [[235, 745], [235, 870], [1410, 870]]);
  poly(s, [[1500, 815], [1540, 815]]);
  dot(s, 340, 575); dot(s, 340, 695); dot(s, 340, 710);
  text(s, 'F', 62, 684, 36, 42, 31, { math: true });
  text(s, 'c × H × W', 60, 738, 108, 31, 21, { math: true, color: C.muted });
  box(s, 'Channel\nsplit', 180, 675, 110, 70, C.neutral, C.faint, 23);
  text(s, 'Fₐ', 292, 674, 45, 32, 25, { math: true });
  box(s, 'Mean over W', 400, 540, 140, 70, C.blue, C.blueStroke, 22);
  box(s, 'Mean over H', 400, 660, 140, 70, C.blue, C.blueStroke, 22);
  box(s, 'DWConv 3 × 1', 605, 540, 150, 70, C.teal, C.tealStroke, 22);
  box(s, 'DWConv 1 × 3', 605, 660, 150, 70, C.teal, C.tealStroke, 22);
  text(s, 'q × H × 1', 580, 612, 200, 29, 21, { math: true, color: C.muted });
  text(s, 'q × 1 × W', 580, 733, 200, 29, 21, { math: true, color: C.muted });
  op(s, '+', 855, 625);
  text(s, 'Broadcast add', 770, 527, 180, 31, 20, { color: C.muted });
  box(s, 'Hardsigmoid', 910, 590, 140, 70, C.peach, C.peachStroke, 22);
  box(s, '2g − 1', 1090, 590, 110, 70, C.peach, C.peachStroke, 26);
  text(s, 'A', 1204, 585, 40, 31, 25, { math: true });
  op(s, '⊙', 1260, 625, 23, C.peach, C.peachStroke);
  text(s, 'Fₐ', 1258, 532, 60, 33, 25, { math: true });
  text(s, 'γ', 1228, 688, 64, 40, 31, { math: true, color: C.gate });
  text(s, 'γ = 0 at initialization', 1115, 733, 250, 30, 21, { color: C.muted });
  op(s, '+', 1340, 790);
  text(s, 'Fₐ′', 1350, 748, 60, 33, 25, { math: true });
  text(s, 'F', 677, 832, 20, 35, 24, { math: true, color: C.muted });
  text(s, 'b', 696, 847, 14, 19, 17, { math: true, color: C.muted });
  text(s, '(unchanged channels)', 721, 832, 310, 35, 23, { color: C.muted, align: 'left' });
  box(s, 'Concat', 1410, 745, 90, 150, C.purple, C.purpleStroke, 22);
  text(s, 'F′', 1502, 766, 45, 35, 29, { math: true });
  text(s, 'Fₐ′ = Fₐ + γ ⊙ Fₐ ⊙ A', 60, 928, 850, 42, 29, { math: true, align: 'left' });
  text(s, 'q = max(1, ⌊c / 4⌋)     ⊙  element-wise product', 880, 928, 660, 42, 23, { math: true, align: 'right', color: C.muted });
}

// Slide 2: the convex-blend form is algebraically equivalent to the implementation.
{
  const s = p.slides.add();
  s.background.fill = C.white;
  s.speakerNotes.text = 'Source: ultralytics/nn/modules/block.py, GatedInterpolationUpsample, and ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml. The displayed convex blend is algebraically equivalent to the implemented N + G * (B - N). The gate is predicted on the coarse input using a biased 3x3 convolution from C channels to one, followed by Hardsigmoid and nearest-neighbor upsampling. Gate-convolution weights are initialized to zero and bias to -2.7, giving initial G = 0.05. The gate is shared across channels and replicated within each 2x2 block. Lateral P4 or P3 features enter only the subsequent concatenation. Bilinear interpolation uses align_corners=False.';
  heading(s, 'GIU', 'Gated interpolation upsampling');
  text(s, 'Content-adaptive reconstruction of higher-level context', 60, 128, 1480, 38, 26, { align: 'left' });
  poly(s, [[142, 450], [250, 450]], { arrow: false });
  poly(s, [[250, 450], [250, 250], [390, 250]]);
  poly(s, [[250, 450], [350, 450]]);
  poly(s, [[250, 450], [250, 650], [390, 650]]);
  dot(s, 250, 450);
  poly(s, [[640, 250], [1177, 250]], { color: C.blueStroke });
  poly(s, [[530, 450], [590, 450]], { color: C.gate });
  poly(s, [[770, 450], [830, 450]], { color: C.gate });
  poly(s, [[1010, 450], [1070, 450]], { color: C.gate, arrow: false });
  poly(s, [[1070, 450], [1070, 360], [1130, 360]], { color: C.gate });
  poly(s, [[1200, 330], [1200, 273]], { color: C.gate });
  poly(s, [[1070, 450], [1070, 550], [1200, 550], [1200, 627]], { color: C.gate });
  dot(s, 1070, 450, C.gate);
  poly(s, [[640, 650], [1177, 650]], { color: C.tealStroke });
  poly(s, [[1223, 250], [1380, 250], [1380, 427]]);
  poly(s, [[1223, 650], [1380, 650], [1380, 473]]);
  poly(s, [[1403, 450], [1490, 450]]);
  text(s, 'X', 85, 419, 50, 51, 37, { math: true });
  text(s, 'C × H × W', 60, 478, 165, 35, 23, { math: true, color: C.muted });
  box(s, 'Nearest × 2', 390, 205, 250, 90, C.blue, C.blueStroke, 30);
  box(s, 'Bilinear × 2', 390, 605, 250, 90, C.teal, C.tealStroke, 30);
  text(s, 'align_corners = False', 380, 699, 270, 31, 21, { color: C.muted });
  box(s, '3 × 3 Conv\nC → 1', 350, 410, 180, 80, C.peach, C.peachStroke, 25);
  box(s, 'Hardsigmoid', 590, 410, 180, 80, C.peach, C.peachStroke, 25);
  box(s, 'Nearest × 2', 830, 410, 180, 80, C.peach, C.peachStroke, 25);
  text(s, '1 × H × W', 565, 496, 225, 31, 22, { math: true, color: C.muted });
  text(s, 'G', 1025, 404, 45, 36, 29, { math: true, color: C.gate });
  text(s, '1 × 2H × 2W', 850, 497, 210, 31, 22, { math: true, color: C.muted, align: 'right' });
  box(s, '1 − G', 1130, 330, 140, 60, C.peach, C.peachStroke, 28);
  text(s, 'N', 780, 207, 70, 36, 30, { math: true });
  text(s, 'B', 780, 607, 70, 36, 30, { math: true });
  text(s, 'C × 2H × 2W', 730, 261, 250, 35, 23, { math: true, color: C.muted });
  text(s, 'C × 2H × 2W', 730, 661, 250, 35, 23, { math: true, color: C.muted });
  op(s, '⊙', 1200, 250, 23, C.white, C.line);
  op(s, '⊙', 1200, 650, 23, C.white, C.line);
  op(s, '+', 1380, 450);
  text(s, 'Y', 1490, 419, 50, 51, 37, { math: true });
  text(s, 'C × 2H × 2W', 1360, 487, 185, 35, 23, { math: true, color: C.muted });
  text(s, 'Y = (1 − G) ⊙ N + G ⊙ B', 60, 768, 1480, 52, 37, { math: true });
  text(s, 'Shared across channels', 60, 835, 460, 33, 23, { color: C.muted, align: 'left' });
  text(s, 'Initial G = 0.05', 605, 835, 390, 33, 23, { color: C.muted });
  text(s, 'Nearest upsampling of the gate', 1000, 835, 540, 33, 23, { color: C.muted, align: 'right' });
  rule(s, 897);
  // Explicitly external lateral fusion, outside the GIU operator itself.
  text(s, 'Top-down FPN', 60, 928, 240, 35, 23, { color: C.muted, align: 'left' });
  text(s, 'GIU output', 330, 918, 220, 50, 25);
  poly(s, [[550, 943], [620, 943]]);
  box(s, 'Concat', 620, 918, 150, 50, C.purple, C.purpleStroke, 23);
  poly(s, [[770, 943], [900, 943]]);
  box(s, 'C2f', 900, 918, 130, 50, C.blue, C.blueStroke, 24);
  text(s, 'Lateral P4 / P3', 1130, 918, 240, 50, 24);
  poly(s, [[1130, 943], [1085, 943], [1085, 985], [695, 985], [695, 968]]);
}

await fs.mkdir(out, { recursive: true });
for (const [i, slide] of p.slides.items.entries()) {
  const blob = await p.export({ slide, format: 'png', scale: 1.5 });
  await fs.writeFile(path.join(out, `${i + 1 === 1 ? 'MCE' : 'GIU'}_preview.png`), new Uint8Array(await blob.arrayBuffer()));
}
const candidate = path.join(build, 'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const skillDir = process.env.PRESENTATIONS_SKILL_DIR;
if (!skillDir || !process.env.CODEX_RUNTIME_PYTHON) throw new Error('Set PRESENTATIONS_SKILL_DIR and CODEX_RUNTIME_PYTHON.');
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({
  workspaceDir: path.resolve(build, '../..'), candidatePath: candidate, finalPath: path.join(out, 'MCE_GIU_Modules.pptx'),
  pythonExecutable: process.env.CODEX_RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '15240000,8572500', '--validate-bullet-geometry', '--validate-heading-fit'],
  requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: 'design', families: ['Arial', 'Cambria Math'] },
  verifyArtifactToolImport: true, receiptPath: path.join(build, 'validation.json'),
});
console.log(`Created ${path.join(out, 'MCE_GIU_Modules.pptx')}`);
