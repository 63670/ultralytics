/** Editable diagrams matching the supplied pastel, tensor-stack paper figure.
 * Run a copy from paper/work/<private-build> with linked runtime node_modules.
 * Usage: node create_mce_giu_reference_ppt.mjs <new-output-directory>
 * Env: PRESENTATIONS_SKILL_DIR, CODEX_RUNTIME_PYTHON, RUNTIME_NODE_MODULES.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const build = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(process.argv[2]);
const p = Presentation.create({ slideSize: { width: 1600, height: 900 } });
const C = {
  ink: '#252524', muted: '#585C51', cream: '#FFFBE9', lavender: '#F3EFFA',
  mint: '#EDF5DD', blue: '#B3DDE9', blueEdge: '#5196AC', green: '#C7DFAB', greenEdge: '#759B4E',
  pink: '#F5B6C1', pinkEdge: '#BF6F8B', purple: '#D8B4D9', purpleEdge: '#98649D',
  yellow: '#F8D887', yellowEdge: '#BD9C47', red: '#E07077', cyan: '#489DBF', white: '#FFFFFF',
};
let seq = 0;
const id = n => `${++seq}-${n}`;
function txt(s, label, x, y, w, h, size = 25, o = {}) {
  const sh = s.shapes.add({ geometry: 'textbox', name: id(label), position: { left:x, top:y, width:w, height:h, rotation:o.rotation ?? 0 }, fill:'none', line:{fill:'none',width:0} });
  sh.text = label;
  sh.text.style = { typeface:o.math ? 'Cambria Math' : 'Times New Roman', fontSize:size,
    color:o.color ?? C.ink, bold:o.bold ?? false, alignment:o.align ?? 'center',
    verticalAlignment:'middle', autoFit:'none', wrap:'none', insets:0 };
  return sh;
}
function rect(s, x, y, w, h, fill, stroke = 'none', radius = 0, dashed = false) {
  return s.shapes.add({ geometry: radius ? 'roundRect' : 'rect', name:id('region'),
    position:{left:x,top:y,width:w,height:h}, fill, borderRadius:radius,
    line:{fill:stroke,width:stroke === 'none' ? 0 : 1.25,style:dashed ? 'dashed' : 'solid'} });
}
function block(s, label, x, y, w, h, fill, edge, size = 25) {
  rect(s,x,y,w,h,fill,edge,3); txt(s,label,x+3,y+2,w-6,h-4,size);
}
function vertical(s,label,x,y,w,h,fill,edge,size=24) {
  rect(s,x,y,w,h,fill,edge,3);
  txt(s,label,x+w/2-(h-10)/2,y+h/2-(w-6)/2,h-10,w-6,size-2,{rotation:270});
}
function custom(s, points, fill, stroke, width=1) {
  const x=Math.min(...points.map(a=>a[0])), y=Math.min(...points.map(a=>a[1]));
  const w=Math.max(1,Math.max(...points.map(a=>a[0]))-x),h=Math.max(1,Math.max(...points.map(a=>a[1]))-y);
  s.shapes.add({geometry:'custom',name:id(fill==='none'?'flow':'plane'),position:{left:x,top:y,width:w,height:h},
    fill,line:{fill:stroke,width},customPaths:[{width:w,height:h,commands:[
      ...points.map(([a,b],i)=>i?{lineTo:{x:a-x,y:b-y}}:{moveTo:{x:a-x,y:b-y}}),
      ...(fill === 'none' ? [] : [{close:{}}]),
    ]}]});
}
function edge(s, pts, color=C.ink, arrow=true, dashed=false) {
  // Orthogonal paths remain editable PowerPoint freeform paths.
  if (dashed) {
    const x=Math.min(...pts.map(a=>a[0])),y=Math.min(...pts.map(a=>a[1]));
    const w=Math.max(1,Math.max(...pts.map(a=>a[0]))-x),h=Math.max(1,Math.max(...pts.map(a=>a[1]))-y);
    s.shapes.add({geometry:'custom',name:id('dashed flow'),position:{left:x,top:y,width:w,height:h},fill:'none',
      line:{fill:color,width:1.5,style:'dashed'},customPaths:[{width:w,height:h,commands:pts.map(([a,b],i)=>i?{lineTo:{x:a-x,y:b-y}}:{moveTo:{x:a-x,y:b-y}})}]});
  } else custom(s,pts,'none',color,1.8);
  if(arrow){const a=pts.at(-2),b=pts.at(-1),l=Math.hypot(b[0]-a[0],b[1]-a[1]),u=(b[0]-a[0])/l,v=(b[1]-a[1])/l;
    custom(s,[b,[b[0]-8*u+3.7*v,b[1]-8*v-3.7*u],[b[0]-8*u-3.7*v,b[1]-8*v+3.7*u]],color,'none',0);}
}
function dot(s,x,y,color=C.ink){s.shapes.add({geometry:'ellipse',name:id('junction'),position:{left:x-3,top:y-3,width:6,height:6},fill:color,line:{fill:'none',width:0}});}
function op(s,label,x,y,color=C.ink,r=20){
  s.shapes.add({geometry:'ellipse',name:id('operator '+label),position:{left:x-r,top:y-r,width:2*r,height:2*r},fill:C.cream,line:{fill:color,width:1.7}});
  txt(s,label,x-r,y-r,2*r,2*r,27,{math:true,color});
}
function tensor(s,x,y,w,h,fill,stroke,n=4){
  // Parallel oblique planes encode a feature tensor, without raster images.
  const gap=6,slant=Math.min(16,h/3);
  for(let i=n-1;i>=0;i--){const a=x+i*gap,b=y-i*3;
    custom(s,[[a,b+slant],[a+w,b],[a+w,b+h-slant],[a,b+h]],fill,stroke,.9);
    edge(s,[[a+3,b+slant+4],[a+3,b+h-4]],'#FFFFFF',false);
  }
  return {right:x+w+(n-1)*gap,cy:y+h/2};
}
function sub(s,base,index,x,y,size=27,color=C.ink){
  txt(s,base,x,y,size*.75,size*1.3,size,{math:true,color});
  txt(s,index,x+size*.66,y+size*.65,size*.6,size*.8,size*.65,{math:true,color});
}
function legend(s, y=840){
  rect(s,35,y-24,1530,64,C.white);
  tensor(s,65,y-6,16,25,C.green,C.greenEdge,3); txt(s,'Feature tensor',110,y-12,170,40,22);
  op(s,'+',330,y+8,C.ink,14);txt(s,'Add',350,y-12,80,40,22);
  op(s,'⊙',470,y+8,C.ink,14);txt(s,'Multiply',490,y-12,130,40,22);
  op(s,'C',665,y+8,C.ink,14);txt(s,'Concat',685,y-12,115,40,22);
  edge(s,[[840,y+8],[900,y+8]],C.red);txt(s,'Context / gate',910,y-12,175,40,22);
  edge(s,[[1130,y+8],[1190,y+8]],C.cyan);txt(s,'Feature path',1200,y-12,180,40,22);
}

{
  const s=p.slides.add(); s.background.fill=C.cream;
  s.speakerNotes.text='Implementation source: ultralytics/nn/modules/block.py, C2fAxisResidual, and ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml. The manuscript names this mechanism MCE. Only the final hidden branch of C2f is modulated, and only q=max(1,floor(c/4)) channels are active. The nano configuration uses two retained bottlenecks at P3 and P4. The descriptor depthwise convolutions have biases. The multiplication operator receives three inputs: the active feature, centered gate, and learned per-channel scale gamma, initialized to zero. Feature stacks are schematic tensor representations, not measured feature values. The design follows the pastel scientific-diagram reference image supplied in the conversation.';
  txt(s,'Morphology-aware Context Encoding (MCE)',25,10,1550,38,31);
  rect(s,25,58,1550,255,C.lavender,'none',20);
  txt(s,'C2f with MCE at P3/8 and P4/16',45,65,1510,35,25);
  // Main C2f aggregation retains the original Z0, Z1 and Z2 paths.
  edge(s,[[129,235],[190,235]],C.cyan);edge(s,[[245,235],[300,235]],C.cyan);
  edge(s,[[365,235],[420,235]],C.cyan);edge(s,[[550,235],[650,235]],C.cyan);
  edge(s,[[780,235],[900,235]],C.cyan);edge(s,[[975,235],[1130,235]],C.red);
  edge(s,[[1185,235],[1290,235]],C.cyan);edge(s,[[1345,235],[1430,235]],C.cyan);
  edge(s,[[332,200],[332,125],[1130,125]],C.greenEdge);
  edge(s,[[390,235],[390,160],[1130,160]],C.greenEdge);
  edge(s,[[600,235],[600,195],[1130,195]],C.greenEdge);
  dot(s,390,235,C.cyan);dot(s,600,235,C.cyan);
  tensor(s,60,200,50,70,C.blue,C.blueEdge,4);txt(s,'X',60,268,75,33,27,{math:true});
  vertical(s,'1 × 1 Conv',190,185,55,100,C.blue,C.blueEdge,22);
  block(s,'Split',300,200,65,70,C.yellow,C.yellowEdge,23);
  block(s,'Bottleneck 1',420,210,130,50,C.green,C.greenEdge,22);
  block(s,'Bottleneck 2',650,210,130,50,C.green,C.greenEdge,22);
  block(s,'MCE',900,200,75,70,C.pink,C.pinkEdge,27);
  vertical(s,'Concat',1130,108,55,160,C.purple,C.purpleEdge,24);
  vertical(s,'1 × 1 Conv',1290,185,55,100,C.blue,C.blueEdge,22);
  tensor(s,1430,200,50,70,C.purple,C.purpleEdge,4);txt(s,'Y',1435,268,70,33,27,{math:true});
  txt(s,'Z₀',760,100,60,25,22,{math:true});txt(s,'Z₁',760,135,60,25,22,{math:true});
  txt(s,'Z₂',820,170,60,25,22,{math:true});txt(s,'F',820,200,50,29,25,{math:true});txt(s,'F′',1020,200,60,29,25,{math:true});

  rect(s,25,330,1550,465,C.mint,'none',20);
  txt(s,'Partial-channel Morphology-aware Encoding',45,338,1510,38,27);
  // Context descriptors plus the two unchanged identity paths.
  edge(s,[[129,607],[185,607]],C.cyan);edge(s,[[240,600],[295,600]],C.cyan);
  edge(s,[[358,600],[390,600]],C.cyan,false);
  edge(s,[[390,600],[390,455],[460,455]],C.red);edge(s,[[390,600],[390,645],[460,645]],C.red);
  edge(s,[[510,455],[553,455]],C.red);edge(s,[[580,455],[655,455]],C.red);
  edge(s,[[510,645],[545,645]],C.red);edge(s,[[627,645],[655,645]],C.red);
  edge(s,[[710,455],[795,455],[795,530]],C.red);edge(s,[[710,645],[795,645],[795,570]],C.red);
  edge(s,[[815,550],[850,550]],C.red);edge(s,[[965,550],[1000,550]],C.red);
  edge(s,[[1100,550],[1180,550]],C.red);
  edge(s,[[390,455],[390,395],[1200,395],[1200,530]],C.cyan);
  edge(s,[[1200,625],[1200,570]],C.red);
  edge(s,[[1220,550],[1328,550],[1328,690]],C.red);
  edge(s,[[390,600],[390,710],[1308,710]],C.cyan);
  edge(s,[[1348,710],[1420,710]],C.red);
  edge(s,[[212,665],[212,765],[1420,765]],C.cyan);
  edge(s,[[1475,725],[1510,725]],C.cyan);
  dot(s,390,455,C.cyan);dot(s,390,600,C.cyan);
  tensor(s,60,572,50,70,C.blue,C.blueEdge,4);txt(s,'F',60,645,75,33,29,{math:true});
  txt(s,'c × H × W',47,682,140,30,21,{math:true});
  vertical(s,'Channel split',185,550,55,115,C.yellow,C.yellowEdge,22);
  tensor(s,295,565,45,70,C.green,C.greenEdge,4);sub(s,'F','a',310,635,25);
  txt(s,'q channels',277,520,100,28,21);
  vertical(s,'Mean W',460,415,50,80,C.yellow,C.yellowEdge,22);
  vertical(s,'Mean H',460,605,50,80,C.yellow,C.yellowEdge,22);
  tensor(s,553,415,9,80,C.pink,C.pinkEdge,4);
  tensor(s,545,638,70,14,C.pink,C.pinkEdge,3);
  txt(s,'q × H × 1',530,502,105,29,20,{math:true});
  txt(s,'q × 1 × W',530,670,105,29,20,{math:true});
  vertical(s,'DW 3 × 1',655,415,55,80,C.blue,C.blueEdge,22);
  vertical(s,'DW 1 × 3',655,605,55,80,C.blue,C.blueEdge,22);
  op(s,'+',795,550,C.red);txt(s,'Broadcast',735,577,125,28,20,{color:C.muted});
  block(s,'Hardsigmoid',850,510,115,80,C.pink,C.pinkEdge,20);
  block(s,'2g − 1',1000,510,100,80,C.yellow,C.yellowEdge,25);
  txt(s,'A',1130,510,50,31,27,{math:true,color:C.red});
  op(s,'⊙',1200,550,C.red);sub(s,'F','a',1212,440,25);
  txt(s,'γ',1180,625,40,39,31,{math:true,color:C.red});
  txt(s,'γ = 0 initially',1070,667,235,30,22,{color:C.muted});
  op(s,'+',1328,710,C.red);txt(s,'Fₐ′',1355,672,65,32,26,{math:true});
  vertical(s,'Concat',1420,670,55,110,C.purple,C.purpleEdge,23);
  tensor(s,1510,690,35,70,C.purple,C.purpleEdge,3);txt(s,'F′',1500,762,65,30,27,{math:true});
  sub(s,'F','b',780,725,24);txt(s,'Unchanged channels',820,724,220,31,22);
  txt(s,'Fₐ′ = Fₐ + γ ⊙ Fₐ ⊙ A',30,798,790,42,28,{math:true});
  txt(s,'q = max(1, ⌊c / 4⌋)',930,798,610,42,26,{math:true});
  legend(s,858);
}

{
  const s=p.slides.add(); s.background.fill=C.cream;
  s.speakerNotes.text='Implementation source: ultralytics/nn/modules/block.py, GatedInterpolationUpsample, and ultralytics/cfg/models/v8/yolov8n-fabric-axis-interp.yaml. The expression (1-G)*N + G*B is algebraically equivalent to the implemented N + G*(B-N). A biased Conv2d(C,1,3,padding=1) predicts the low-resolution gate, followed by Hardsigmoid and nearest upsampling. Weights start at zero and bias at -2.7, giving initial G=0.05. The gate is shared across all channels and replicated within 2x2 blocks. Bilinear interpolation uses align_corners=False. Output is subsequently concatenated with lateral P4 or P3 features, which do not enter the gate predictor. Feature stacks represent multi-channel tensors, while single planes represent the one-channel gate. The visual style follows the supplied reference image.';
  txt(s,'Gated Interpolation Upsampling (GIU)',25,10,1550,38,31);
  rect(s,25,58,1550,718,C.mint,'none',20);
  txt(s,'Content-dependent Reconstruction in the Top-down Feature Pyramid',50,70,1500,38,26);
  rect(s,290,155,945,200,'#E8F3F9','none',15);
  rect(s,290,575,945,190,'#FAE8EE','none',15);
  rect(s,290,385,795,180,'#FFF1C5','none',15);
  // Three independent branches start at the same coarse feature tensor.
  edge(s,[[178,465],[250,465]],C.cyan,false);
  edge(s,[[250,465],[250,265],[350,265]],C.cyan);
  edge(s,[[250,465],[350,465]],C.red);
  edge(s,[[250,465],[250,665],[350,665]],C.cyan);
  dot(s,250,465,C.cyan);
  edge(s,[[410,265],[510,265]],C.cyan);edge(s,[[600,265],[1150,265]],C.cyan);
  edge(s,[[410,665],[510,665]],C.pinkEdge);edge(s,[[600,665],[1150,665]],C.pinkEdge);
  edge(s,[[410,465],[520,465]],C.red);edge(s,[[575,465],[650,465]],C.red);
  edge(s,[[700,465],[800,465]],C.red);edge(s,[[860,465],[960,465]],C.red);
  edge(s,[[1025,465],[1080,465]],C.red,false);
  edge(s,[[1080,465],[1080,377],[1120,377]],C.red);
  edge(s,[[1170,350],[1170,285]],C.red);
  edge(s,[[1080,465],[1080,560],[1170,560],[1170,645]],C.red);
  dot(s,1080,465,C.red);
  edge(s,[[1190,265],[1365,265],[1365,445]],C.cyan);
  edge(s,[[1190,665],[1365,665],[1365,485]],C.pinkEdge);
  edge(s,[[1385,465],[1490,465]],C.cyan);
  tensor(s,75,425,73,80,C.blue,C.blueEdge,6);txt(s,'X',70,525,130,38,31,{math:true});
  txt(s,'C × H × W',55,565,175,33,24,{math:true});
  vertical(s,'Nearest × 2',350,205,60,120,C.blue,C.blueEdge,25);
  vertical(s,'Bilinear × 2',350,605,60,120,C.pink,C.pinkEdge,25);
  tensor(s,510,215,65,100,C.blue,C.blueEdge,5);txt(s,'N',515,170,100,36,29,{math:true});
  tensor(s,510,615,65,100,C.pink,C.pinkEdge,5);txt(s,'B',515,580,100,33,29,{math:true});
  txt(s,'C × 2H × 2W',690,282,290,36,25,{math:true});
  txt(s,'C × 2H × 2W',690,682,290,36,25,{math:true});
  vertical(s,'3 × 3 Conv',350,405,60,120,C.yellow,C.yellowEdge,25);
  txt(s,'C → 1',341,532,80,28,21);
  vertical(s,'Hardsigmoid',520,405,55,120,C.pink,C.pinkEdge,23);
  tensor(s,650,435,50,60,C.yellow,C.yellowEdge,1);
  txt(s,'1 × H × W',610,510,150,29,22,{math:true});
  vertical(s,'Nearest × 2',800,405,60,120,C.yellow,C.yellowEdge,25);
  tensor(s,960,425,65,80,C.yellow,C.yellowEdge,1);
  txt(s,'G',960,385,65,34,29,{math:true,color:C.red});
  txt(s,'1 × 2H × 2W',890,526,180,30,22,{math:true});
  block(s,'1 − G',1120,350,100,55,C.yellow,C.yellowEdge,27);
  op(s,'⊙',1170,265,C.cyan);op(s,'⊙',1170,665,C.pinkEdge);op(s,'+',1365,465,C.greenEdge);
  tensor(s,1490,425,50,80,C.purple,C.purpleEdge,4);txt(s,'Y',1490,523,75,38,31,{math:true});
  txt(s,'C × 2H × 2W',1370,570,205,35,24,{math:true});
  txt(s,'Shared across channels',1110,505,230,35,21,{color:C.muted});
  txt(s,'Y = (1 − G) ⊙ N + G ⊙ B',45,787,970,50,33,{math:true});
  txt(s,'Initial G = 0.05',1110,790,420,40,26);
  rect(s,35,843,1530,45,C.white);
  txt(s,'Bilinear: align_corners = False',45,847,490,33,23);
  txt(s,'Gate: nearest replication',570,847,440,33,23);
  txt(s,'Output: fusion with lateral P4 / P3',1010,847,540,33,23);
}

await fs.mkdir(out,{recursive:true});
for(const [i,s] of p.slides.items.entries()){
  const b=await p.export({slide:s,format:'png',scale:1.5});
  await fs.writeFile(path.join(out,`${i===0?'MCE':'GIU'}_reference_preview.png`),new Uint8Array(await b.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const skill=process.env.PRESENTATIONS_SKILL_DIR;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({workspaceDir:path.resolve(build,'../..'),candidatePath:candidate,
  finalPath:path.join(out,'MCE_GIU_Reference_Style_v2.pptx'),pythonExecutable:process.env.CODEX_RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit'],
  requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'design',families:['Times New Roman','Cambria Math']},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'validation-v2.json')});
console.log('Created reference-style MCE/GIU PPTX');
