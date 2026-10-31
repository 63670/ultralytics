# 同步划分 YOLO / COCO 检测数据集

在仓库根目录运行（需要 Python 和 Pillow，Ultralytics 环境已包含 Pillow）：

```powershell
python scripts/split_hybrid_dataset.py --source "C:\Users\tankz\Desktop\pingwen_hybrid" --output "datasets/pingwen_hybrid_split" --seed 42
```

输入目录结构：

```text
pingwen_hybrid/
├── classes.txt                     # 每行一个类别，顺序对应 YOLO 的 0 起始类别 ID
├── images/                         # 图片，可以包含子目录及不参与划分的原始标注 JSON
├── yolo_labels/                    # 相同相对路径、同名 TXT；背景图片也必须有空 TXT
└── coco_labels/coco_detection.json
```

输出目录结构：

```text
pingwen_hybrid_split/
├── split_manifest.json             # 种子、比例、各子集完整文件清单和类别统计
├── yolo/
│   ├── data.yaml
│   ├── classes.txt
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── coco/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations/
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── instances_test2017.json
```

脚本仅随机打乱一次，然后将同一份 train/val/test 列表用于两种格式。固定种子和相同输入会得到相同划分。按 7:1:2 用最大余数法分配整数张数；534 张对应 **374 / 53 / 107** 张。图片按字节复制，原始文件保留。已有输出目录会报错，重新导出时请指定新的输出目录。

脚本先核对类别名、实际图片尺寸、框的位置与数量。源 YOLO 标签按六位小数导出的精度比较，归一化坐标容差为 `1e-6`。校验通过后，以 COCO 框为共同来源生成两种格式，YOLO 写出十位小数，避免各自转换产生不同标注。COCO 类别 ID 保留，YOLO 类别 ID 取 `classes.txt` 的行序，COCO 的类别列表也按该行序排列。

未列入 COCO 且 YOLO TXT 为空的图片会作为背景样本加入两套数据集。COCO 中已有但没有框的图片同样保留。缺少 TXT、未列入 COCO 的非空标签、标签不一致、无效框、crowd/ignore 标注会报错。重复的 COCO 标注 ID 会重新生成唯一 ID；图片 ID 也重新编号。只处理目标检测，输出 YOLO 标签为 `class cx cy width height`。

YOLO 训练时使用输出的 `yolo/data.yaml`。COCO 的 `file_name` 相对于对应的 `train2017/`、`val2017/` 或 `test2017/`，训练框架的图片前缀应指向相应目录。

运行回归测试：

```powershell
python -m unittest scripts.test_split_hybrid_dataset -v
```
