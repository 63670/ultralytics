"""Deformable DETR-R50 baseline for the 3-class fabric-defect COCO dataset.

Copy this file to <mmdetection>/configs/fabric/ before training.
"""

_base_ = '../deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

data_root = '/home/tkz/datasets/pingwen_coco/'
classes = ('row', 'col', 'hole')
metainfo = dict(classes=classes)

model = dict(bbox_head=dict(num_classes=len(classes)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test2017.json')

max_epochs = 300
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[240, 270], gamma=0.1),
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10, save_best='coco/bbox_mAP', rule='greater'))
randomness = dict(seed=0, deterministic=True)
