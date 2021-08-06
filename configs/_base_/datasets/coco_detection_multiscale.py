dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(240, 136), ratio_range=(0.5, 1.), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(240, 136),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(240, 136), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('Mask', 'No-Mask')
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/facemask/face-mask-combined-dataset',
        ann_file='annotations/train.json',
        img_prefix='images/train/',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/facemask/face-mask-combined-dataset',
        ann_file='annotations/val.json',
        img_prefix='images/val/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/facemask/face-mask-combined-dataset',
        ann_file='annotations/val.json',
        img_prefix='images/val/',
        classes=classes,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='bbox')
