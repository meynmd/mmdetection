_base_ = './cascade_rcnn_r50_fpn_20e_crowdhuman.py'
model = dict(
    type='CascadeRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
)

# dataset settings
dataset_type = 'CocoDataset'
classes = ('person',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        data_root='/proj/vacsine/matt/data/crowdhuman/',
        ann_file='annotation_train.json',
        img_prefix='train/Images/',
        # samples_per_gpu=2,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        data_root='/proj/vacsine/matt/data/crowdhuman/',
        ann_file='annotation_val.json',
        img_prefix='val/Images/',
        # samples_per_gpu=12,
    ),
    test=dict(
        # samples_per_gpu=12,
    )
)

# optimizer
# lr is set for a batch size of 2
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[2, 4])
total_epochs = 28  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)

load_from = '/proj/vacsine/matt/detect/mmdetection/pretrained/cascade_rcnn_x101_64x4d_fpn/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
