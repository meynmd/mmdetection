_base_ = './cascade_rcnn_r50_fpn_20e_coco.py'
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
        style='pytorch'))

dataset_type = 'CocoDataset'
classes = ('person',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/coco/',
        ann_file='instances_train2017.json',
        img_prefix='images/train2017/',
        samples_per_gpu=1
    ),
    val=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/coco/',
        ann_file='instances_train2017.json',
        img_prefix='images/val2017/',
        samples_per_gpu=8
    ),
    test=dict(
        samples_per_gpu=8
    )
)
