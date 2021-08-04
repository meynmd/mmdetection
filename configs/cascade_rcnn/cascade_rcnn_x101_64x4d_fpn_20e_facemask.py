_base_ = './cascade_rcnn_r50_fpn_20e_coco_2class.py'
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
classes = ('Mask', 'No-Mask')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/facemask/face-mask-combined-dataset',
        ann_file='annotations/train.json',
        img_prefix='images/train/',
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        data_root='/proj/vacsine/matt/data/facemask/face-mask-combined-dataset',
        ann_file='annotations/val.json',
        img_prefix='images/val/',
        classes=classes
    ),
    test=dict(
        samples_per_gpu=4
    )
)

load_from = '/proj/vacsine/matt/detect/mmdetection/pretrained/cascade_rcnn_x101_64x4d_fpn/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
