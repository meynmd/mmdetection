_base_ = './cascade_rcnn_r50_fpn_1x_crowdhuman.py'
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=28)
