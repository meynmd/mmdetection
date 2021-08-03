MODELS = {
    'cascade_rcnn': {
        'config': 'configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py',
        'checkpoint': 'pretrained/cascade_rcnn_x101_64x4d_fpn/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
    },
    'cascade_rcnn_face': {
        'config': 'configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_facemask.py',
        'checkpoint': '/proj/vacsine/matt/detect/mmdetection/work_dirs/cascade_rcnn_x101_64x4d_fpn_20e_facemask/epoch_20.pth'
    }
}
