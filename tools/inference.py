import os

import numpy as np
import mmcv

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def get_model(config, checkpoint, device):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    return model


def estimate_boxes(model, img, score_thr=0.3, classes=[0]):

    result = inference_detector(model, img)

    all_bboxes, all_labels = [], []
    for img_result in result:

        img_result = [img_result[c] for c in classes]
        labels = [
            np.full(boxes.shape[0], classes[i], dtype=np.int32)
            for i, boxes in enumerate(img_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.concatenate(img_result, axis=0)

        score_mask = bboxes[:, -1] >= score_thr
        bboxes, labels = bboxes[score_mask], labels[score_mask]
        # import pdb; pdb.set_trace()
        all_bboxes.append(bboxes)
        all_labels.append(labels)

    return all_bboxes, all_labels


def visualize(imgfile, bboxes, labels, out_file=None):

    img = mmcv.imread(imgfile)        
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        bbox_color='green',
        text_color='blue',
        # thickness=thickness,
        font_scale=0.25,
        show=False,
        out_file=out_file)

    return img

