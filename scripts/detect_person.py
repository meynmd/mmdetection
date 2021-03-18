import os
import json
from glob import glob
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

from tools.inference import get_model, estimate_boxes, visualize
from resources.models import MODELS


def get_args():
    
    parser = ArgumentParser()
    parser.add_argument('-i', '--img_dir', help='Image directory')
    parser.add_argument('-m', '--model_type', default='cascade_rcnn',
                        help='model name from resources.models.MODELS')
    parser.add_argument('--device', default='cuda:0', 
                        help='Device used for inference')
    parser.add_argument('-t', '--thresholds', 
                        type=float, nargs='*', default=[0.3], 
                        help='bbox score thresholds')
    parser.add_argument('-b', '--batch_size', default=16)
    parser.add_argument('-s', '--save', default=None)
    args = parser.parse_args()

    return args


def compute_detections(model, img_list, batch_size=64, threshold=0., 
                       classes=[0]):

    imgfiles = np.array(img_list)
    n_batches = int(np.ceil(imgfiles.size/batch_size))
    img_batches = np.array_split(imgfiles, n_batches)
    
    boxes, labels = [], []
    for i in tqdm(range(n_batches), desc='computing detections'):
        batch = img_batches[i]
        bb, bl = estimate_boxes(model, batch.tolist(), threshold, classes)
        # import pdb; pdb.set_trace()
        boxes += bb
        labels += bl
    
    return boxes, labels


def main():

    opts = get_args()
    model_opts = MODELS[opts.model_type]
    
    out_dir = opts.save
    out_img_dir = os.path.join(opts.save, 'visualization')
    if opts.save and not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

    # build the model from a config file and a checkpoint file
    model = get_model(model_opts['config'], checkpoint=model_opts['checkpoint'], 
                      device=opts.device)

    imgfiles = glob(os.path.join(opts.img_dir, '*'))
    
    # boxes: list of ndarrays size (N_DET, 5) 
    boxes, class_labels = compute_detections(model, imgfiles, 
                                              batch_size=opts.batch_size)

    # show the results
    print('writing results...')
    results = []
    for imgf, box, label in zip(imgfiles, boxes, class_labels):
        img_name = os.path.splitext(os.path.basename(imgf))[0]
        for threshold in opts.thresholds:
            out_path = os.path.join(out_img_dir, 
                                    '{}.det.{:.2f}.jpg'.format(img_name, 
                                                               threshold))
            score_mask = box[:, -1] >= threshold
            box_t, label_t = box[score_mask], label[score_mask]
            visualize(imgf, box_t, label_t, out_file=out_path)

        box = box[box[:, -1] >= 1e-2]
        detection_annotation = {
            'image_filename': os.path.basename(imgf),
            'detection_threshold': threshold,
            'n_detections': box.shape[0],
            'detections': []
        }
        for i in range(box.shape[0]):
            detection_annotation['detections'].append({
                'x_min': str(box[i, 0]),
                'y_min': str(box[i, 1]),
                'width': str(box[i, 2]),
                'height': str(box[i, 3]),
                'confidence': '{:.4f}'.format(box[i, 4])
            })
        results.append(detection_annotation)

    results_path = os.path.join(opts.save, 'detections.json')
    with open(results_path, 'w') as wp:
        json.dump(results, wp, indent=4)
    print('wrote detection results to: {}'.format(results_path))

if __name__ == '__main__':
    main()
