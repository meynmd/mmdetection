import os

from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    img_name = os.path.splitext(os.path.basename(args.img))[0]
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, 
                       save=os.path.join(args.save, '{}_det.jpg'.format(img_name)))


if __name__ == '__main__':
    main()
