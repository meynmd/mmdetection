import os
import json
from glob import glob
from argparse import ArgumentParser

from tqdm import tqdm
from PIL import Image


def get_args():
    
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('save_file', help='path to save list of files')
    args = parser.parse_args()
    return args


def main():

    opts = get_args()
    
    out_path = opts.save_file
    imgfiles = glob(os.path.join(opts.img_dir, '*'))
    
    wh, included = None, []
    for _, imgf in enumerate(tqdm(imgfiles)):
        imgf = os.path.abspath(imgf)
        stats = os.stat(imgf)
        try:
            img = Image.open(imgf)
        except:
            print('error reading {}'.format(imgf))
            continue
        w, h = img.width, img.height
        n_pix = w*h
        if n_pix < 720*480:
            # print('file {} low resolution: excluding'.format(imgf))
            continue
        if stats.st_size < 5*1048 or stats.st_size/n_pix < 0.01:
            # print('file {} too small: excluding'.format(imgf))
            continue

        if wh is None:
            wh = (w, h)
        elif wh != (w, h):
            # print('inconsistent image dimensions: excluding {}'.format(imgf))
            continue

        included.append(imgf)

    if len(included) < 90:
        print('too few images; not writing image list')
    else:
        print('writing {}'.format(out_path))
        with open(out_path, 'w') as wp:
            for filename in included:
                wp.write('{}\n'.format(filename))

if __name__ == '__main__':
    main()
