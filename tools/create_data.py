# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import json
import codecs
import glob
import warnings
import matplotlib.pyplot as plt

import mmcv
import numpy as np
from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='create cityscape-like dataset')
    parser.add_argument(
        '--avm-dir',
        default=False,
        type=str,
        help='Directory of AVM dataset')
    parser.add_argument(
        '--output-dir',
        default='./output',
        type=str,
        help='Directory of output dataset')
    args = parser.parse_args()
    return args


def save_data(avm_dir, output_dir):
    output_img_path = os.path.join(output_dir, 'train', 'images')
    output_anno_path = os.path.join(output_dir, 'train', 'annfiles')
    output_seg_path = os.path.join(output_dir, 'train', 'segments')
    avm_img_path = os.path.join(avm_dir, 'images')
    avm_det_anno_path = os.path.join(avm_dir, 'det_annotations')
    avm_seg_path = os.path.join(avm_dir, 'mask')
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    if not os.path.exists(output_anno_path):
        os.makedirs(output_anno_path)
    if not os.path.exists(output_seg_path):
        os.makedirs(output_seg_path)
    print('Saving Data ...')

    filename_list = os.listdir(avm_img_path)
    for img_filename in filename_list:
        avm_img_file = os.path.join(avm_img_path, img_filename, 'avm')
        avm_det_anno_file = os.path.join(avm_det_anno_path, img_filename)
        avm_seg_file = os.path.join(avm_seg_path, img_filename)
        img_files = sorted(os.listdir(avm_img_file))
        det_anno_files = os.listdir(avm_det_anno_file)
        seg_files = os.listdir(avm_seg_file)

        print(f'\nProcessing {img_filename}: ')
        progress_bar = mmcv.ProgressBar(len(img_files))
        for img_name in img_files:
            # assert img_name.split('.')[0] == anno_name.split('.')[0]
            # for debug
            basename = img_name.split('.')[0]
            anno_name = basename + '.json'
            seg_name = basename + '.png'
            if (anno_name not in det_anno_files) or (seg_name not in seg_files):
                progress_bar.update()
                continue

            img_path = os.path.join(avm_img_file, img_name)
            anno_path = os.path.join(avm_det_anno_file, anno_name)
            seg_path = os.path.join(avm_seg_file, seg_name)

            img = mmcv.imread(img_path)
            save_img_file = img_filename + '_' + basename + '.png'
            save_img_path = os.path.join(output_img_path, save_img_file)
            mmcv.imwrite(img, save_img_path)

            seg = mmcv.imread(seg_path)[..., -1]
            save_seg_path = os.path.join(output_seg_path, save_img_file)
            mmcv.imwrite(seg, save_seg_path)

            with open(anno_path, 'r') as load_f:
                anno_dict = json.load(load_f)
            save_anno_file = img_filename + '_' + img_name.split('.')[0] + '.txt'
            save_anno_path = os.path.join(output_anno_path, save_anno_file)

            bboxes_num = len(anno_dict['annotations'])
            with codecs.open(save_anno_path, 'w', 'utf-8') as f_out:
                if bboxes_num == 0:
                    pass
                else:
                    for idx in range(bboxes_num):
                        diffs = '0'
                        anno = anno_dict['annotations'][idx]
                        polygon = anno['polygon']
                        label = 'unavailable_parking_lines' if anno['category'] == 0 else 'available_parking_lines'

                        entry = np.array(anno['entry'])
                        polygon_array = np.array(polygon).reshape(-1, 2)
                        inv_polygon_array = polygon2rect(entry, polygon_array)
                        polygon = np.around(inv_polygon_array, 1).reshape(-1).tolist()

                        # vis_polygon(img, polygon_array)
                        # vis_polygon(img, inv_polygon_array)

                        outline = ' '.join(list(map(str, polygon)))
                        outline = outline + ' ' + label + ' ' + diffs
                        f_out.write(outline + '\n')
            progress_bar.update()

    print('\nDone')


def vis_polygon(img, polygon_array, color='red'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    poly = plt.Polygon(polygon_array, color=color, alpha=0.3)
    ax.add_patch(poly)
    plt.imshow(img)
    plt.show()


def polygon2rect(entry, polygon_array):
    delta_xy = entry[:2] - entry[2:]
    theta = np.pi / 2 - np.arctan2(delta_xy[1], delta_xy[0])
    rot_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    inv_rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rot_polygon_array = np.dot(polygon_array, rot_matrix)
    min_xy = rot_polygon_array.min(0)
    max_xy = rot_polygon_array.max(0)
    rot_rect_array = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]]
    ])  # in clockwise order
    inv_polygon_array = np.dot(rot_rect_array, inv_rot_matrix)
    return inv_polygon_array

def main():
    args = parse_args()
    avm_dir = args.avm_dir
    output_dir = args.output_dir

    save_data(avm_dir, output_dir)
    # vis_polygon(avm_dir, output_dir)


if __name__ == '__main__':
    main()
