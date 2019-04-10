#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a
   separate tsv file that can be merged later (e.g. by using merge_tsv function).
   Modify the load_image_ids script as necessary for your data location. """

# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014

## N.B.: Several paths need to be in the PYTHONPATH, we add them here so we make sure that the scripts is going ton use the correct imports
import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')

import argparse
import caffe
import csv
import cv2
import gzip
import h5py
import json
import numpy as np
import os
import pprint
import random
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.config import cfg_from_list
import glob
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

guesswhat_files = {
    "train": "guesswhat.train.jsonl.gz",
    "valid": "guesswhat.valid.jsonl.gz",
    "test": "guesswhat.test.jsonl.gz"
}
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36
FEATURE_MAP_SIZE = 2048


def read_raw_dataset_games(file_path):
    if file_path.endswith(".gz"):
        with gzip.open(file_path) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip("\n"))

                yield game
    else:
        with open(file_path) as f:
            for line in f:
                game = json.loads(line.strip("\n"))

                yield game


def load_image_ids(data_path=None, img_path=None):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []

    loaded_images = set()

    for game in tqdm(read_raw_dataset_games(data_path)):
        image_id = game["image"]["id"]
        if image_id not in loaded_images:
            filepath = os.path.join(img_path, str(image_id) + '.jpg')
            split.append((filepath, image_id))
            loaded_images.add(image_id)

    return split


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes]
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        required=True, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        required=True, type=str)
    parser.add_argument('--output_path', dest='output_path',
                        help='output filepath',
                        required=True, type=str)
    parser.add_argument('--guesswhat_folder', dest='guesswhat_folder',
                        required=True, help='Guesswhat folder',type=str)
    parser.add_argument('--img_dir', required=True, dest='img_dir', help='MSCOCO image directory', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def generate_features(gpu_id, prototxt, weights, image_ids, boxes_dataset, features_dataset):
    if gpu_id >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    for i, (im_file, image_id) in enumerate(tqdm(image_ids)):
        proc_image = get_detections_from_im(net, im_file, image_id)
        boxes_dataset[i] = proc_image["boxes"]
        features_dataset[i] = proc_image["features"]


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    for split_name, split_file in guesswhat_files.items():
        subdir = "val" if split_name == "valid" else split_name
        feat_h5_file = h5py.File(os.path.join(args.output_path, split_name) + "_features.h5", 'w')
        idx2img = []
        image_index = 0

        image_ids = load_image_ids(os.path.join(args.guesswhat_folder, split_file), args.img_dir)
        num_images = len(image_ids)
        print("Number of images for {} set: {}".format(split_name, num_images))

        '''
        The original FastRCNN feature files contained the following fields:
        {
            'image_id': image_id,
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes': len(keep_boxes),
            'boxes': base64.b64encode(cls_boxes[keep_boxes]),
            'features': base64.b64encode(pool5[keep_boxes])
        }
        '''

        boxes_dataset = feat_h5_file.create_dataset(
            name="boxes",
            dtype='float32',
            shape=(num_images, MAX_BOXES, 4)
        )

        features_dataset = feat_h5_file.create_dataset(
            name="features",
            dtype='float32',
            shape=(num_images, MAX_BOXES, FEATURE_MAP_SIZE)
        )

        feat_h5_file.create_dataset(name="idx2img", dtype='int64', data=np.array([image_id for (_, image_id) in image_ids]))

        random.seed(10)

        generate_features(
            gpu_id,
            args.prototxt,
            args.caffemodel,
            image_ids,
            boxes_dataset,
            features_dataset
        )
