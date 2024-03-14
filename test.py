import argparse
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from external.nms import nms
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from utils import parse_outputs, regionize_image

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = \
    ['FreeSans'] + plt.rcParams['font.sans-serif']


def parse_args():
    parser = argparse.ArgumentParser(description='Example test script')
    parser.add_argument('-i', '--image', required=True,
                        help='Sample image')
    parser.add_argument('-c', '--config', required=True,
                        help='Detection model configuration')
    return parser.parse_args()


def visualize_outputs(image, bboxes, scores, save_path):
    fig = plt.figure(dpi=400, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(image)
    cmap = plt.cm.get_cmap('terrain', len(bboxes))
    for index, (bbox, score) in enumerate(zip(bboxes, scores)):
        origin = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        length = bbox[3] - bbox[1]
        rect = patches.Rectangle(
            origin, width, length,
            linewidth=2, edgecolor=cmap(index),
            facecolor='w', alpha=0.5)
        ax.add_patch(rect)
        ax.text(
            bbox[0] + 2, bbox[3] - 5,
            '{:.2f}'.format(score), color='k', fontsize=3.0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def main(args):
    dataset_name = 'b3d_test'
    annotations_path = 'vision/annotations/test.json'
    images_path = 'vision/images/test'
    register_coco_instances(dataset_name, {}, annotations_path, images_path)
    MetadataCatalog.get(dataset_name).thing_classes = ['vehicle']

    with open(args.config) as fp:
        config = json.load(fp)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['config']))
    cfg.MODEL.WEIGHTS = config['weights']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
    cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']

    image_path = args.image
    image = cv2.imread(image_path)
    predictor = DefaultPredictor(cfg)
    image_regions = regionize_image(image)
    bboxes = []
    scores = []
    for _image, _offset in image_regions:
        _outputs = predictor(_image)
        _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
        bboxes += _bboxes
        scores += _scores
    nms_threshold = config['nms_threshold']
    nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)

    save_path = os.path.join(cfg.OUTPUT_DIR, 'out.jpg')
    visualize_outputs(image, nms_bboxes, nms_scores, save_path)


if __name__ == '__main__':
    main(parse_args())
