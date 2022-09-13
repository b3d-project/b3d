import argparse
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plt.rcParams['font.family'] = 'FreeSans'


def parse_args():
    parser = argparse.ArgumentParser(description='Example test script')
    parser.add_argument('-i', '--image', required=True,
                        help='Sample image')
    parser.add_argument('-c', '--config', required=True,
                        help='Detection model configuration')
    return parser.parse_args()


def visualize_outputs(image, outputs, save_path):
    instances = outputs['instances'].to('cpu')
    cmap = plt.cm.get_cmap('terrain', len(instances.pred_boxes))
    fig = plt.figure(dpi=400, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(image)
    for index, (box, score) in enumerate(zip(
            instances.pred_boxes, instances.scores)):
        origin = (box[0], box[1])
        width = box[2] - box[0]
        length = box[3] - box[1]
        rect = patches.Rectangle(
            origin, width, length, 
            linewidth=2, edgecolor=cmap(index),
            facecolor='w', alpha=0.5)
        ax.add_patch(rect)
        ax.text(
            box[0] + 2, box[3] - 5, '{:.2f}'.format(score), 
            color='k', size='small')
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
    outputs = predictor(image)
    save_path = os.path.join(cfg.OUTPUT_DIR, 'out.jpg')
    visualize_outputs(image, outputs, save_path)


if __name__ == '__main__':
    main(parse_args())
