import argparse
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from external.nms import nms
from external.sort import Sort
import json
import numpy as np
import os
from utils import mask_frame, parse_outputs, regionize_image
from xml.etree import ElementTree


def parse_args():
    parser = argparse.ArgumentParser(
        description='Example detection and tracking script')
    parser.add_argument('-v', '--video', required=True,
                        help='Input video')
    parser.add_argument('-c', '--config', required=True,
                        help='Detection model configuration')
    parser.add_argument('-m', '--mask', required=True,
                        help='Mask for the video')
    return parser.parse_args()


def main(args):
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
    predictor = DefaultPredictor(cfg)
    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()

    tracker = Sort(max_age=5)
    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    trajectories = {}
    rendering = {}
    frame_index = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        print('Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
        success, frame = cap.read()
        if not success:
            break
        frame_masked = mask_frame(frame, mask)

        image_regions = regionize_image(frame_masked)
        bboxes = []
        scores = []
        for _image, _offset in image_regions:
            _outputs = predictor(_image)
            _bboxes, _scores, _ = parse_outputs(_outputs, _offset)
            bboxes += _bboxes
            scores += _scores
        nms_threshold = config['nms_threshold']
        nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
        detections = np.zeros((len(nms_bboxes), 5))
        detections[:, 0:4] = nms_bboxes
        detections[:, 4] = nms_scores

        tracked_objects = tracker.update(detections)
        rendering[frame_index] = []
        for tracked_object in tracked_objects:
            tl = (int(tracked_object[0]), int(tracked_object[1]))
            br = (int(tracked_object[2]), int(tracked_object[3]))
            object_index = int(tracked_object[4])
            if object_index not in trajectories:
                trajectories[object_index] = []
            trajectories[object_index].append([
                frame_index, tl[0], tl[1], br[0], br[1]])
            rendering[frame_index].append([
                object_index, tl[0], tl[1], br[0], br[1]])

        frame_index = frame_index + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    scenario = args.video.replace('videos/', '').replace('.mp4', '')
    with open('output/{}_t.json'.format(scenario), 'w') as fp:
        json.dump(trajectories, fp)
    with open('output/{}_r.json'.format(scenario), 'w') as fp:
        json.dump(rendering, fp)


if __name__ == '__main__':
    main(parse_args())
