import argparse
import cv2
import json
from utils import mask_frame
from xml.etree import ElementTree
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Example rendering script')
    parser.add_argument('-v', '--video', required=True,
                        help='Input video')
    parser.add_argument('-r', '--rendering', required=True,
                        help='Data for rendering detected and tracking results')
    parser.add_argument('-m', '--mask', required=True,
                        help='Mask for the video')
    return parser.parse_args()


def main(args):
    tree = ElementTree.parse(args.mask)
    mask = tree.getroot()
    cap = cv2.VideoCapture(os.path.expanduser(args.video))
    with open(args.rendering) as fp:
        rendering = json.load(fp)
    frame_index = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    while cap.isOpened():
        print('Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
        success, frame = cap.read()
        if not success:
            break
        masked_frame = mask_frame(frame, mask)
        tracked_objects = rendering['{:d}'.format(frame_index)]
        for tracked_object in tracked_objects:
            object_index = int(tracked_object[0])
            tl = (int(tracked_object[1]), int(tracked_object[2]))
            br = (int(tracked_object[3]), int(tracked_object[4]))
            cv2.rectangle(masked_frame, tl, br, (255, 0, 0), 2)
            cv2.putText(
                masked_frame, '{:d}'.format(object_index), (br[0]+10, br[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        display_width = int(masked_frame.shape[1] * 0.5)
        display_height = int(masked_frame.shape[0] * 0.5)
        resized_frame = cv2.resize(
            masked_frame, (display_width, display_height))
        if out is None:
            scenario = args.video.replace('videos/', '').replace('.mp4', '')
            out = cv2.VideoWriter(
                'output/{}.mp4'.format(scenario),
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                (display_width,display_height))
        out.write(resized_frame)

        # cv2.imshow('Frame', resized_frame)
        frame_index = frame_index + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_args())
