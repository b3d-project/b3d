from matplotlib.path import Path
import numpy as np


def mask_frame(frame, mask):
    domain = mask.find('.//polygon[@label="domain"]').attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)
    width, height = int(frame.shape[1]), int(frame.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width))
    frame[bitmap == 0] = 0
    frame_masked = frame[tl[0]:br[0], tl[1]:br[1], :]
    return frame_masked


def parse_outputs(outputs, offset):
    instances = outputs['instances'].to('cpu')
    bboxes = []
    scores = []
    classes = []
    for bbox, score, pred_class in zip(
            instances.pred_boxes, instances.scores, instances.pred_classes):
        bbox[0] += offset[0]
        bbox[1] += offset[1]
        bbox[2] += offset[0]
        bbox[3] += offset[1]
        bboxes.append(bbox.numpy())
        scores.append(score.numpy())
        classes.append(pred_class.numpy())
    return bboxes, scores, classes


def regionize_image(image):
    height, width, _ = image.shape
    split_width = width
    while(split_width / height > 4):
        split_width = int(split_width / 2)
    batch = []
    covered_width = 0
    while(covered_width < width):
        stop_width = min(covered_width + split_width, width)
        if (stop_width - covered_width < 0.75 * split_width):
            break
        batch.append(
            [image[:, covered_width:stop_width, :], (covered_width, 0)])
        covered_width = min(covered_width + int(split_width / 2), width)
    return batch
