import argparse
import cv2
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from xml.etree import ElementTree


def parse_args():
    parser = argparse.ArgumentParser(description='Example masking script')
    parser.add_argument('-i', '--image', required=True,
                        help='Sample image')
    parser.add_argument('-m', '--mask', required=True,
                        help='Specification of the mask')
    return parser.parse_args()


def visualize_masking(image, domain_poly):
    image = image[:, :, ::-1]
    fig = plt.figure(dpi=300, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(image)
    patch = patches.PathPatch(
        domain_poly, facecolor='r', alpha=0.5, edgecolor='none')
    ax.add_patch(patch)
    plt.savefig('output/mask_overlay.png', bbox_inches='tight', pad_inches=0)


def main(args):
    tree = ElementTree.parse(args.mask)
    root = tree.getroot()
    domain = root.find('.//polygon[@label="domain"]').attrib['points']
    domain = domain.replace(';', ',')
    domain = np.array([
        float(pt) for pt in domain.split(',')]).reshape((-1, 2))
    tl = (int(np.min(domain[:, 1])), int(np.min(domain[:, 0])))
    br = (int(np.max(domain[:, 1])), int(np.max(domain[:, 0])))
    domain_poly = Path(domain)

    image = cv2.imread(args.image)

    visualize_masking(image, domain_poly)

    width, height = int(image.shape[1]), int(image.shape[0])
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T
    bitmap = domain_poly.contains_points(pixel_points)
    bitmap = bitmap.reshape((height, width))
    image[bitmap == 0] = 0
    image_masked = image[tl[0]:br[0], tl[1]:br[1], :]
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/masked_image.png', image_masked)


if __name__ == '__main__':
    main(parse_args())
