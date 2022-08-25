import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
import json
import os
import torch
setup_logger()
print(
    'Torch version:', torch.__version__, 
    'CUDA availability:', torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser(description='Example train script')
    parser.add_argument('-c', '--config', required=True,
                        help='Detection model configuration')
    return parser.parse_args()


def main(args):
    dataset_name = 'b3d_train'
    annotation_path = 'vision/annotations/train.json'
    image_path = 'vision/images/train'
    register_coco_instances(dataset_name, {}, annotation_path, image_path)
    MetadataCatalog.get(dataset_name).thing_classes = ['vehicle']

    with open(args.config) as fp:
        config = json.load(fp)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['config']))
    cfg.DATASETS.TRAIN = ('b3d_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = config['dataloader_num_workers']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['config'])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['batch_size_per_image']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
    cfg.SOLVER.IMS_PER_BATCH = config['ims_per_batch']
    cfg.SOLVER.BASE_LR = config['base_lr']
    cfg.SOLVER.MAX_ITER = config['max_iter']

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main(parse_args())
