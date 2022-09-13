# Berkeley DeepDrive Drone Dataset

## Introduction

The Berkeley DeepDrive Drone (B3D) Dataset allows researchers to study implicit driving etiquette in *understructured* road environments.
The dataset consists:
1. A set of 20 aerial videos recording understructured driving,
2. A collection of 16002 images and annotations to train vehicle detection models, and
3. A few example scripts for illustrating typical usages.

To download the videos and annotated images, run
```
pip install gdown
python download.py
```

After downloading, the *full* structure of the dataset repository should be as follows:
```
.
├── configs
│   ├── config_quick.json
│   └── config_refined.json
├── Dockerfile
├── download.py
├── LICENSE
├── README.md
├── test.py
├── train.py
├── videos
│   └── <20 mp4 files>
└── vision
    ├── annotations
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    └── images
        ├── test
        │   └── <1636 jpg files>
        ├── train
        │   └── <12700 jpg files>
        └── val
            └── <1666 jpg files>
```

## Getting Started
We recommend running the script in a Docker container.
Please follow the instructions [here](https://docs.docker.com/engine/install/) to install Docker and 
instructions [here](https://github.com/NVIDIA/nvidia-docker) to install NVIDIA Container Toolkit.

After installing Docker and NVIDIA Container Toolkit, build the required Docker image
```
docker build -t detectron2:latest .
```

## Usage
To inspect and edit the annotations, please use the open source image annotation tool CVAT. 
Note that the training dataset might need to be split into several smaller datasets for it to be properly parsed by CVAT.

To train a vehicle detection model using the annotated images, one could use the Detectron2 library.
The example `train.py` script is provided to show how to use Detectron2 to train for a vehicle detection model.

To run the trainer script, open a docker container and run
```
docker run --shm-size 16G -p 8899:8888 --rm --gpus all -it -v [path/to/b3d]:/data -w /data detectron2 bash
# Use config_refined.json for better accuracy
python train.py -c configs/config_quick.json
```
The trained model will be saved to `output/model_final.pth`.

Alternatively, one can skip the training by downloading a pre-trained model as follows
```
python download.py --skip_videos --skip_images --pull_model
```
The trained model will be downloaded to `output/model_final.pth`.
Note that this model is trained with `config_refined.json`.

A test script `test.py` is provided to run the trained model on a sample image.

For instance, to use the test script on the image `vision/images/test/01_034_01.jpg`, run
```
# Use config_refined.json if the model_final.pth is generated by it
python test.py -i vision/images/test/01_034_01.jpg -c configs/config_quick.json
```
The result will be exported to `output/out.jpg`.

## Citation
The dataset will be documented in details in a paper that is to be published.
