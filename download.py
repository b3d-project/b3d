import argparse
import gdown
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset download script')
    parser.add_argument(
        '--skip_videos', action='store_true',
        help='Skip downloading videos')
    parser.add_argument(
        '--skip_images', action='store_true',
        help='Skip downloading annotated images')
    parser.add_argument(
        '--pull_model', action='store_true',
        help='Download the model trained with config_refined.json')
    return parser.parse_args()


def main(args):
    if not args.skip_videos:
        print('Downloading videos...')
        cmd = 'mkdir -p videos'
        subprocess.run(cmd.split(' '))
        gdown.download_folder(id='1UcVuWcqHdxq4D5O8M02o4zZKSvDRtEd6')
    if not args.skip_images:
        print('Downloading annotated images...')
        gdown.download(id='1v2Go30iTtbNDnOcmoSPueF4Mp93P5Lbg')
        cmd = 'unzip vision.zip'
        subprocess.run(cmd.split(' '))
        cmd = 'rm vision.zip'
        subprocess.run(cmd.split(' '))
    if args.pull_model:
        print('Downloading model...')
        gdown.download(id='17ZiwW_11q5oLldTCXuXjCpd8FQ7MjKaD')
        cmd = 'mkdir -p output'
        subprocess.run(cmd.split(' '))
        cmd = 'mv model_final.pth output/'
        subprocess.run(cmd.split(' '))


if __name__ == '__main__':
    main(parse_args())
