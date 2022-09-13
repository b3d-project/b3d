import argparse
import gdown
import subprocess

VIDEO_IDS = [
    '1n2smr1asZ03vUMtsTfv9DxYK9ufO7A2I',
    '1l7mU0SGmNNohn45j981zp-a9uiEN2kXQ',
    '1cFoX_kuHHRedSbL2bC1cY4xE3AzW1ZSP',
    '1A5e1PdLUX75JgjwZRDBBzSHTQxqW7kME',
    '14jrTJl0_tB5xHZNQxghP7VX5PFA4mHd6',
    '1wjE_gka-60WcOcfYZ0wP64YPOhM4c-0U',
    '1oK_C8L8eqRgs9CIzF1VhBYOhxou0PWPs',
    '1Mv2kZj1tWCo19HziyvwP_p2qfmvj2Sgk',
    '1Ib2Xo2_1zCQO9ozr-V72TQBcsg06PFkH',
    '1vGEKz_9ljxWmPGdqy8O4PEfHrYGRiYqJ',
    '1-gEmPQPHUyjusxpTpF5JX6ymEEGvpO0i',
    '1sBgvPBwCbvP_R7PNBg_okiifVFUvdQKe',
    '1NrnZHyZg2n-4jmpVbJSBMWo_krz5sSsL',
    '1shRvCElefMR_UCt5gha46qcxEFTb8QWi',
    '1PTdMt7QkVa9gYB1m--PyOZ_Bj2ufq9XM',
    '1VjQbi_ff1yOWKJJ-M0fgpfk9dWrEsE2f',
    '1WufoU3zdihfa2M-LiiPX_6vEsXQ5JHZf',
    '1I0WnjnErsnTH1x0b2rWDXqKssUmr4-OJ',
    '1dUbZIU1Bl6oH4QsNG6yUjnem-KUf4qTT',
    '1ablEWJWkbY6TPLkM4vBlohiD30XaK8LV',
]


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
        for video_id in VIDEO_IDS:
            gdown.download(id=video_id, output='videos/')
        subprocess.run(cmd.split(' '))
    if not args.skip_images:
        print('Downloading annotated images...')
        gdown.download(id='1fAF6oX6bukU6EZruWWh3DtPa8inT8syy')
        cmd = 'unzip vision.zip'
        subprocess.run(cmd.split(' '))
        cmd = 'rm vision.zip'
        subprocess.run(cmd.split(' '))
    if args.pull_model:
        print('Downloading model...')
        gdown.download(id='1ZDnOz0Opu7fhwXvcL5TMNyUEQc73dnGG')
        cmd = 'mkdir -p output'
        subprocess.run(cmd.split(' '))
        cmd = 'mv model_final.pth output/'
        subprocess.run(cmd.split(' '))


if __name__ == '__main__':
    main(parse_args())
