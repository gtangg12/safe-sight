import os
import time
import argparse
import multiprocessing

from PIL import Image
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('-o', '--output_path', default='data/example', type=str,
        help='')
    parser.add_argument('-n', '--num_frames', default=30, type=int,
        help='')
    parser.add_argument('--fps', default=30, type=int,
        help='')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    video_cap = cv2.VideoCapture(1)
    cap_prop = lambda x : int(video_cap.get(x))
    width, height = \
        cap_prop(cv2.CAP_PROP_FRAME_WIDTH), cap_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera dimensions: {}x{}".format(height, width))

    start = time.time()
    frames = []
    while True:
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break
        frames.append(frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    print ("Recording time taken : {0} seconds".format(time.time() - start))

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame.save(f'{args.output_path}/{str(i)}.png')

    video_cap.release()


if __name__ == '__main__':
    main()

