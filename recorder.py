import os
import time
import argparse
import multiprocessing

from PIL import Image
import cv2

from text2audio import synthesize_text


def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('-o', '--output_path', default='dock', type=str,
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
    while True:
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break
        
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame.save('dock/input_frame.png')

            time.sleep(2)
            os.system('scp dock/input_frame.png gtang@txe1-login.mit.edu:/home/gridsan/gtang/safe-sight-dock')
            time.sleep(2)
            os.remove('dock/input_frame.png')
            time.sleep(10)
            os.system('scp gtang@txe1-login.mit.edu:/home/gridsan/gtang/safe-sight-dock dock/output.txt')
            time.sleep(2)

            with open('dock/output.txt') as fin:
                texts = fin.read().split('\n')
                synthesize_text(texts, 'dock')
                os.startfile('dock/result.mp3')

            os.remove('dock/output.txt')
            time.sleep(10)

    video_cap.release()


if __name__ == '__main__':
    main()

