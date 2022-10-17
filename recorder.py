import os
import time
import subprocess
import argparse
import multiprocessing

from PIL import Image
import cv2

from text2audio import synthesize_text


def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('-o', '--output_path', default='dock', type=str,
        help='')
    parser.add_argument('-n', '--num_frames', default=15000, type=int,
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

    count = 0
    start = time.time()
    while True:
        success, frame = video_cap.read()
        if not success or count > args.num_frames - 1:
            break
        cv2.imshow('Frame', frame)
        count += 1
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print('Sending frame to server to process...')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame.save('dock/input_frame.png')

            time.sleep(2)
            os.system('scp dock/input_frame.png gtang@txe1-login.mit.edu:/home/gridsan/gtang/safe-sight/dock')
            time.sleep(2)
            os.remove('dock/input_frame.png')

            print('Waiting on server processing...')

            time.sleep(20)
            os.system('scp gtang@txe1-login.mit.edu:/home/gridsan/gtang/safe-sight/dock/output.txt dock/output.txt')
            time.sleep(2)

            print('Received output from server. Generating audio...')

            with open('dock/output.txt', 'r') as fin:
                texts = fin.read().split('\n')[:-1]
                print(texts)
                synthesize_text(texts, 'dock')
                return_code = subprocess.call(["afplay", 'dock/result.mp3'])

            print('Ready for next query...')

            os.remove('dock/output.txt')
            os.remove('dock/result.mp3')
            time.sleep(10)

    video_cap.release()


if __name__ == '__main__':
    main()

